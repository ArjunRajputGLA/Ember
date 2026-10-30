#!/usr/bin/env python3
"""
scripts/prepare_ember_data.py
------------------------------
Extracts, formats, and saves reproducible, leakage-free data splits
for Native EMBER Multimodal Learning.

Key Methodological Guarantees:
1. Shard-Balanced Sampling: Samples are drawn proportionally across ALL 6 training
   shards (train_features_0.jsonl through train_features_5.jsonl) to prevent temporal
   or partition bias.
2. Complete Test Set Quarantine: test_features.jsonl is NEVER touched or accessed.
3. Supervised Label Integrity: Label -1 (unlabeled samples) is strictly excluded.
4. Stratified Split: Exactly 16,000 train (8,000 benign, 8,000 malware) and
   4,000 validation (2,000 benign, 2,000 malware).
5. Cryptographic Sample Identification: Every sample is tracked by its SHA-256 hash.
   Train/Val hash intersection is strictly asserted to be empty (zero leakage).
6. Native Feature Extraction:
   - Modality A: 512-D Byte Statistics (histogram + byte-entropy)
   - Modality B: Textual metadata (API imports, exports, section names)
   - Modality C: 66-D Structural PE numerical metrics
7. Numerical Sanitation: Explicit checks asserting no NaN or infinite values.
"""

import os
import sys
import json
import math
import random
import argparse
from collections import Counter
import numpy as np


def extract_modality_a(record):
    """
    Modality A: Byte-level statistics (512 dimensions).
    - 256 byte histogram values normalized by sum.
    - 256 byte-entropy histogram values normalized by sum.
    """
    hist = record.get('histogram', [0] * 256)
    byte_ent = record.get('byteentropy', [0] * 256)

    hist_arr = np.array(hist, dtype=np.float32)
    h_sum = hist_arr.sum()
    hist_norm = hist_arr / (h_sum + 1e-7) if h_sum > 0 else hist_arr

    ent_arr = np.array(byte_ent, dtype=np.float32)
    e_sum = ent_arr.sum()
    ent_norm = ent_arr / (e_sum + 1e-7) if e_sum > 0 else ent_arr

    out = np.concatenate([hist_norm, ent_norm]).astype(np.float32)
    assert not np.isnan(out).any(), "NaN detected in Modality A"
    assert not np.isinf(out).any(), "Inf detected in Modality A"
    return out


def extract_modality_b(record):
    """
    Modality B: Textual API names, exports, section names.
    Preserves token boundaries.
    """
    tokens = []

    # 1. Imports
    imports = record.get('imports', {})
    if isinstance(imports, dict):
        for dll, funcs in imports.items():
            dll_clean = str(dll).lower().replace('.dll', '').strip()
            if dll_clean:
                tokens.append(f"dll_{dll_clean}")
            if isinstance(funcs, list):
                for fn in funcs:
                    fn_str = str(fn).lower().strip()
                    if fn_str:
                        tokens.append(f"fn_{fn_str}")

    # 2. Exports
    exports = record.get('exports', [])
    if isinstance(exports, list):
        for exp in exports:
            exp_str = str(exp).lower().strip()
            if exp_str:
                tokens.append(f"exp_{exp_str}")

    # 3. Section names
    sec_obj = record.get('section', {})
    if isinstance(sec_obj, dict):
        entry = sec_obj.get('entry', '')
        if entry:
            tokens.append(f"entry_{str(entry).lower().strip()}")
        sections = sec_obj.get('sections', [])
        if isinstance(sections, list):
            for sec in sections:
                if isinstance(sec, dict) and 'name' in sec and sec['name']:
                    sec_name = str(sec['name']).lower().strip().replace('.', '')
                    tokens.append(f"sec_{sec_name}")

    return " ".join(tokens)


def extract_modality_c(record):
    """
    Modality C: 66-D structural PE numerical metrics.
    No post-label metadata (no 'appeared', 'label', or 'sha256').
    Log1p transform applied to wide-range counters.
    """
    vec = []

    # 1. General PE info (10 features)
    gen = record.get('general', {}) if isinstance(record.get('general'), dict) else {}
    vec.append(math.log1p(max(0, float(gen.get('size', 0)))))
    vec.append(math.log1p(max(0, float(gen.get('vsize', 0)))))
    vec.append(float(gen.get('has_debug', 0)))
    vec.append(math.log1p(max(0, float(gen.get('exports', 0)))))
    vec.append(math.log1p(max(0, float(gen.get('imports', 0)))))
    vec.append(float(gen.get('has_relocations', 0)))
    vec.append(float(gen.get('has_resources', 0)))
    vec.append(float(gen.get('has_signature', 0)))
    vec.append(float(gen.get('has_tls', 0)))
    vec.append(math.log1p(max(0, float(gen.get('symbols', 0)))))

    # 2. Header info (11 features)
    header = record.get('header', {}) if isinstance(record.get('header'), dict) else {}
    coff = header.get('coff', {}) if isinstance(header.get('coff'), dict) else {}
    chars = coff.get('characteristics', [])
    vec.append(float(len(chars) if isinstance(chars, list) else 0))
    machine = str(coff.get('machine', '')).upper()
    vec.append(1.0 if '386' in machine or 'I386' in machine else 0.0)
    vec.append(1.0 if 'AMD64' in machine or '64' in machine else 0.0)

    opt = header.get('optional', {}) if isinstance(header.get('optional'), dict) else {}
    subsys = str(opt.get('subsystem', '')).upper()
    vec.append(1.0 if 'GUI' in subsys else 0.0)
    vec.append(1.0 if 'CUI' in subsys else 0.0)
    vec.append(float(opt.get('major_image_version', 0)))
    vec.append(float(opt.get('minor_image_version', 0)))
    vec.append(float(opt.get('major_linker_version', 0)))
    vec.append(math.log1p(max(0, float(opt.get('sizeof_code', 0)))))
    vec.append(math.log1p(max(0, float(opt.get('sizeof_headers', 0)))))
    vec.append(math.log1p(max(0, float(opt.get('sizeof_heap_commit', 0)))))

    # 3. Section aggregates (7 features)
    sec_obj = record.get('section', {}) if isinstance(record.get('section'), dict) else {}
    sections = sec_obj.get('sections', [])
    if isinstance(sections, list) and len(sections) > 0:
        sec_count = len(sections)
        entropies = [float(s.get('entropy', 0)) for s in sections if isinstance(s, dict)]
        raw_sizes = [float(s.get('size', 0)) for s in sections if isinstance(s, dict)]
        vsizes = [float(s.get('vsize', 0)) for s in sections if isinstance(s, dict)]
        n_exec = sum(1 for s in sections if isinstance(s, dict) and any('EXECUTE' in str(p) for p in s.get('props', [])))
        n_write = sum(1 for s in sections if isinstance(s, dict) and any('WRITE' in str(p) for p in s.get('props', [])))

        vec.append(float(sec_count))
        vec.append(float(np.mean(entropies)) if entropies else 0.0)
        vec.append(float(np.max(entropies)) if entropies else 0.0)
        vec.append(math.log1p(max(0, sum(raw_sizes))))
        vec.append(math.log1p(max(0, sum(vsizes))))
        vec.append(float(n_exec))
        vec.append(float(n_write))
    else:
        vec.extend([0.0] * 7)

    # 4. Data Directories (30 features: 15 tables x 2 metrics: size and flag)
    datadirs = record.get('datadirectories', [])
    for i in range(15):
        if isinstance(datadirs, list) and i < len(datadirs) and isinstance(datadirs[i], dict):
            size = float(datadirs[i].get('size', 0))
            vec.append(math.log1p(max(0, size)))
            vec.append(1.0 if size > 0 else 0.0)
        else:
            vec.extend([0.0, 0.0])

    # 5. String summary stats (8 features)
    strings = record.get('strings', {}) if isinstance(record.get('strings'), dict) else {}
    vec.append(math.log1p(max(0, float(strings.get('numstrings', 0)))))
    vec.append(float(strings.get('avlength', 0)))
    vec.append(math.log1p(max(0, float(strings.get('printables', 0)))))
    vec.append(float(strings.get('entropy', 0)))
    vec.append(math.log1p(max(0, float(strings.get('paths', 0)))))
    vec.append(math.log1p(max(0, float(strings.get('urls', 0)))))
    vec.append(math.log1p(max(0, float(strings.get('registry', 0)))))
    vec.append(math.log1p(max(0, float(strings.get('MZ', 0)))))

    assert len(vec) == 66, f"Expected 66 structural features, got {len(vec)}"
    out = np.array(vec, dtype=np.float32)
    assert not np.isnan(out).any(), "NaN detected in Modality C"
    assert not np.isinf(out).any(), "Inf detected in Modality C"
    return out


def collect_proportional_samples_across_all_shards(train_files, target_benign, target_malware, seed=42):
    """
    Collects target_benign and target_malware samples proportionally distributed
    across ALL 6 training shards to ensure complete temporal and shard representation.
    """
    rng = random.Random(seed)

    # Shard labeled counts (verified during full forensic audit):
    # train_0: 50,000 benign, 0 malware
    # train_1: 52,338 benign, 63,713 malware
    # train_2: 46,563 benign, 47,044 malware
    # train_3: 47,004 benign, 47,309 malware
    # train_4: 48,433 benign, 48,413 malware
    # train_5: 55,662 benign, 93,521 malware
    # Total available: 300,000 benign, 300,000 malware

    shard_benign_avail = [50000, 52338, 46563, 47004, 48433, 55662]
    shard_malware_avail = [0, 63713, 47044, 47309, 48413, 93521]

    total_avail_benign = sum(shard_benign_avail)
    total_avail_malware = sum(shard_malware_avail)

    # Allocate quotas per shard
    benign_quotas = [int(round(target_benign * (cnt / total_avail_benign))) for cnt in shard_benign_avail]
    # Adjust rounding discrepancy on last shard
    benign_quotas[-1] += target_benign - sum(benign_quotas)

    malware_quotas = [int(round(target_malware * (cnt / total_avail_malware))) for cnt in shard_malware_avail]
    malware_quotas[-1] += target_malware - sum(malware_quotas)

    print("Proportional quota allocation across shards:")
    for i, (b_q, m_q) in enumerate(zip(benign_quotas, malware_quotas)):
        print(f"  Shard train_features_{i}.jsonl -> Target Benign: {b_q:>5,}, Target Malware: {m_q:>5,}")

    collected_benign = []
    collected_malware = []

    for i, fpath in enumerate(train_files):
        target_b = benign_quotas[i]
        target_m = malware_quotas[i]
        b_count = 0
        m_count = 0

        fname = os.path.basename(fpath)
        print(f"Reading {fname}...")
        with open(fpath, 'r', encoding='utf-8') as fp:
            for line in fp:
                if b_count >= target_b and m_count >= target_m:
                    break
                try:
                    rec = json.loads(line)
                    lbl = rec.get('label')
                    if lbl == 0 and b_count < target_b:
                        collected_benign.append(rec)
                        b_count += 1
                    elif lbl == 1 and m_count < target_m:
                        collected_malware.append(rec)
                        m_count += 1
                except json.JSONDecodeError:
                    continue

    print(f"\nTotal sampled: {len(collected_benign):,} Benign, {len(collected_malware):,} Malware across all 6 shards.")
    return collected_benign, collected_malware


def build_arrays(records):
    shas = []
    labels = []
    mod_a = []
    mod_b = []
    mod_c = []

    for r in records:
        shas.append(r.get('sha256', ''))
        labels.append(int(r.get('label', 0)))
        mod_a.append(extract_modality_a(r))
        mod_b.append(extract_modality_b(r))
        mod_c.append(extract_modality_c(r))

    X_a = np.stack(mod_a, axis=0).astype(np.float32)
    X_b = np.array(mod_b, dtype=object)
    X_c = np.stack(mod_c, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return shas, y, X_a, X_b, X_c


def main():
    parser = argparse.ArgumentParser(description="Prepare EMBER native multimodal dataset splits")
    parser.add_argument('--data_dir', type=str, default=r'c:\Users\imsto\Desktop\Final Year Paper\ember_dataset_2018_2\ember2018')
    parser.add_argument('--output_dir', type=str, default=r'c:\Users\imsto\Desktop\Final Year Paper\Old Repo Data\data\prepared_splits')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_count', type=int, default=16000)
    parser.add_argument('--val_count', type=int, default=4000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 75)
    print("PREPARING SHARD-BALANCED LEAKAGE-FREE NATIVE EMBER SPLITS")
    print(f"Random Seed: {args.seed} | Train Target: {args.train_count:,} | Val Target: {args.val_count:,}")
    print(f"Dataset Path: {args.data_dir}")
    print("=" * 75)

    train_files = [os.path.join(args.data_dir, f"train_features_{i}.jsonl") for i in range(6)]
    total_samples = args.train_count + args.val_count
    target_b = total_samples // 2
    target_m = total_samples // 2

    # Collect across all shards
    benign_pool, malware_pool = collect_proportional_samples_across_all_shards(
        train_files, target_b, target_m, seed=args.seed
    )

    rng = random.Random(args.seed)
    rng.shuffle(benign_pool)
    rng.shuffle(malware_pool)

    n_train_b = args.train_count // 2
    n_val_b = args.val_count // 2

    train_records = benign_pool[:n_train_b] + malware_pool[:n_train_b]
    val_records = benign_pool[n_train_b:n_train_b + n_val_b] + malware_pool[n_train_b:n_train_b + n_val_b]

    rng.shuffle(train_records)
    rng.shuffle(val_records)

    # Extract arrays
    print("\nExtracting feature vectors and text sequences...")
    train_shas, y_train, X_a_train, X_b_train, X_c_train = build_arrays(train_records)
    val_shas, y_val, X_a_val, X_b_val, X_c_val = build_arrays(val_records)

    # Check for NaN / Inf
    assert not np.isnan(X_a_train).any() and not np.isnan(X_a_val).any(), "NaN found in Modality A"
    assert not np.isnan(X_c_train).any() and not np.isnan(X_c_val).any(), "NaN found in Modality C"

    # Integrity verification
    train_sha_set = set(train_shas)
    val_sha_set = set(val_shas)
    overlap = train_sha_set.intersection(val_sha_set)

    print("\n--- Integrity Verification ---")
    print(f"Train samples : {len(train_shas):,} (Benign: {(y_train == 0).sum():,}, Malware: {(y_train == 1).sum():,})")
    print(f"Val samples   : {len(val_shas):,} (Benign: {(y_val == 0).sum():,}, Malware: {(y_val == 1).sum():,})")
    print(f"Train/Val SHA-256 Overlap: {len(overlap)} (MUST BE ZERO)")
    assert len(overlap) == 0, f"Critical Leakage: Found {len(overlap)} overlapping hashes!"

    # Save manifest
    manifest = {
        "random_seed": args.seed,
        "n_train": len(train_shas),
        "n_val": len(val_shas),
        "test_quarantined": True,
        "train_sha256": train_shas,
        "val_sha256": val_shas
    }
    manifest_path = os.path.join(args.output_dir, "split_manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    # Save compressed arrays
    train_out = os.path.join(args.output_dir, "ember_train_16k.npz")
    val_out = os.path.join(args.output_dir, "ember_val_4k.npz")

    np.savez_compressed(train_out, X_byte=X_a_train, X_text=X_b_train, X_struct=X_c_train, y=y_train)
    np.savez_compressed(val_out, X_byte=X_a_val, X_text=X_b_val, X_struct=X_c_val, y=y_val)

    print(f"Saved: {train_out} ({os.path.getsize(train_out) / 1024 / 1024:.2f} MB)")
    print(f"Saved: {val_out} ({os.path.getsize(val_out) / 1024 / 1024:.2f} MB)")
    print(f"Saved manifest: {manifest_path}")
    print("\nData preparation complete and verified.")


if __name__ == '__main__':
    main()
