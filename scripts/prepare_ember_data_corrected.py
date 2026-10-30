#!/usr/bin/env python3
"""
scripts/prepare_ember_data_corrected.py
---------------------------------------
Rigorous, unbiased streaming Reservoir Sampling across all 6 EMBER training shards.
Guarantees:
1. Every record in each shard has a mathematically equal selection probability.
2. Complete file traversal (no early exit bias).
3. Preserves exact proportional shard quotas for benign and malware.
4. Excludes unlabeled records (label == -1).
5. Exact 16,000 train / 4,000 val stratified split.
6. Asserts 0 train/val SHA-256 hash overlap.
7. Computes:
   - Modality A: 512-D Byte Histogram & Byte Entropy
   - Modality B: Text token sequences (Vocabulary fitted ONLY on train)
   - Modality C: 66-D Structural features (StandardScaler fitted ONLY on train)
   - Official EMBER 2,381 Features (exact character-sequence reproduction of sklearn 0.20)
"""

import os
import sys
import json
import math
import random
import argparse
from collections import Counter
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher
import ember.features as ef


def extract_modality_a(record):
    """512-D Byte Histogram and Byte Entropy."""
    hist = np.array(record.get('histogram', [0]*256), dtype=np.float32)
    h_sum = hist.sum()
    hist_norm = hist / (h_sum + 1e-7) if h_sum > 0 else hist

    ent = np.array(record.get('byteentropy', [0]*256), dtype=np.float32)
    e_sum = ent.sum()
    ent_norm = ent / (e_sum + 1e-7) if e_sum > 0 else ent

    out = np.concatenate([hist_norm, ent_norm]).astype(np.float32)
    assert not np.isnan(out).any() and not np.isinf(out).any()
    return out


def extract_modality_b_tokens(record):
    """API Imports, Exports, and Section Names."""
    tokens = []
    imports = record.get('imports', {})
    if isinstance(imports, dict):
        for dll, funcs in imports.items():
            dll_clean = str(dll).lower().replace('.dll', '').strip()
            if dll_clean:
                tokens.append(f"dll_{dll_clean}")
            if isinstance(funcs, list):
                for fn in funcs:
                    fn_s = str(fn).lower().strip()
                    if fn_s:
                        tokens.append(f"fn_{fn_s}")
    exports = record.get('exports', [])
    if isinstance(exports, list):
        for exp in exports:
            exp_s = str(exp).lower().strip()
            if exp_s:
                tokens.append(f"exp_{exp_s}")
    sec_obj = record.get('section', {})
    if isinstance(sec_obj, dict):
        entry = sec_obj.get('entry', '')
        if entry:
            tokens.append(f"entry_{str(entry).lower().strip()}")
        for sec in sec_obj.get('sections', []):
            if isinstance(sec, dict) and sec.get('name'):
                s_name = str(sec['name']).lower().strip().replace('.', '')
                tokens.append(f"sec_{s_name}")
    return tokens


def extract_modality_c(record):
    """66-D Structural PE Metrics."""
    vec = []
    gen = record.get('general', {}) if isinstance(record.get('general'), dict) else {}
    vec.extend([
        math.log1p(max(0, float(gen.get('size', 0)))),
        math.log1p(max(0, float(gen.get('vsize', 0)))),
        float(gen.get('has_debug', 0)),
        math.log1p(max(0, float(gen.get('exports', 0)))),
        math.log1p(max(0, float(gen.get('imports', 0)))),
        float(gen.get('has_relocations', 0)),
        float(gen.get('has_resources', 0)),
        float(gen.get('has_signature', 0)),
        float(gen.get('has_tls', 0)),
        math.log1p(max(0, float(gen.get('symbols', 0))))
    ])

    hdr = record.get('header', {}) if isinstance(record.get('header'), dict) else {}
    coff = hdr.get('coff', {}) if isinstance(hdr.get('coff'), dict) else {}
    chars = coff.get('characteristics', [])
    vec.append(float(len(chars) if isinstance(chars, list) else 0))
    machine = str(coff.get('machine', '')).upper()
    vec.append(1.0 if ('386' in machine or 'I386' in machine) else 0.0)
    vec.append(1.0 if ('AMD64' in machine or '64' in machine) else 0.0)

    opt = hdr.get('optional', {}) if isinstance(hdr.get('optional'), dict) else {}
    subsys = str(opt.get('subsystem', '')).upper()
    vec.append(1.0 if 'GUI' in subsys else 0.0)
    vec.append(1.0 if 'CUI' in subsys else 0.0)
    vec.extend([
        float(opt.get('major_image_version', 0)),
        float(opt.get('minor_image_version', 0)),
        float(opt.get('major_linker_version', 0)),
        math.log1p(max(0, float(opt.get('sizeof_code', 0)))),
        math.log1p(max(0, float(opt.get('sizeof_headers', 0)))),
        math.log1p(max(0, float(opt.get('sizeof_heap_commit', 0))))
    ])

    sec_obj = record.get('section', {}) if isinstance(record.get('section'), dict) else {}
    sections = sec_obj.get('sections', [])
    if isinstance(sections, list) and len(sections) > 0:
        entropies = [float(s.get('entropy', 0)) for s in sections if isinstance(s, dict)]
        raw_sizes = [float(s.get('size', 0)) for s in sections if isinstance(s, dict)]
        vsizes = [float(s.get('vsize', 0)) for s in sections if isinstance(s, dict)]
        n_exec = sum(1 for s in sections if isinstance(s, dict) and any('EXECUTE' in str(p) for p in s.get('props', [])))
        n_write = sum(1 for s in sections if isinstance(s, dict) and any('WRITE' in str(p) for p in s.get('props', [])))
        vec.extend([
            float(len(sections)),
            float(np.mean(entropies)) if entropies else 0.0,
            float(np.max(entropies)) if entropies else 0.0,
            math.log1p(max(0, sum(raw_sizes))),
            math.log1p(max(0, sum(vsizes))),
            float(n_exec),
            float(n_write)
        ])
    else:
        vec.extend([0.0] * 7)

    datadirs = record.get('datadirectories', [])
    for i in range(15):
        if isinstance(datadirs, list) and i < len(datadirs) and isinstance(datadirs[i], dict):
            sz = float(datadirs[i].get('size', 0))
            vec.extend([math.log1p(max(0, sz)), 1.0 if sz > 0 else 0.0])
        else:
            vec.extend([0.0, 0.0])

    strings = record.get('strings', {}) if isinstance(record.get('strings'), dict) else {}
    vec.extend([
        math.log1p(max(0, float(strings.get('numstrings', 0)))),
        float(strings.get('avlength', 0)),
        math.log1p(max(0, float(strings.get('printables', 0)))),
        float(strings.get('entropy', 0)),
        math.log1p(max(0, float(strings.get('paths', 0)))),
        math.log1p(max(0, float(strings.get('urls', 0)))),
        math.log1p(max(0, float(strings.get('registry', 0)))),
        math.log1p(max(0, float(strings.get('MZ', 0))))
    ])

    assert len(vec) == 66, f"Expected 66 structural features, got {len(vec)}"
    out = np.array(vec, dtype=np.float32)
    assert not np.isnan(out).any() and not np.isinf(out).any()
    return out


# Rigorous SectionInfo patch matching sklearn 0.20 character-sequence hashing
def patched_section_process(self, raw_obj):
    sections = raw_obj.get('sections', [])
    general = [
        len(sections),
        sum(1 for s in sections if s.get('size', 0) == 0),
        sum(1 for s in sections if s.get('name', '') == ''),
        sum(1 for s in sections if 'MEM_READ' in s.get('props', []) and 'MEM_EXECUTE' in s.get('props', [])),
        sum(1 for s in sections if 'MEM_WRITE' in s.get('props', []))
    ]
    section_sizes = [(s.get('name', ''), s.get('size', 0)) for s in sections]
    section_sizes_hashed = FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0]
    section_entropy = [(s.get('name', ''), s.get('entropy', 0)) for s in sections]
    section_entropy_hashed = FeatureHasher(50, input_type="pair").transform([section_entropy]).toarray()[0]
    section_vsize = [(s.get('name', ''), s.get('vsize', 0)) for s in sections]
    section_vsize_hashed = FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0]
    entry_val = raw_obj.get('entry', '')
    # Exact sklearn 0.20 character hashing trick:
    entry_name_hashed = FeatureHasher(50, input_type="string").transform([list(entry_val)]).toarray()[0]
    characteristics = [p for s in sections for p in s.get('props', []) if s.get('name', '') == entry_val]
    characteristics_hashed = FeatureHasher(50, input_type="string").transform([characteristics]).toarray()[0]

    return np.hstack([
        general, section_sizes_hashed, section_entropy_hashed, section_vsize_hashed, entry_name_hashed,
        characteristics_hashed
    ]).astype(np.float32)

ef.SectionInfo.process_raw_features = patched_section_process


def reservoir_sample_shard(file_path, target_b, target_m, rng):
    """
    Performs streaming Reservoir Sampling (Algorithm R) across an entire shard file.
    Every benign record has equal probability target_b / total_b.
    Every malware record has equal probability target_m / total_m.
    Unlabeled records (label == -1) are ignored.
    """
    res_b = []
    res_m = []
    count_b = 0
    count_m = 0
    total_lines = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            lbl = rec.get('label')
            if lbl == 0:
                count_b += 1
                if len(res_b) < target_b:
                    res_b.append(rec)
                else:
                    j = rng.randint(0, count_b - 1)
                    if j < target_b:
                        res_b[j] = rec
            elif lbl == 1:
                count_m += 1
                if len(res_m) < target_m:
                    res_m.append(rec)
                else:
                    j = rng.randint(0, count_m - 1)
                    if j < target_m:
                        res_m[j] = rec

    return res_b, res_m, count_b, count_m, total_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=r'c:\Users\imsto\Desktop\Final Year Paper\ember_dataset_2018_2\ember2018')
    parser.add_argument('--output_dir', type=str, default=r'c:\Users\imsto\Desktop\Final Year Paper\Old Repo Data\experiments\baseline_correction_2026_10_30\data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_train', type=int, default=16000)
    parser.add_argument('--n_val', type=int, default=4000)
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--vocab_size', type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 80)
    print("CORRECTED RESERVOIR SAMPLING PIPELINE ACROSS ALL 6 EMBER TRAINING SHARDS")
    print(f"Random Seed: {args.seed} | Train Target: {args.n_train:,} | Val Target: {args.n_val:,}")
    print(f"Dataset Path: {args.data_dir}")
    print("=" * 80)

    # Shard proportional quotas (audited from full 1,000,000 record scan)
    shard_b_avail = [50000, 52338, 46563, 47004, 48433, 55662]
    shard_m_avail = [0, 63713, 47044, 47309, 48413, 93521]
    tot_b = sum(shard_b_avail)
    tot_m = sum(shard_m_avail)

    total_samples = args.n_train + args.n_val
    target_b_total = total_samples // 2
    target_m_total = total_samples // 2

    b_quotas = [int(round(target_b_total * (c / tot_b))) for c in shard_b_avail]
    b_quotas[-1] += target_b_total - sum(b_quotas)

    m_quotas = [int(round(target_m_total * (c / tot_m))) for c in shard_m_avail]
    m_quotas[-1] += target_m_total - sum(m_quotas)

    rng = random.Random(args.seed)

    all_b = []
    all_m = []
    shard_manifest_stats = []

    for i in range(6):
        fname = f"train_features_{i}.jsonl"
        fpath = os.path.join(args.data_dir, fname)
        t_b = b_quotas[i]
        t_m = m_quotas[i]
        print(f"Reservoir sampling {fname} -> Target Benign: {t_b:>5,}, Target Malware: {t_m:>5,}...")
        res_b, res_m, n_b_seen, n_m_seen, total_lines = reservoir_sample_shard(fpath, t_b, t_m, rng)
        print(f"  [DONE] Streamed {total_lines:,} lines. Available: {n_b_seen:,} Benign, {n_m_seen:,} Malware.")
        print(f"  Sampled: {len(res_b):,} Benign, {len(res_m):,} Malware.")

        all_b.extend(res_b)
        all_m.extend(res_m)

        shard_manifest_stats.append({
            "shard_file": fname,
            "total_lines_streamed": total_lines,
            "available_benign": n_b_seen,
            "available_malware": n_m_seen,
            "quota_benign": t_b,
            "quota_malware": t_m,
            "sampled_benign": len(res_b),
            "sampled_malware": len(res_m)
        })

    assert len(all_b) == target_b_total, f"Expected {target_b_total} benign, got {len(all_b)}"
    assert len(all_m) == target_m_total, f"Expected {target_m_total} malware, got {len(all_m)}"

    print(f"\nTotal Reservoir Sampled: {len(all_b):,} Benign, {len(all_m):,} Malware.")

    # Stratified Train/Val Split
    rng.shuffle(all_b)
    rng.shuffle(all_m)

    n_tr_b = args.n_train // 2
    n_va_b = args.n_val // 2

    train_recs = all_b[:n_tr_b] + all_m[:n_tr_b]
    val_recs = all_b[n_tr_b:n_tr_b + n_va_b] + all_m[n_tr_b:n_tr_b + n_va_b]

    rng.shuffle(train_recs)
    rng.shuffle(val_recs)

    train_shas = [r['sha256'] for r in train_recs]
    val_shas = [r['sha256'] for r in val_recs]

    # Cryptographic overlap assertion
    overlap = set(train_shas).intersection(set(val_shas))
    print(f"Train/Val SHA-256 Hash Overlap: {len(overlap)} (MUST BE ZERO)")
    assert len(overlap) == 0, f"Critical Leakage: Found {len(overlap)} overlapping hashes!"

    y_train = np.array([int(r['label']) for r in train_recs], dtype=np.int64)
    y_val = np.array([int(r['label']) for r in val_recs], dtype=np.int64)

    print("\n--- Feature Extraction ---")
    # 1. Modality A (512-D)
    print("Extracting Modality A (512-D Byte Statistics)...")
    X_byte_tr = np.stack([extract_modality_a(r) for r in train_recs], axis=0)
    X_byte_va = np.stack([extract_modality_a(r) for r in val_recs], axis=0)

    # 2. Modality B (Token sequences, vocab fitted ONLY on train)
    print(f"Building vocabulary strictly on 16k train partition (Modality B, max_seq={args.max_seq_len})...")
    tr_toks = [extract_modality_b_tokens(r) for r in train_recs]
    va_toks = [extract_modality_b_tokens(r) for r in val_recs]

    tok_counts = Counter(t for sub in tr_toks for t in sub)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for t, _ in tok_counts.most_common(args.vocab_size - 2):
        vocab[t] = len(vocab)
    print(f"  Vocabulary size: {len(vocab):,} tokens.")

    def encode(toks):
        ids = [vocab.get(t, 1) for t in toks[:args.max_seq_len]]
        if len(ids) < args.max_seq_len:
            ids.extend([0] * (args.max_seq_len - len(ids)))
        return ids

    X_text_tr = np.array([encode(t) for t in tr_toks], dtype=np.int64)
    X_text_va = np.array([encode(t) for t in va_toks], dtype=np.int64)

    # 3. Modality C (66-D, scaler fitted ONLY on train)
    print("Extracting Modality C (66-D Structural PE Metrics)...")
    raw_struct_tr = np.stack([extract_modality_c(r) for r in train_recs], axis=0)
    raw_struct_va = np.stack([extract_modality_c(r) for r in val_recs], axis=0)

    scaler = StandardScaler()
    X_struct_tr = scaler.fit_transform(raw_struct_tr).astype(np.float32)
    X_struct_va = scaler.transform(raw_struct_va).astype(np.float32)

    # 4. Official EMBER 2,381 Features
    print("Extracting official EMBER 2,381-D features (with verified sklearn 0.20 SectionInfo)...")
    extractor = ef.PEFeatureExtractor(feature_version=2, print_feature_warning=False)
    X_ember_tr = np.array([extractor.process_raw_features(r) for r in train_recs], dtype=np.float32)
    X_ember_va = np.array([extractor.process_raw_features(r) for r in val_recs], dtype=np.float32)

    assert X_ember_tr.shape == (args.n_train, 2381)
    assert X_ember_va.shape == (args.n_val, 2381)
    assert np.isfinite(X_ember_tr).all(), "NaN/Inf in X_ember_tr"
    assert np.isfinite(X_ember_va).all(), "NaN/Inf in X_ember_va"

    # Save arrays
    tr_out = os.path.join(args.output_dir, "ember_train_16k_corrected.npz")
    va_out = os.path.join(args.output_dir, "ember_val_4k_corrected.npz")

    np.savez_compressed(
        tr_out,
        X_byte=X_byte_tr,
        X_text=X_text_tr,
        X_struct=X_struct_tr,
        X_ember2381=X_ember_tr,
        y=y_train,
        shas=np.array(train_shas)
    )
    np.savez_compressed(
        va_out,
        X_byte=X_byte_va,
        X_text=X_text_va,
        X_struct=X_struct_va,
        X_ember2381=X_ember_va,
        y=y_val,
        shas=np.array(val_shas)
    )

    print(f"\nSaved Train: {tr_out} ({os.path.getsize(tr_out)/1024/1024:.2f} MB)")
    print(f"Saved Val  : {va_out} ({os.path.getsize(va_out)/1024/1024:.2f} MB)")

    # Save complete sample manifest
    manifest = {
        "pipeline": "Corrected Reservoir Sampling across full shards",
        "random_seed": args.seed,
        "n_train": len(train_shas),
        "n_val": len(val_shas),
        "test_quarantined": True,
        "shard_statistics": shard_manifest_stats,
        "vocabulary_size": len(vocab),
        "feature_dimensions": {
            "modality_a_byte": 512,
            "modality_b_text": args.max_seq_len,
            "modality_c_struct": 66,
            "tabular_ember": 2381
        },
        "train_sha256_sample": train_shas[:10],
        "val_sha256_sample": val_shas[:10],
        "train_sha256_all": train_shas,
        "val_sha256_all": val_shas
    }

    manifest_path = os.path.join(args.output_dir, "sample_manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved Manifest: {manifest_path}")

    # Also save vocab for model inference
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f)
    print(f"Saved Vocabulary: {vocab_path}")
    print("\n[SUCCESS] Corrected data preparation complete.")


if __name__ == '__main__':
    main()

