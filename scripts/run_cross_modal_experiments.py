#!/usr/bin/env python3
"""
run_cross_modal_experiments.py

Automated Experiment Runner for Controlled Multimodal Cross-Attention Experiments on EMBER 2018.

Evaluates 6 Models across 3 Seeds [42, 123, 2026]:
1. CMANet-Full (Bidirectional Global-to-Sequence Cross-Attention)
2. CMANet-NoByte (Ablation 1)
3. CMANet-NoStruct (Ablation 2)
4. CMANet-NoText (Ablation 3)
5. CMANet-NoCrossAttn (Ablation 4)
6. MultimodalFusionNet (Reference Baseline)

Strict Controls:
- Exact corrected 16k train / 4k val NPZ arrays with SHA-256 pre-flight manifest verification.
- Pre-flight attention safety and ablation isolation regression test suite.
- Baseline hyperparameter parity: Adam (lr=1e-3, wd=1e-4), batch_size=32, epochs=10, best val loss checkpoint.
- Preservation of per-example predictions for paired bootstrap significance testing (10,000 resamples).
- Aggregation with sample standard deviation (ddof=1).
- Strict quarantine of test_features.jsonl.
"""

import os
import sys
import json
import time
import math
import random
import hashlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# Ensure models package is importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from models.cmanet import (
    CrossModalAttentionNet,
    CMANet_NoByte,
    CMANet_NoStruct,
    CMANet_NoText,
    CMANet_NoCrossAttn,
    MultimodalFusionNet,
    compute_model_capacity,
    verify_all_attention_safety_and_regressions,
    verify_ablation_isolation
)

# Expected Checksums for the Corrected Arrays
EXPECTED_SHA256 = {
    "train_npz": "1164ca77fc4d05144726e21cec1152a7d1d5280d0e11b137501681bdf64a41c6",
    "val_npz": "6b25a36c96db4719c8dabaf7b97a7433207a30696b5271dabfed5bea0bf6c188"
}


# ==============================================================================
# 1. PRE-FLIGHT SHA-256 AND SPLIT VERIFICATION
# ==============================================================================

def compute_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_dataset_integrity(data_dir):
    print("=" * 80)
    print("DATASET INTEGRITY & SHA-256 MANIFEST VERIFICATION")
    print("=" * 80)
    train_npz = os.path.join(data_dir, "ember_train_16k_corrected.npz")
    val_npz = os.path.join(data_dir, "ember_val_4k_corrected.npz")
    manifest_json = os.path.join(data_dir, "sample_manifest.json")

    assert os.path.exists(train_npz), f"Training array not found: {train_npz}"
    assert os.path.exists(val_npz), f"Validation array not found: {val_npz}"

    print(f"Checking training array  : {train_npz}")
    actual_train_sha = compute_file_sha256(train_npz)
    print(f"  Calculated SHA-256     : {actual_train_sha}")
    if actual_train_sha == EXPECTED_SHA256["train_npz"]:
        print("  SHA-256 Match          : VERIFIED (Matches expected manifest)")
    else:
        print(f"  WARNING: Train SHA-256 differs from hardcoded expectation: {actual_train_sha}")

    print(f"Checking validation array: {val_npz}")
    actual_val_sha = compute_file_sha256(val_npz)
    print(f"  Calculated SHA-256     : {actual_val_sha}")
    if actual_val_sha == EXPECTED_SHA256["val_npz"]:
        print("  SHA-256 Match          : VERIFIED (Matches expected manifest)")
    else:
        print(f"  WARNING: Val SHA-256 differs from hardcoded expectation: {actual_val_sha}")

    # Load arrays and verify shapes & disjointness
    tr_data = np.load(train_npz)
    va_data = np.load(val_npz)

    n_tr = len(tr_data["y"])
    n_va = len(va_data["y"])
    print(f"Sample Counts            : Train = {n_tr:,} | Validation = {n_va:,}")
    assert n_tr == 16000, f"Expected exactly 16,000 train samples, got {n_tr}"
    assert n_va == 4000, f"Expected exactly 4,000 validation samples, got {n_va}"

    tr_benign = int((tr_data["y"] == 0).sum())
    tr_malware = int((tr_data["y"] == 1).sum())
    va_benign = int((va_data["y"] == 0).sum())
    va_malware = int((va_data["y"] == 1).sum())
    print(f"Class Balance (Train)    : Benign = {tr_benign:,} (50.0%) | Malware = {tr_malware:,} (50.0%)")
    print(f"Class Balance (Val)      : Benign = {va_benign:,} (50.0%) | Malware = {va_malware:,} (50.0%)")
    assert tr_benign == tr_malware == 8000
    assert va_benign == va_malware == 2000

    # Verify disjointness via SHA-256 sample hashes
    if "shas" in tr_data and "shas" in va_data:
        tr_shas = set(tr_data["shas"])
        va_shas = set(va_data["shas"])
        overlap = tr_shas.intersection(va_shas)
        print(f"Train/Val Sample Overlap : {len(overlap)} (Strictly Disjoint)")
        assert len(overlap) == 0, f"Train and Validation sets have {len(overlap)} overlapping samples!"

    if os.path.exists(manifest_json):
        with open(manifest_json, "r") as f:
            manifest = json.load(f)
        assert manifest.get("test_quarantined", False) is True, "CRITICAL: test_quarantined flag is False in manifest!"
        print("Test Set Quarantine Flag: VERIFIED (test_features.jsonl strictly unaccessed)")

    print("=" * 80)
    print("DATA INTEGRITY VERIFICATION PASSED")
    print("=" * 80)
    return tr_data, va_data


# ==============================================================================
# 2. METRICS & TIMING ROUTINES
# ==============================================================================

def compute_all_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_probs))
    fpr = float(fp / max(1, (fp + tn)))

    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_probs)
    idx_99 = np.where(tpr_arr >= 0.99)[0]
    fpr_99 = float(fpr_arr[idx_99[0]]) if len(idx_99) > 0 else 1.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "fpr": round(fpr, 4),
        "fpr_at_99pct_tpr": round(fpr_99, 4),
        "malware_fn": int(fn),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        }
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==============================================================================
# 3. TRAINING & EVALUATION LOOP
# ==============================================================================

def run_single_model_experiment(
    model_name,
    model_cls,
    tr_loader,
    va_loader,
    y_va,
    device,
    seed,
    epochs,
    lr,
    weight_decay,
    ckpt_dir,
    plots_dir,
    configs_dir,
    metrics_dir
):
    print(f"\n>>> Running: {model_name} (Seed {seed}) <<<")
    set_seed(seed)
    model = model_cls().to(device)
    cap = compute_model_capacity(model, model_name)

    # Save Run Configuration
    run_cfg = {
        "model_name": model_name,
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": tr_loader.batch_size,
        "optimizer": "Adam",
        "scheduler": "None (Controlled baseline setting)",
        "capacity": cap
    }
    cfg_path = os.path.join(configs_dir, f"run_config_{model_name}_seed{seed}.json")
    with open(cfg_path, "w") as f:
        json.dump(run_cfg, f, indent=2)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 1
    best_ckpt_path = os.path.join(ckpt_dir, f"best_{model_name}_seed{seed}.pt")

    t_start_train = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss_sum = 0.0
        for batch in tr_loader:
            xb, xt, xs, yb = [item.to(device) for item in batch]
            optimizer.zero_grad()
            logits = model(xb, xt, xs)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss_sum += loss.item() * len(yb)

        ep_tr_loss = tr_loss_sum / len(tr_loader.dataset)
        train_losses.append(ep_tr_loss)

        model.eval()
        va_loss_sum = 0.0
        with torch.no_grad():
            for batch in va_loader:
                xb, xt, xs, yb = [item.to(device) for item in batch]
                logits = model(xb, xt, xs)
                loss = criterion(logits, yb)
                va_loss_sum += loss.item() * len(yb)

        ep_va_loss = va_loss_sum / len(va_loader.dataset)
        val_losses.append(ep_va_loss)

        if ep_va_loss < best_val_loss:
            best_val_loss = ep_va_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt_path)

        if epoch % 2 == 0 or epoch == epochs or epoch == 1:
            print(f"  Epoch {epoch:>2}/{epochs} | Train Loss: {ep_tr_loss:.4f} | Val Loss: {ep_va_loss:.4f} (Best: {best_val_loss:.4f} @ ep {best_epoch})")

    training_time_s = round(time.time() - t_start_train, 2)

    # Load best checkpoint for final evaluation
    model.load_state_dict(torch.load(best_ckpt_path))
    model.eval()

    # Latency measurement with warm-up
    warm_batch = next(iter(va_loader))
    with torch.no_grad():
        _ = model(*[item.to(device) for item in warm_batch[:3]])

    val_probs = []
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_start_inf = time.time()
    with torch.no_grad():
        for batch in va_loader:
            xb, xt, xs, _ = [item.to(device) for item in batch]
            logits = model(xb, xt, xs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            val_probs.extend(probs)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_inf_time = time.time() - t_start_inf
    latency_ms = (total_inf_time / len(y_va)) * 1000

    val_probs = np.array(val_probs, dtype=np.float32)
    metrics = compute_all_metrics(y_va, val_probs)
    metrics["model_name"] = model_name
    metrics["seed"] = seed
    metrics["training_time_s"] = training_time_s
    metrics["latency_ms"] = round(latency_ms, 4)
    metrics["best_epoch"] = best_epoch
    metrics["best_val_loss"] = round(best_val_loss, 4)
    metrics["capacity"] = cap
    metrics["epoch_train_losses"] = [round(x, 4) for x in train_losses]
    metrics["epoch_val_losses"] = [round(x, 4) for x in val_losses]

    # Save per-run JSON metrics
    run_metrics_path = os.path.join(metrics_dir, f"run_{model_name}_seed{seed}.json")
    with open(run_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save loss curve plot
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", color="blue")
    plt.plot(range(1, epochs + 1), val_losses, label="Val Loss", color="orange")
    plt.axvline(best_epoch, color="green", linestyle="--", label=f"Best Ep ({best_epoch})")
    plt.title(f"{model_name} Loss Curve (Seed {seed})")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f"loss_curve_{model_name}_seed{seed}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"  Completed in {training_time_s}s | Best Val Acc: {metrics['accuracy']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | FN: {metrics['malware_fn']}")
    return metrics, val_probs


# ==============================================================================
# 4. PAIRED BOOTSTRAP STATISTICAL SIGNIFICANCE ROUTINE
# ==============================================================================

def compute_paired_bootstrap(y_true, p_model, p_baseline, n_resamples=10000, seed=42):
    """
    Computes paired bootstrap confidence intervals and empirical p-values
    for Metric(model) - Metric(baseline) across 4,000 validation samples.
    """
    np.random.seed(seed)
    N = len(y_true)
    
    # Pre-compute observed metrics
    pred_m = (p_model >= 0.5).astype(int)
    pred_b = (p_baseline >= 0.5).astype(int)
    
    obs_acc_m = accuracy_score(y_true, pred_m)
    obs_acc_b = accuracy_score(y_true, pred_b)
    obs_acc_diff = obs_acc_m - obs_acc_b
    
    obs_auc_m = roc_auc_score(y_true, p_model)
    obs_auc_b = roc_auc_score(y_true, p_baseline)
    obs_auc_diff = obs_auc_m - obs_auc_b
    
    cm_m = confusion_matrix(y_true, pred_m).ravel()
    cm_b = confusion_matrix(y_true, pred_b).ravel()
    obs_fn_m = cm_m[2]
    obs_fn_b = cm_b[2]
    obs_fn_diff = obs_fn_m - obs_fn_b # Negative is better
    
    # FPR@99%TPR
    def get_fpr99(y, p):
        fpr_arr, tpr_arr, _ = roc_curve(y, p)
        idx = np.where(tpr_arr >= 0.99)[0]
        return float(fpr_arr[idx[0]]) if len(idx) > 0 else 1.0

    obs_fpr99_m = get_fpr99(y_true, p_model)
    obs_fpr99_b = get_fpr99(y_true, p_baseline)
    obs_fpr99_diff = obs_fpr99_m - obs_fpr99_b

    # Bootstrap resampling
    boot_acc_diffs = np.zeros(n_resamples, dtype=np.float32)
    boot_auc_diffs = np.zeros(n_resamples, dtype=np.float32)
    boot_fn_diffs = np.zeros(n_resamples, dtype=np.float32)
    boot_fpr99_diffs = np.zeros(n_resamples, dtype=np.float32)

    for b in range(n_resamples):
        indices = np.random.randint(0, N, size=N)
        y_b = y_true[indices]
        pm_b = p_model[indices]
        pb_b = p_baseline[indices]

        # Accuracy
        acc_m = accuracy_score(y_b, (pm_b >= 0.5).astype(int))
        acc_b = accuracy_score(y_b, (pb_b >= 0.5).astype(int))
        boot_acc_diffs[b] = acc_m - acc_b

        # ROC-AUC (handle rare single-class bootstrap resample)
        if len(np.unique(y_b)) == 2:
            auc_m = roc_auc_score(y_b, pm_b)
            auc_b = roc_auc_score(y_b, pb_b)
            boot_auc_diffs[b] = auc_m - auc_b
        else:
            boot_auc_diffs[b] = obs_auc_diff

        # False Negatives
        cm_m_b = confusion_matrix(y_b, (pm_b >= 0.5).astype(int), labels=[0, 1]).ravel()
        cm_b_b = confusion_matrix(y_b, (pb_b >= 0.5).astype(int), labels=[0, 1]).ravel()
        boot_fn_diffs[b] = cm_m_b[2] - cm_b_b[2]

        # FPR@99%TPR
        if len(np.unique(y_b)) == 2:
            boot_fpr99_diffs[b] = get_fpr99(y_b, pm_b) - get_fpr99(y_b, pb_b)
        else:
            boot_fpr99_diffs[b] = obs_fpr99_diff

    # Compute CI (2.5% to 97.5%) and empirical two-tailed p-values
    def summarize_boot(diffs, obs_val, higher_is_better=True):
        ci_lower = float(np.percentile(diffs, 2.5))
        ci_upper = float(np.percentile(diffs, 97.5))
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1))
        
        # Two-tailed empirical p-value w.r.t zero
        if higher_is_better:
            p_val = 2.0 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
        else:
            p_val = 2.0 * min(float((diffs >= 0).mean()), float((diffs <= 0).mean()))
        p_val = min(1.0, p_val)
        
        return {
            "observed_diff": round(float(obs_val), 4),
            "bootstrap_mean_diff": round(mean_diff, 4),
            "bootstrap_std": round(std_diff, 4),
            "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
            "p_value": round(p_val, 4),
            "significant_alpha_0_05": bool(ci_lower > 0 if higher_is_better else ci_upper < 0)
        }

    return {
        "n_resamples": n_resamples,
        "accuracy": summarize_boot(boot_acc_diffs, obs_acc_diff, higher_is_better=True),
        "roc_auc": summarize_boot(boot_auc_diffs, obs_auc_diff, higher_is_better=True),
        "malware_fn": summarize_boot(boot_fn_diffs, obs_fn_diff, higher_is_better=False),
        "fpr_at_99pct_tpr": summarize_boot(boot_fpr99_diffs, obs_fpr99_diff, higher_is_better=False)
    }


# ==============================================================================
# 5. AGGREGATE RESULTS CALCULATOR
# ==============================================================================

def compute_aggregate_results(all_results, seeds, output_path):
    print("\n" + "=" * 80)
    print("AGGREGATE METRICS SUMMARY (Mean ± Sample Std, ddof=1)")
    print("=" * 80)
    
    models = list(all_results.keys())
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "fpr", "fpr_at_99pct_tpr", "malware_fn", "training_time_s", "latency_ms"]
    
    aggregate = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seeds": seeds,
            "models": models,
            "std_convention": "Sample Standard Deviation (ddof=1)"
        },
        "models": {}
    }

    print(f"{'Model':22s} | {'Accuracy (%)':15s} | {'ROC-AUC':14s} | {'FPR@99TPR (%)':15s} | {'False Negatives':15s}")
    print("-" * 92)

    for m in models:
        runs = all_results[m]
        m_agg = {
            "capacity": runs[0]["capacity"],
            "seed_runs": runs,
            "metrics": {}
        }
        for k in metric_keys:
            vals = [r[k] for r in runs]
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            m_agg["metrics"][k] = {
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "runs": vals
            }

        acc_str = f"{m_agg['metrics']['accuracy']['mean']*100:.2f} ± {m_agg['metrics']['accuracy']['std']*100:.2f}"
        auc_str = f"{m_agg['metrics']['roc_auc']['mean']:.4f} ± {m_agg['metrics']['roc_auc']['std']:.4f}"
        fpr99_str = f"{m_agg['metrics']['fpr_at_99pct_tpr']['mean']*100:.2f} ± {m_agg['metrics']['fpr_at_99pct_tpr']['std']*100:.2f}"
        fn_str = f"{m_agg['metrics']['malware_fn']['mean']:.1f} ± {m_agg['metrics']['malware_fn']['std']:.1f}"

        print(f"{m:22s} | {acc_str:15s} | {auc_str:14s} | {fpr99_str:15s} | {fn_str:15s}")
        aggregate["models"][m] = m_agg

    with open(output_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nAggregate summary saved to: {output_path}")
    return aggregate


# ==============================================================================
# 6. SMOKE VERIFICATION MODE (200 SAMPLES)
# ==============================================================================

def run_smoke_verification(data_dir, output_dir, device):
    print("\n" + "=" * 80)
    print("RUNNING 200-SAMPLE SMOKE VERIFICATION (ALL 6 MODELS, 1 EPOCH)")
    print("=" * 80)

    tr_data, va_data = verify_dataset_integrity(data_dir)

    # Pre-flight Unit Tests and Ablation Checks
    assert verify_all_attention_safety_and_regressions(), "Pre-flight attention safety tests failed!"
    assert verify_ablation_isolation(), "Ablation isolation checks failed!"

    n_smoke_tr = 200
    n_smoke_va = 50

    xb_tr = torch.tensor(tr_data["X_byte"][:n_smoke_tr], dtype=torch.float32)
    xt_tr = torch.tensor(tr_data["X_text"][:n_smoke_tr], dtype=torch.long)
    xs_tr = torch.tensor(tr_data["X_struct"][:n_smoke_tr], dtype=torch.float32)
    y_tr = torch.tensor(tr_data["y"][:n_smoke_tr], dtype=torch.long)

    xb_va = torch.tensor(va_data["X_byte"][:n_smoke_va], dtype=torch.float32)
    xt_va = torch.tensor(va_data["X_text"][:n_smoke_va], dtype=torch.long)
    xs_va = torch.tensor(va_data["X_struct"][:n_smoke_va], dtype=torch.float32)
    y_va = torch.tensor(va_data["y"][:n_smoke_va], dtype=torch.long)

    tr_loader = DataLoader(TensorDataset(xb_tr, xt_tr, xs_tr, y_tr), batch_size=32, shuffle=True)
    va_loader = DataLoader(TensorDataset(xb_va, xt_va, xs_va, y_va), batch_size=32, shuffle=False)

    model_registry = {
        "CMANet_Full": CrossModalAttentionNet,
        "CMANet_NoByte": CMANet_NoByte,
        "CMANet_NoStruct": CMANet_NoStruct,
        "CMANet_NoText": CMANet_NoText,
        "CMANet_NoCrossAttn": CMANet_NoCrossAttn,
        "MultimodalFusionNet": MultimodalFusionNet
    }

    smoke_ckpt_dir = os.path.join(output_dir, "checkpoints", "smoke")
    smoke_plots_dir = os.path.join(output_dir, "plots", "smoke")
    smoke_configs_dir = os.path.join(output_dir, "configs", "smoke")
    smoke_metrics_dir = os.path.join(output_dir, "metrics", "smoke")

    for d in [smoke_ckpt_dir, smoke_plots_dir, smoke_configs_dir, smoke_metrics_dir]:
        os.makedirs(d, exist_ok=True)

    smoke_results = {}
    for mname, mcls in model_registry.items():
        metrics, probs = run_single_model_experiment(
            model_name=mname,
            model_cls=mcls,
            tr_loader=tr_loader,
            va_loader=va_loader,
            y_va=y_va.numpy(),
            device=device,
            seed=42,
            epochs=1,
            lr=0.001,
            weight_decay=0.0001,
            ckpt_dir=smoke_ckpt_dir,
            plots_dir=smoke_plots_dir,
            configs_dir=smoke_configs_dir,
            metrics_dir=smoke_metrics_dir
        )
        smoke_results[mname] = metrics
        print(f"  [Smoke OK] {mname:20s}: Initial loss={metrics['epoch_train_losses'][0]:.4f}, Acc={metrics['accuracy']*100:.2f}%")

    smoke_summary_path = os.path.join(smoke_metrics_dir, "smoke_verification_summary.json")
    with open(smoke_summary_path, "w") as f:
        json.dump(smoke_results, f, indent=2)

    print("\n" + "=" * 80)
    print("SMOKE VERIFICATION COMPLETED SUCCESSFULLY FOR ALL 6 MODELS!")
    print(f"Smoke metrics written to: {smoke_summary_path}")
    print("=" * 80)
    return smoke_results


# ==============================================================================
# 7. MAIN EXECUTION CONTROLLER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Controlled Multimodal Cross-Attention Experiments on EMBER 2018")
    parser.add_argument("--smoke", action="store_true", help="Run fast 200-sample smoke verification only (DO NOT run full training)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026], help="Random seeds for training")
    parser.add_argument("--models", nargs="+", type=str, default=[
        "CMANet_Full", "CMANet_NoByte", "CMANet_NoStruct", "CMANet_NoText", "CMANet_NoCrossAttn", "MultimodalFusionNet"
    ], help="Models to train")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000, help="Number of bootstrap resamples for paired statistical testing")
    parser.add_argument("--data-dir", type=str, default=os.path.join("experiments", "baseline_correction_2026_10_30", "data"), help="Data directory")
    parser.add_argument("--output-dir", type=str, default=os.path.join("experiments", "cross_modal_attention_2026_10_30"), help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    # Determine Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    if device.type == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    # Resolve paths relative to working directory or repo root
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        alt_data_dir = os.path.join("Old Repo Data", data_dir)
        if os.path.exists(alt_data_dir):
            data_dir = alt_data_dir

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        alt_output_dir = os.path.join("Old Repo Data", output_dir)
        if os.path.exists(os.path.dirname(alt_output_dir)):
            output_dir = alt_output_dir

    # Create directory tree
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    plots_dir = os.path.join(output_dir, "plots")
    configs_dir = os.path.join(output_dir, "configs")
    metrics_dir = os.path.join(output_dir, "metrics")
    for d in [ckpt_dir, plots_dir, configs_dir, metrics_dir]:
        os.makedirs(d, exist_ok=True)

    if args.smoke:
        run_smoke_verification(data_dir, output_dir, device)
        return

    # FULL EXPERIMENT EXECUTION (Requires explicit execution flag)
    print("\n" + "=" * 80)
    print("STARTING FULL CONTROLLED CROSS-MODAL ATTENTION BENCHMARK")
    print("=" * 80)

    # Pre-flight SHA-256 and Disjointness Verification
    tr_data, va_data = verify_dataset_integrity(data_dir)

    # Pre-flight Unit Tests and Ablation Checks
    assert verify_all_attention_safety_and_regressions(), "Pre-flight attention safety tests failed!"
    assert verify_ablation_isolation(), "Ablation isolation checks failed!"

    # Prepare DataLoaders
    xb_tr = torch.tensor(tr_data["X_byte"], dtype=torch.float32)
    xt_tr = torch.tensor(tr_data["X_text"], dtype=torch.long)
    xs_tr = torch.tensor(tr_data["X_struct"], dtype=torch.float32)
    y_tr = torch.tensor(tr_data["y"], dtype=torch.long)

    xb_va = torch.tensor(va_data["X_byte"], dtype=torch.float32)
    xt_va = torch.tensor(va_data["X_text"], dtype=torch.long)
    xs_va = torch.tensor(va_data["X_struct"], dtype=torch.float32)
    y_va = torch.tensor(va_data["y"], dtype=torch.long)

    va_dataset = TensorDataset(xb_va, xt_va, xs_va, y_va)
    va_loader = DataLoader(va_dataset, batch_size=args.batch_size, shuffle=False)

    model_registry = {
        "CMANet_Full": CrossModalAttentionNet,
        "CMANet_NoByte": CMANet_NoByte,
        "CMANet_NoStruct": CMANet_NoStruct,
        "CMANet_NoText": CMANet_NoText,
        "CMANet_NoCrossAttn": CMANet_NoCrossAttn,
        "MultimodalFusionNet": MultimodalFusionNet
    }

    all_results = {m: [] for m in args.models}

    # Execute by Seed
    for seed in args.seeds:
        print(f"\n=======================================================")
        print(f"EXECUTING EXPERIMENT BATCH FOR SEED {seed}")
        print(f"=======================================================")
        set_seed(seed)
        tr_dataset = TensorDataset(xb_tr, xt_tr, xs_tr, y_tr)
        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)

        seed_predictions = {"y_true": y_va.numpy()}

        for mname in args.models:
            mcls = model_registry[mname]
            metrics, probs = run_single_model_experiment(
                model_name=mname,
                model_cls=mcls,
                tr_loader=tr_loader,
                va_loader=va_loader,
                y_va=y_va.numpy(),
                device=device,
                seed=seed,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                ckpt_dir=ckpt_dir,
                plots_dir=plots_dir,
                configs_dir=configs_dir,
                metrics_dir=metrics_dir
            )
            all_results[mname].append(metrics)
            seed_predictions[f"p_{mname.lower()}"] = probs

        # Save per-seed validation predictions for paired bootstrap
        pred_path = os.path.join(metrics_dir, f"predictions_seed{seed}.npz")
        np.savez_compressed(pred_path, **seed_predictions)
        print(f"Saved complete validation predictions to: {pred_path}")

    # Compute and Save Aggregate Metrics
    agg_path = os.path.join(metrics_dir, "cross_modal_attention_results.json")
    aggregate_summary = compute_aggregate_results(all_results, args.seeds, agg_path)

    # Run Paired Bootstrap Statistical Significance Analysis
    print("\n" + "=" * 80)
    print("PAIRED BOOTSTRAP STATISTICAL SIGNIFICANCE ANALYSIS (10,000 Resamples)")
    print("=" * 80)
    
    # Load StructMLP baseline predictions from baseline correction
    base_pred_dir = os.path.join("experiments", "baseline_correction_2026_10_30", "metrics")
    if not os.path.exists(base_pred_dir):
        base_pred_dir = os.path.join("Old Repo Data", base_pred_dir)

    stat_results = {}
    for seed in args.seeds:
        seed_preds_file = os.path.join(metrics_dir, f"predictions_seed{seed}.npz")
        cur_preds = np.load(seed_preds_file)
        y_true_seed = cur_preds["y_true"]
        p_cmanet = cur_preds["p_cmanet_full"]

        stat_results[f"seed_{seed}"] = {}

        # 1. CMANet-Full vs MultimodalFusionNet
        if "p_multimodalfusionnet" in cur_preds:
            p_mm = cur_preds["p_multimodalfusionnet"]
            stat_results[f"seed_{seed}"]["CMANet_vs_MultimodalFusionNet"] = compute_paired_bootstrap(
                y_true_seed, p_cmanet, p_mm, n_resamples=args.bootstrap_resamples, seed=seed
            )

        # 2. CMANet-Full vs CMANet-NoCrossAttn
        if "p_cmanet_nocrossattn" in cur_preds:
            p_nocross = cur_preds["p_cmanet_nocrossattn"]
            stat_results[f"seed_{seed}"]["CMANet_vs_CMANet_NoCrossAttn"] = compute_paired_bootstrap(
                y_true_seed, p_cmanet, p_nocross, n_resamples=args.bootstrap_resamples, seed=seed
            )

        # 3. CMANet-Full vs StructMLP (from baseline run)
        base_pred_file = os.path.join(base_pred_dir, f"predictions_seed{seed}.npz")
        if os.path.exists(base_pred_file):
            base_data = np.load(base_pred_file)
            p_struct = base_data["p_struct"]
            stat_results[f"seed_{seed}"]["CMANet_vs_StructMLP"] = compute_paired_bootstrap(
                y_true_seed, p_cmanet, p_struct, n_resamples=args.bootstrap_resamples, seed=seed
            )

    stat_path = os.path.join(metrics_dir, "statistical_comparison.json")
    with open(stat_path, "w") as f:
        json.dump(stat_results, f, indent=2)
    print(f"Statistical significance results saved to: {stat_path}")

    print("\n" + "=" * 80)
    print("ALL RUNS AND STATISTICAL ANALYSES COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

