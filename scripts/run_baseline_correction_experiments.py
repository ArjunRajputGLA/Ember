#!/usr/bin/env python3
"""
scripts/run_baseline_correction_experiments.py
----------------------------------------------
Executes the five fair baseline experiments on the corrected EMBER dataset:
1. Fresh LightGBM (2,381 features)
2. ByteMLP (512-D)
3. MaskedTextCNN (256-D API token sequence)
4. StructMLP (66-D PE structural metrics)
5. MultimodalFusionNet (joint 320-D representation)

Controlled multi-seed evaluation:
- Seeds: [42, 123, 2026]
- Dataset split held strictly fixed (seed 42) to isolate model initialization variability
- Reports per-run metrics and aggregate Mean ± Std Dev
- Saves per-epoch loss curves, confusion matrices, predictions, and checkpoints
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)


# ------------------------------------------------------------------------------
# Model Architectures (Strictly Preserving Initial Hyperparameters)
# ------------------------------------------------------------------------------

class ByteMLP(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x_byte):
        return self.net(x_byte)


class MaskedTextCNN(nn.Module):
    def __init__(self, vocab_sz=10000, emb_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(emb_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.clf = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
    def forward(self, x_text):
        mask = (x_text != 0).float()
        h = self.emb(x_text).transpose(1, 2)
        h1 = self.relu(self.conv1(h)) * mask.unsqueeze(1)
        h2 = self.relu(self.conv2(h1)) * mask.unsqueeze(1)
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = h2.sum(dim=-1) / valid_counts
        return self.clf(pooled)


class StructMLP(nn.Module):
    def __init__(self, in_dim=66):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x_struct):
        return self.net(x_struct)


class MultimodalFusionNet(nn.Module):
    def __init__(self, vocab_sz=10000, byte_dim=512, struct_dim=66, text_emb_dim=128):
        super().__init__()
        self.byte_enc = nn.Sequential(nn.Linear(byte_dim, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.struct_enc = nn.Sequential(nn.Linear(struct_dim, 64), nn.BatchNorm1d(64), nn.ReLU())
        self.text_emb = nn.Embedding(vocab_sz, text_emb_dim, padding_idx=0)
        self.text_conv1 = nn.Conv1d(text_emb_dim, 128, kernel_size=3, padding=1)
        self.text_conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Joint 320-D representation: 128 + 128 + 64
        self.classifier = nn.Sequential(
            nn.Linear(320, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x_byte, x_text, x_struct):
        h_b = self.byte_enc(x_byte)
        h_s = self.struct_enc(x_struct)

        mask = (x_text != 0).float()
        t_h = self.text_emb(x_text).transpose(1, 2)
        t_h1 = self.relu(self.text_conv1(t_h)) * mask.unsqueeze(1)
        t_h2 = self.relu(self.text_conv2(t_h1)) * mask.unsqueeze(1)
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        h_t = t_h2.sum(dim=-1) / valid_counts

        fused = torch.cat([h_b, h_t, h_s], dim=1)
        return self.classifier(fused)


# ------------------------------------------------------------------------------
# Consistent Metrics & Timing Routine
# ------------------------------------------------------------------------------

def compute_all_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred))
    rec = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
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


# ------------------------------------------------------------------------------
# Training Routines
# ------------------------------------------------------------------------------

def run_lightgbm_experiment(X_tr, y_tr, X_va, y_va, seed, ckpt_dir):
    print(f"\n--- Fresh LightGBM (Seed {seed}) ---")
    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        n_jobs=-1,
        verbose=-1
    )
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    tr_time = round(time.time() - t0, 2)

    # Standardized timing: warm up, then time full validation set
    _ = clf.predict_proba(X_va[:100])
    t1 = time.time()
    probs = clf.predict_proba(X_va)[:, 1]
    latency = (time.time() - t1) / len(y_va) * 1000

    metrics = compute_all_metrics(y_va, probs)
    metrics["training_time_s"] = tr_time
    metrics["latency_ms"] = round(latency, 4)
    metrics["seed"] = seed

    ckpt_path = os.path.join(ckpt_dir, f"fresh_lightgbm_seed{seed}.txt")
    clf.booster_.save_model(ckpt_path)

    return metrics, probs


def run_neural_experiment(model_name, model_fn, tr_loader, va_loader, y_va, device, seed, epochs, lr, ckpt_dir, plots_dir):
    print(f"\n--- {model_name} (Seed {seed}) ---")
    set_seed(seed)
    model = model_fn().to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 1
    best_ckpt = os.path.join(ckpt_dir, f"best_{model_name.replace(' ', '_')}_seed{seed}.pt")

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss_sum = 0.0
        for batch in tr_loader:
            inputs = [item.to(device) for item in batch[:-1]]
            y_b = batch[-1].to(device)
            opt.zero_grad()
            out = model(*inputs)
            loss = crit(out, y_b)
            loss.backward()
            opt.step()
            tr_loss_sum += loss.item() * len(y_b)
        ep_tr = tr_loss_sum / len(tr_loader.dataset)
        train_losses.append(ep_tr)

        model.eval()
        va_loss_sum = 0.0
        with torch.no_grad():
            for batch in va_loader:
                inputs = [item.to(device) for item in batch[:-1]]
                y_b = batch[-1].to(device)
                va_loss_sum += crit(model(*inputs), y_b).item() * len(y_b)
        ep_va = va_loss_sum / len(va_loader.dataset)
        val_losses.append(ep_va)

        if ep_va < best_val_loss:
            best_val_loss = ep_va
            best_epoch = ep
            torch.save(model.state_dict(), best_ckpt)

        if ep % 2 == 0 or ep == epochs:
            print(f"  Epoch {ep:>2}/{epochs} | Train Loss: {ep_tr:.4f} | Val Loss: {ep_va:.4f} (Best: {best_val_loss:.4f} @ ep {best_epoch})")

    tr_time = round(time.time() - t0, 2)

    # Load best checkpoint
    model.load_state_dict(torch.load(best_ckpt))
    model.eval()

    # Consistent latency measurement
    warm_batch = next(iter(va_loader))
    with torch.no_grad():
        _ = model(*[item.to(device) for item in warm_batch[:-1]])

    val_probs = []
    t1 = time.time()
    with torch.no_grad():
        for batch in va_loader:
            inputs = [item.to(device) for item in batch[:-1]]
            logits = model(*inputs)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            val_probs.extend(p)
    latency = (time.time() - t1) / len(y_va) * 1000

    val_probs = np.array(val_probs, dtype=np.float32)
    metrics = compute_all_metrics(y_va, val_probs)
    metrics["training_time_s"] = tr_time
    metrics["latency_ms"] = round(latency, 4)
    metrics["best_epoch"] = best_epoch
    metrics["best_val_loss"] = round(best_val_loss, 4)
    metrics["seed"] = seed
    metrics["epoch_train_losses"] = [round(x, 4) for x in train_losses]
    metrics["epoch_val_losses"] = [round(x, 4) for x in val_losses]

    # Save loss plot for seed 42
    if seed == 42:
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue')
        plt.plot(range(1, epochs + 1), val_losses, label='Val Loss', color='orange')
        plt.axvline(best_epoch, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
        plt.title(f"{model_name} Training Curve (Seed {seed})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"loss_{model_name.replace(' ', '_')}.png"), dpi=150)
        plt.close()

    return metrics, val_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=r'c:\Users\imsto\Desktop\Final Year Paper\Old Repo Data\experiments\baseline_correction_2026_10_30\data')
    parser.add_argument('--exp_dir', type=str, default=r'c:\Users\imsto\Desktop\Final Year Paper\Old Repo Data\experiments\baseline_correction_2026_10_30')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    metrics_dir = os.path.join(args.exp_dir, "metrics")
    checkpoints_dir = os.path.join(args.exp_dir, "checkpoints")
    plots_dir = os.path.join(args.exp_dir, "plots")
    configs_dir = os.path.join(args.exp_dir, "configs")
    logs_dir = os.path.join(args.exp_dir, "logs")

    for d in [metrics_dir, checkpoints_dir, plots_dir, configs_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 80)
    print("RUNNING EMBER BASELINE CORRECTION EXPERIMENTS ACROSS 3 SEEDS")
    print("Seeds: [42, 123, 2026]")
    print(f"Data Source: {args.data_dir}")
    print("=" * 80)

    tr_data = np.load(os.path.join(args.data_dir, "ember_train_16k_corrected.npz"))
    va_data = np.load(os.path.join(args.data_dir, "ember_val_4k_corrected.npz"))

    y_tr = tr_data['y']
    y_va = va_data['y']
    X_byte_tr, X_byte_va = tr_data['X_byte'], va_data['X_byte']
    X_text_tr, X_text_va = tr_data['X_text'], va_data['X_text']
    X_struct_tr, X_struct_va = tr_data['X_struct'], va_data['X_struct']
    X_ember_tr, X_ember_va = tr_data['X_ember2381'], va_data['X_ember2381']

    # Load vocab size
    with open(os.path.join(args.data_dir, "vocab.json"), 'r') as f:
        vocab = json.load(f)
    vocab_sz = len(vocab)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Active Compute Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")
    else:
        print(f"CPU Mode (Torch threads: {torch.get_num_threads()})")

    # Prepare Tensor DataLoaders
    t_b_tr = torch.tensor(X_byte_tr, dtype=torch.float32)
    t_b_va = torch.tensor(X_byte_va, dtype=torch.float32)
    t_t_tr = torch.tensor(X_text_tr, dtype=torch.long)
    t_t_va = torch.tensor(X_text_va, dtype=torch.long)
    t_s_tr = torch.tensor(X_struct_tr, dtype=torch.float32)
    t_s_va = torch.tensor(X_struct_va, dtype=torch.float32)
    t_y_tr = torch.tensor(y_tr, dtype=torch.long)
    t_y_va = torch.tensor(y_va, dtype=torch.long)

    ld_byte_tr = DataLoader(TensorDataset(t_b_tr, t_y_tr), batch_size=args.batch_size, shuffle=True)
    ld_byte_va = DataLoader(TensorDataset(t_b_va, t_y_va), batch_size=args.batch_size, shuffle=False)

    ld_text_tr = DataLoader(TensorDataset(t_t_tr, t_y_tr), batch_size=args.batch_size, shuffle=True)
    ld_text_va = DataLoader(TensorDataset(t_t_va, t_y_va), batch_size=args.batch_size, shuffle=False)

    ld_struct_tr = DataLoader(TensorDataset(t_s_tr, t_y_tr), batch_size=args.batch_size, shuffle=True)
    ld_struct_va = DataLoader(TensorDataset(t_s_va, t_y_va), batch_size=args.batch_size, shuffle=False)

    ld_mm_tr = DataLoader(TensorDataset(t_b_tr, t_t_tr, t_s_tr, t_y_tr), batch_size=args.batch_size, shuffle=True)
    ld_mm_va = DataLoader(TensorDataset(t_b_va, t_t_va, t_s_va, t_y_va), batch_size=args.batch_size, shuffle=False)

    SEEDS = [42, 123, 2026]
    all_runs = {}

    for model_name in ["Fresh_LightGBM", "ByteMLP", "MaskedTextCNN", "StructMLP", "MultimodalFusionNet"]:
        all_runs[model_name] = []

    # Model evaluation across 3 seeds
    for seed in SEEDS:
        print(f"\n{'#' * 30} EXECUTING SEED {seed} {'#' * 30}")

        # 1. Fresh LightGBM
        m_lgb, p_lgb = run_lightgbm_experiment(X_ember_tr, y_tr, X_ember_va, y_va, seed, checkpoints_dir)
        all_runs["Fresh_LightGBM"].append(m_lgb)

        # 2. ByteMLP
        m_byte, p_byte = run_neural_experiment("ByteMLP", lambda: ByteMLP(), ld_byte_tr, ld_byte_va, y_va, device, seed, args.epochs, args.lr, checkpoints_dir, plots_dir)
        all_runs["ByteMLP"].append(m_byte)

        # 3. MaskedTextCNN
        m_text, p_text = run_neural_experiment("MaskedTextCNN", lambda: MaskedTextCNN(vocab_sz), ld_text_tr, ld_text_va, y_va, device, seed, args.epochs, args.lr, checkpoints_dir, plots_dir)
        all_runs["MaskedTextCNN"].append(m_text)

        # 4. StructMLP
        m_struct, p_struct = run_neural_experiment("StructMLP", lambda: StructMLP(), ld_struct_tr, ld_struct_va, y_va, device, seed, args.epochs, args.lr, checkpoints_dir, plots_dir)
        all_runs["StructMLP"].append(m_struct)

        # 5. MultimodalFusionNet
        m_mm, p_mm = run_neural_experiment("MultimodalFusionNet", lambda: MultimodalFusionNet(vocab_sz), ld_mm_tr, ld_mm_va, y_va, device, seed, args.epochs, args.lr, checkpoints_dir, plots_dir)
        all_runs["MultimodalFusionNet"].append(m_mm)

        if seed == 42:
            # Save predictions for primary seed
            np.savez_compressed(
                os.path.join(metrics_dir, "predictions_seed42.npz"),
                y_true=y_va,
                p_lgb=p_lgb,
                p_byte=p_byte,
                p_text=p_text,
                p_struct=p_struct,
                p_mm=p_mm
            )

            # Generate combined ROC Curve plot
            plt.figure(figsize=(8, 6))
            for m_key, p_arr in [("Fresh LightGBM", p_lgb), ("ByteMLP", p_byte), ("MaskedTextCNN", p_text), ("StructMLP", p_struct), ("MultimodalFusionNet", p_mm)]:
                fpr_c, tpr_c, _ = roc_curve(y_va, p_arr)
                auc_c = roc_auc_score(y_va, p_arr)
                plt.plot(fpr_c, tpr_c, label=f"{m_key} (AUC = {auc_c:.4f})")
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance')
            plt.xlabel("False Positive Rate (FPR)")
            plt.ylabel("True Positive Rate (TPR)")
            plt.title("Validation ROC Curves (Corrected Reservoir Sampling - Seed 42)")
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "combined_roc_curves_corrected.png"), dpi=150)
            plt.close()

    # Aggregate Statistics (Mean ± Std)
    summary_table = []
    print("\n" + "=" * 90)
    print("CONSOLIDATED METRICS: MEAN ± STD ACROSS 3 RUNS (Seeds 42, 123, 2026)")
    print("=" * 90)

    for model_name, runs in all_runs.items():
        accs = [r['accuracy'] for r in runs]
        precs = [r['precision'] for r in runs]
        recs = [r['recall'] for r in runs]
        f1s = [r['f1_score'] for r in runs]
        aucs = [r['roc_auc'] for r in runs]
        fprs = [r['fpr'] for r in runs]
        fpr99s = [r['fpr_at_99pct_tpr'] for r in runs]
        fns = [r['malware_fn'] for r in runs]
        times = [r['training_time_s'] for r in runs]
        lats = [r['latency_ms'] for r in runs]

        summary_table.append({
            "model": model_name,
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "accuracy_std": round(float(np.std(accs)), 4),
            "precision_mean": round(float(np.mean(precs)), 4),
            "precision_std": round(float(np.std(precs)), 4),
            "recall_mean": round(float(np.mean(recs)), 4),
            "recall_std": round(float(np.std(recs)), 4),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
            "roc_auc_mean": round(float(np.mean(aucs)), 4),
            "roc_auc_std": round(float(np.std(aucs)), 4),
            "fpr_mean": round(float(np.mean(fprs)), 4),
            "fpr_std": round(float(np.std(fprs)), 4),
            "fpr_at_99pct_tpr_mean": round(float(np.mean(fpr99s)), 4),
            "fpr_at_99pct_tpr_std": round(float(np.std(fpr99s)), 4),
            "malware_fn_mean": round(float(np.mean(fns)), 1),
            "malware_fn_std": round(float(np.std(fns)), 1),
            "training_time_mean_s": round(float(np.mean(times)), 2),
            "latency_mean_ms": round(float(np.mean(lats)), 4),
            "individual_runs": runs
        })

        print(f"{model_name:<22}: Acc = {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}% | AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f} | F1 = {np.mean(f1s):.4f} | FN Misses = {np.mean(fns):.1f}")

    # Save summary metrics JSON
    metrics_path = os.path.join(metrics_dir, "baseline_correction_results.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(summary_table, f, indent=2)
    print(f"\n[OK] Saved aggregated results to: {metrics_path}")

    # Generate Comparative Bar Chart
    plt.figure(figsize=(10, 5))
    x = np.arange(len(summary_table))
    w = 0.25
    m_labels = [s['model'] for s in summary_table]
    plt.bar(x - w, [s['accuracy_mean'] for s in summary_table], width=w, yerr=[s['accuracy_std'] for s in summary_table], capsize=4, label='Accuracy', color='#2b5c8f')
    plt.bar(x, [s['f1_mean'] for s in summary_table], width=w, yerr=[s['f1_std'] for s in summary_table], capsize=4, label='F1-Score', color='#d95f02')
    plt.bar(x + w, [s['roc_auc_mean'] for s in summary_table], width=w, yerr=[s['roc_auc_std'] for s in summary_table], capsize=4, label='ROC-AUC', color='#7570b3')
    plt.xticks(x, m_labels, rotation=15, ha='right')
    plt.ylim(0.70, 1.0)
    plt.ylabel("Score")
    plt.title("Corrected Baseline Comparison: Mean ± Std across 3 Seeds (42, 123, 2026)")
    plt.legend(loc="lower right")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    bar_path = os.path.join(plots_dir, "baseline_metrics_comparison_corrected.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"[OK] Saved comparative bar chart to: {bar_path}")

    # Save experiment manifest
    manifest_path = os.path.join(configs_dir, "experiment_manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({
            "experiment": "EMBER Baseline Correction Run",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "seeds": SEEDS,
            "data_partition": "16k train / 4k val (proportional reservoir sampling)",
            "device": str(device),
            "models_evaluated": list(all_runs.keys()),
            "metrics": summary_table
        }, f, indent=2)
    print(f"[OK] Saved manifest to: {manifest_path}")


if __name__ == '__main__':
    main()

