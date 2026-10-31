import os
import sys
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

repo_root = r"c:\Users\imsto\Desktop\Final Year Paper\Old Repo Data"
cma_metrics_dir = os.path.join(repo_root, "experiments", "cross_modal_attention_2026_10_30", "metrics")
base_metrics_dir = os.path.join(repo_root, "experiments", "baseline_correction_2026_10_30", "metrics")

print("=" * 80)
print("INDEPENDENT RE-CALCULATION & AUDIT OF STORED METRICS")
print("=" * 80)

# Check all 18 runs in CMA
seeds = [42, 123, 2026]
models = [
    "CMANet_Full", "CMANet_NoByte", "CMANet_NoStruct", 
    "CMANet_NoText", "CMANet_NoCrossAttn", "MultimodalFusionNet"
]

recalc_cma = {}
diff_found = False

for seed in seeds:
    pred_file = os.path.join(cma_metrics_dir, f"predictions_seed{seed}.npz")
    assert os.path.exists(pred_file), f"Missing {pred_file}"
    preds = np.load(pred_file)
    y_true = preds["y_true"]
    
    for m in models:
        key = f"p_{m.lower()}"
        y_probs = preds[key]
        
        # Load reported JSON metrics
        json_file = os.path.join(cma_metrics_dir, f"run_{m}_seed{seed}.json")
        with open(json_file, "r") as f:
            rep = json.load(f)
            
        y_pred = (y_probs >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        acc = round(float(accuracy_score(y_true, y_pred)), 4)
        prec = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
        rec = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
        f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
        auc = round(float(roc_auc_score(y_true, y_probs)), 4)
        fpr = round(float(fp / (fp + tn)), 4)
        
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_probs)
        idx_99 = np.where(tpr_arr >= 0.99)[0]
        fpr99 = round(float(fpr_arr[idx_99[0]]) if len(idx_99) > 0 else 1.0, 4)
        
        # Compare
        discrepancies = []
        for metric, val in [("accuracy", acc), ("precision", prec), ("recall", rec), ("f1_score", f1), ("roc_auc", auc), ("fpr", fpr), ("fpr_at_99pct_tpr", fpr99), ("malware_fn", fn)]:
            rep_val = rep[metric]
            if val != rep_val:
                discrepancies.append((metric, rep_val, val))
                diff_found = True
                
        if discrepancies:
            print(f"DISCREPANCY in {m} Seed {seed}: {discrepancies}")
        else:
            pass

print(f"Metrics verification against raw prediction arrays: {'FAILED' if diff_found else 'ALL 18 RUNS EXACTLY VERIFIED (Zero Discrepancy)'}")

# Task 7: F1 vs Accuracy audit for CMANet-Full
print("\n" + "=" * 80)
print("TASK 7: F1 vs ACCURACY MATHEMATICAL DISSECTION")
print("=" * 80)
cmanet_runs = []
for seed in seeds:
    preds = np.load(os.path.join(cma_metrics_dir, f"predictions_seed{seed}.npz"))
    y_true = preds["y_true"]
    y_probs = preds["p_cmanet_full"]
    y_pred = (y_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Calculate harmonic mean of P and R
    harmonic_pr = 2 * (prec * rec) / (prec + rec)
    # Balanced class accuracy is (P_class0_recall + P_class1_recall) / 2 = (specificity + sensitivity)/2
    spec = tn / (tn + fp)
    bal_acc = (spec + rec) / 2
    
    print(f"Seed {seed}:")
    print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp} (Total={len(y_true)})")
    print(f"  Precision    : {prec:.6f} ({prec*100:.2f}%)")
    print(f"  Recall       : {rec:.6f} ({rec*100:.2f}%)")
    print(f"  Accuracy     : {acc:.6f} ({acc*100:.2f}%)")
    print(f"  F1 Score     : {f1:.6f} ({f1*100:.2f}%)")
    print(f"  Harmonic(P,R): {harmonic_pr:.6f}")
    print(f"  Delta(Acc-F1): {acc - f1:.6f} (Acc is strictly higher by {(acc-f1)*100:.2f}%)")
    cmanet_runs.append((acc, f1, prec, rec))

mean_acc = np.mean([r[0] for r in cmanet_runs])
mean_f1 = np.mean([r[1] for r in cmanet_runs])
print(f"\nMean CMANet-Full Accuracy : {mean_acc:.6f} ({mean_acc*100:.2f}%)")
print(f"Mean CMANet-Full F1-Score : {mean_f1:.6f} ({mean_f1*100:.2f}%)")
print(f"Delta (Mean Acc - Mean F1): {mean_acc - mean_f1:.6f} ({(mean_acc-mean_f1)*100:.2f}%)")

# Task 8: Check StructMLP comparison provenance
print("\n" + "=" * 80)
print("TASK 8: STRUCTMLP PREDICTION PROVENANCE AUDIT")
print("=" * 80)
for seed in seeds:
    cma_pred_file = os.path.join(cma_metrics_dir, f"predictions_seed{seed}.npz")
    base_pred_file = os.path.join(base_metrics_dir, f"predictions_seed{seed}.npz")
    print(f"Seed {seed}:")
    print(f"  Base pred file exists: {os.path.exists(base_pred_file)}")
    if os.path.exists(base_pred_file):
        cma_p = np.load(cma_pred_file)
        base_p = np.load(base_pred_file)
        print(f"  CMA y_true shape: {cma_p['y_true'].shape}, Base keys: {base_p.files}")
        if "y_true" in base_p:
            y_match = np.array_equal(cma_p["y_true"], base_p["y_true"])
            print(f"  y_true exact match: {y_match}")
        elif "y_va" in base_p:
            y_match = np.array_equal(cma_p["y_true"], base_p["y_va"])
            print(f"  y_true vs y_va exact match: {y_match}")
        
        # Check StructMLP probs
        p_struct = base_p["p_struct"]
        struct_acc = accuracy_score(cma_p["y_true"], (p_struct >= 0.5).astype(int))
        struct_auc = roc_auc_score(cma_p["y_true"], p_struct)
        print(f"  StructMLP re-computed on CMA y_true: Acc={struct_acc*100:.2f}%, AUC={struct_auc:.4f}")