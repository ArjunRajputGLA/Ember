import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

repo = r"c:\Users\imsto\Desktop\Final Year Paper\Old Repo Data"
pred_path = os.path.join(repo, "experiments", "final_test_evaluation_2026_10_31", "prediction_outputs", "test_predictions_all_models.npz")
data = np.load(pred_path)
y_true = data["y_true"]

print("=" * 80)
print("TASK 4 & 5: ENSEMBLE CALCULATION & METRIC IMPLEMENTATION VERIFICATION")
print("=" * 80)

print(f"Total test samples: {len(y_true)}")
print(f"Label distribution: Benign={np.sum(y_true == 0)}, Malware={np.sum(y_true == 1)}")

# 1. Ensemble Calculation Verification
p42 = data["p_multimodalfusionnet_seed42"]
p123 = data["p_multimodalfusionnet_seed123"]
p2026 = data["p_multimodalfusionnet_seed2026"]
p_ens = data["p_multimodalfusionnet_ensemble"]

expected_ens = (p42 + p123 + p2026) / 3.0
diff = np.max(np.abs(p_ens - expected_ens))
print(f"\n1. Ensemble Verification:")
print(f"   Max abs diff between saved ensemble and (p42+p123+p2026)/3: {diff:.10e}")
assert diff < 1e-7, "Ensemble does not match arithmetic mean!"
print("   [PASS] Ensemble is verified to be exact arithmetic mean soft-voting.")

# 2. FPR@99%TPR formula check
def compute_fpr_at_99_tpr(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.where(tpr >= 0.99)[0]
    if len(idx) == 0:
        return 1.0
    return fpr[idx[0]]

# 3. Comprehensive verification against seed_wise_metrics.csv
df_seed = pd.read_csv(os.path.join(repo, "experiments", "final_test_evaluation_2026_10_31", "seed_wise_metrics.csv"))
print(f"\n2. Verification of all models against seed_wise_metrics.csv ({len(df_seed)} rows):")

all_matched = True
for idx, row in df_seed.iterrows():
    m_name = row["model_id"]
    if m_name == "MultimodalFusionNet_Ensemble":
        key = "p_multimodalfusionnet_ensemble"
    else:
        key = f"p_{m_name.lower()}"
    
    if key not in data:
        print(f"   [MISSING KEY] {key}")
        all_matched = False
        continue
    
    prob = data[key]
    pred = (prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred)
    rec = recall_score(y_true, pred)
    f1 = f1_score(y_true, pred)
    auc = roc_auc_score(y_true, prob)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    fpr = fp / (fp + tn)
    fpr99 = compute_fpr_at_99_tpr(y_true, prob)
    
    # Check diffs
    diff_acc = abs(acc - row["accuracy"])
    diff_prec = abs(prec - row["precision"])
    diff_rec = abs(rec - row["recall"])
    diff_f1 = abs(f1 - row["f1_score"])
    diff_auc = abs(auc - row["roc_auc"])
    diff_fpr = abs(fpr - row["fpr"])
    diff_fpr99 = abs(fpr99 - row["fpr_at_99pct_tpr"])
    diff_fn = abs(fn - row["malware_fn"])
    
    max_d = max(diff_acc, diff_prec, diff_rec, diff_f1, diff_auc, diff_fpr, diff_fpr99, diff_fn)
    if max_d > 1e-4:
        print(f"   [MISMATCH] {m_name}: max diff {max_d}")
        all_matched = False
    else:
        print(f"   [OK] {m_name:30s} | Acc={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f} | FPR@99%TPR={fpr99:.4f} | FN={fn:5d}")

if all_matched:
    print("\n   [ALL VERIFIED] Every single metric in seed_wise_metrics.csv matches independent calculation!")

# 4. Verification of final_test_metrics.csv (aggregates)
df_agg = pd.read_csv(os.path.join(repo, "experiments", "final_test_evaluation_2026_10_31", "final_test_metrics.csv"))
print(f"\n3. Aggregate Metrics Verification ({len(df_agg)} models):")
print(df_agg.to_string())
