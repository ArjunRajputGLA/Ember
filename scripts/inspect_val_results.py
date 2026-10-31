import os
import json

repo = r"c:\Users\imsto\Desktop\Final Year Paper\Old Repo Data"
p1 = os.path.join(repo, "experiments", "baseline_correction_2026_10_30", "metrics", "baseline_correction_results.json")
with open(p1) as f:
    d1 = json.load(f)

print("--- BASELINE CORRECTION RESULTS (VAL) ---")
if isinstance(d1, list):
    for item in d1:
        print(item)
elif isinstance(d1, dict):
    for m, vals in d1.items():
        print(m, vals)

p2 = os.path.join(repo, "experiments", "cross_modal_attention_2026_10_30", "metrics", "cross_modal_attention_results.json")
with open(p2) as f:
    d2 = json.load(f)

print("\n--- CROSS MODAL ATTENTION RESULTS (VAL) ---")
if "models" in d2:
    for m, vals in d2["models"].items():
        if isinstance(vals, dict) and "aggregate" in vals:
            agg = vals["aggregate"]
            print(f"{m:25s}: Acc={agg.get('accuracy_mean',0)*100:.2f} +/- {agg.get('accuracy_std',0)*100:.2f}%, F1={agg.get('f1_mean',0)*100:.2f}%, AUC={agg.get('roc_auc_mean',0):.4f}")
