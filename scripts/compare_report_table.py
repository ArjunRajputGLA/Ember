import os
import json

repo = r"c:\Users\imsto\Desktop\Final Year Paper\Old Repo Data"
metrics_dir = os.path.join(repo, "experiments", "cross_modal_attention_2026_10_30", "metrics")
json_path = os.path.join(metrics_dir, "cross_modal_attention_results.json")

with open(json_path) as f:
    d = json.load(f)

print(f"{'Model':20s} | {'Seed':5s} | {'Acc (%)':8s} | {'Prec (%)':8s} | {'Rec (%)':8s} | {'F1 (%)':8s} | {'FN':5s}")
print("-" * 75)
for m, mdata in d['models'].items():
    for run in mdata['seed_runs']:
        s = run['seed']
        acc = run['accuracy'] * 100
        prec = run['precision'] * 100
        rec = run['recall'] * 100
        f1 = run['f1_score'] * 100
        fn = run['malware_fn']
        print(f"{m:20s} | {s:5d} | {acc:8.2f} | {prec:8.2f} | {rec:8.2f} | {f1:8.2f} | {fn:5d}")

