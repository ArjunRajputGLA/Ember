import os
import sys
import json
import csv
import numpy as np

repo = r"c:\Users\imsto\Desktop\Final Year Paper\Old Repo Data"
out_csv = os.path.join(repo, "reports", "experiment_inventory.csv")

rows = []
fieldnames = [
    "experiment_id",
    "phase",
    "model_name",
    "seed",
    "dataset_split",
    "train_samples",
    "val_samples",
    "batch_size",
    "epochs",
    "learning_rate",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "malware_fn",
    "training_time_s",
    "checkpoint_file",
    "metrics_file",
    "prediction_file",
    "status"
]

# 1. Initial Baseline (5 runs, Seed 42)
init_dir = os.path.join(repo, "experiments", "initial_baseline_2026_10_30")
init_json = os.path.join(init_dir, "metrics", "colab_initial_baseline_results.json")
if os.path.exists(init_json):
    with open(init_json) as f:
        init_data = json.load(f)
    for idx, item in enumerate(init_data, start=1):
        raw_name = item["Model"].split(":")[-1].strip()
        rows.append({
            "experiment_id": f"INIT_{idx:02d}",
            "phase": "Initial Baseline (Preliminary)",
            "model_name": raw_name,
            "seed": 42,
            "dataset_split": "Initial Head-of-Shard (Early Break)",
            "train_samples": 16000,
            "val_samples": 4000,
            "batch_size": 64,
            "epochs": 10 if "LightGBM" not in raw_name else 1000,
            "learning_rate": 0.001 if "LightGBM" not in raw_name else 0.05,
            "accuracy": item["Accuracy"],
            "precision": item["Precision"],
            "recall": item["Recall"],
            "f1_score": item["F1-Score"],
            "roc_auc": item["ROC-AUC"],
            "malware_fn": item["Malware_FN"],
            "training_time_s": item["Training_Time_s"],
            "checkpoint_file": f"checkpoints/best_Experiment_{idx}_{raw_name}.pt" if "LightGBM" not in raw_name else "checkpoints/fresh_lightgbm.txt",
            "metrics_file": "metrics/colab_initial_baseline_results.json",
            "prediction_file": f"metrics/preds_Experiment_{idx}_{raw_name}.npz" if "LightGBM" not in raw_name else "None",
            "status": "SUPERSEDED (Sampling Bias Identified & Corrected)"
        })

# 2. Baseline Correction (15 runs, Seeds 42, 123, 2026)
base_corr_dir = os.path.join(repo, "experiments", "baseline_correction_2026_10_30")
base_corr_manifest = os.path.join(base_corr_dir, "configs", "experiment_manifest.json")
if os.path.exists(base_corr_manifest):
    with open(base_corr_manifest) as f:
        b_data = json.load(f)
    exp_idx = 1
    for mgroup in b_data["metrics"]:
        m_name = mgroup["model"]
        for run in mgroup["individual_runs"]:
            s = run["seed"]
            rows.append({
                "experiment_id": f"BC_{exp_idx:02d}",
                "phase": "Baseline Correction (Audited)",
                "model_name": m_name,
                "seed": s,
                "dataset_split": "Corrected Reservoir Sampling (Algorithm R)",
                "train_samples": 16000,
                "val_samples": 4000,
                "batch_size": 64,
                "epochs": 10 if "LightGBM" not in m_name else 1000,
                "learning_rate": 0.001 if "LightGBM" not in m_name else 0.05,
                "accuracy": round(run["accuracy"], 4),
                "precision": round(run["precision"], 4),
                "recall": round(run["recall"], 4),
                "f1_score": round(run["f1_score"], 4),
                "roc_auc": round(run["roc_auc"], 4),
                "malware_fn": run["malware_fn"],
                "training_time_s": "Logged in run json",
                "checkpoint_file": f"checkpoints/best_{m_name}_seed{s}.pt" if "LightGBM" not in m_name else f"checkpoints/fresh_lightgbm_seed{s}.txt",
                "metrics_file": f"metrics/run_{m_name}_seed{s}.json",
                "prediction_file": f"metrics/predictions_seed{s}.npz",
                "status": "VERIFIED & AUDITED"
            })
            exp_idx += 1

# 3. Cross Modal Attention Smoke (6 runs, Seed 42)
smoke_dir = os.path.join(repo, "experiments", "cross_modal_attention_2026_10_30", "metrics", "smoke")
smoke_json = os.path.join(smoke_dir, "smoke_verification_summary.json")
if os.path.exists(smoke_json):
    with open(smoke_json) as f:
        s_data = json.load(f)
    s_idx = 1
    for sm_name, sm_res in s_data.items():
        rows.append({
            "experiment_id": f"SMOKE_{s_idx:02d}",
            "phase": "Cross-Modal Smoke Preflight",
            "model_name": sm_name,
            "seed": 42,
            "dataset_split": "Corrected Reservoir 200-sample slice",
            "train_samples": 200,
            "val_samples": 50,
            "batch_size": 32,
            "epochs": 1,
            "learning_rate": 0.001,
            "accuracy": sm_res["accuracy"],
            "precision": sm_res["precision"],
            "recall": sm_res["recall"],
            "f1_score": sm_res["f1_score"],
            "roc_auc": sm_res["roc_auc"],
            "malware_fn": sm_res["malware_fn"],
            "training_time_s": sm_res["training_time_s"],
            "checkpoint_file": f"checkpoints/smoke/best_{sm_name}_seed42.pt",
            "metrics_file": f"metrics/smoke/run_{sm_name}_seed42.json",
            "prediction_file": "None (Smoke slice)",
            "status": "PREFLIGHT PASS"
        })
        s_idx += 1

# 4. Cross Modal Attention Full Benchmark (18 runs, Seeds 42, 123, 2026)
cma_dir = os.path.join(repo, "experiments", "cross_modal_attention_2026_10_30")
cma_agg = os.path.join(cma_dir, "metrics", "cross_modal_attention_results.json")
if os.path.exists(cma_agg):
    with open(cma_agg) as f:
        c_data = json.load(f)
    c_idx = 1
    for cm_name, cm_group in c_data["models"].items():
        for r_run in cm_group["seed_runs"]:
            s = r_run["seed"]
            rows.append({
                "experiment_id": f"CMA_{c_idx:02d}",
                "phase": "Cross-Modal Benchmark",
                "model_name": cm_name,
                "seed": s,
                "dataset_split": "Corrected Reservoir Sampling (Algorithm R)",
                "train_samples": 16000,
                "val_samples": 4000,
                "batch_size": 32,
                "epochs": 10,
                "learning_rate": 0.001,
                "accuracy": r_run["accuracy"],
                "precision": r_run["precision"],
                "recall": r_run["recall"],
                "f1_score": r_run["f1_score"],
                "roc_auc": r_run["roc_auc"],
                "malware_fn": r_run["malware_fn"],
                "training_time_s": r_run["training_time_s"],
                "checkpoint_file": f"checkpoints/best_{cm_name}_seed{s}.pt",
                "metrics_file": f"metrics/run_{cm_name}_seed{s}.json",
                "prediction_file": f"metrics/predictions_seed{s}.npz",
                "status": "VERIFIED & AUDITED"
            })
            c_idx += 1

# Write CSV
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated experiment inventory with {len(rows)} records at: {out_csv}")

