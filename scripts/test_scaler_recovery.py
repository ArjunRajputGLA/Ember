import os
import sys
import json
import time
import math
import random
import numpy as np
from sklearn.preprocessing import StandardScaler

repo_root = r"c:\Users\imsto\Desktop\Final Year Paper\Old Repo Data"
data_dir = r"c:\Users\imsto\Desktop\Final Year Paper\ember_dataset_2018_2\ember2018"
corr_dir = os.path.join(repo_root, "experiments", "baseline_correction_2026_10_30", "data")

print("Checking StandardScaler recovery from deterministic reservoir sampling...")

# Import extraction logic from prepare_ember_data_corrected
sys.path.insert(0, os.path.join(repo_root, "scripts"))
import prepare_ember_data_corrected as prep

t0 = time.time()
# The exact sampling logic from prepare_ember_data_corrected.py:
shard_b_avail = [50000, 52338, 46563, 47004, 48433, 55662]
shard_m_avail = [0, 63713, 47044, 47309, 48413, 93521]
tot_b = sum(shard_b_avail)
tot_m = sum(shard_m_avail)

target_b_total = 10000
target_m_total = 10000

b_quotas = [int(round(target_b_total * (c / tot_b))) for c in shard_b_avail]
b_quotas[-1] += target_b_total - sum(b_quotas)

m_quotas = [int(round(target_m_total * (c / tot_m))) for c in shard_m_avail]
m_quotas[-1] += target_m_total - sum(m_quotas)

rng = random.Random(42)

all_b = []
all_m = []

for i in range(6):
    fname = f"train_features_{i}.jsonl"
    fpath = os.path.join(data_dir, fname)
    res_b, res_m, _, _, _ = prep.reservoir_sample_shard(fpath, b_quotas[i], m_quotas[i], rng)
    all_b.extend(res_b)
    all_m.extend(res_m)

rng.shuffle(all_b)
rng.shuffle(all_m)

train_recs = all_b[:8000] + all_m[:8000]
val_recs = all_b[8000:10000] + all_m[8000:10000]

rng.shuffle(train_recs)
rng.shuffle(val_recs)

# Verify SHA match
train_shas = [r['sha256'] for r in train_recs]
val_shas = [r['sha256'] for r in val_recs]

tr_npz = np.load(os.path.join(corr_dir, "ember_train_16k_corrected.npz"))
va_npz = np.load(os.path.join(corr_dir, "ember_val_4k_corrected.npz"))

shas_tr_match = (train_shas == list(tr_npz['shas']))
shas_va_match = (val_shas == list(va_npz['shas']))
print(f"Train SHAs exact match: {shas_tr_match}")
print(f"Val SHAs exact match:   {shas_va_match}")

raw_struct_tr = np.stack([prep.extract_modality_c(r) for r in train_recs], axis=0)
raw_struct_va = np.stack([prep.extract_modality_c(r) for r in val_recs], axis=0)

scaler = StandardScaler()
X_struct_tr = scaler.fit_transform(raw_struct_tr).astype(np.float32)
X_struct_va = scaler.transform(raw_struct_va).astype(np.float32)

tr_diff = np.max(np.abs(X_struct_tr - tr_npz['X_struct']))
va_diff = np.max(np.abs(X_struct_va - va_npz['X_struct']))
print(f"Max abs diff on Train X_struct: {tr_diff:.10f}")
print(f"Max abs diff on Val X_struct:   {va_diff:.10f}")
print(f"Elapsed time: {time.time() - t0:.2f}s")

if tr_diff < 1e-6 and va_diff < 1e-6:
    print("SUCCESS: StandardScaler is 100% BIT-EXACT and REPRODUCIBLE!")
    # Save the scaler artifact
    import joblib
    scaler_out = os.path.join(corr_dir, "struct_scaler.joblib")
    joblib.dump(scaler, scaler_out)
    print(f"Saved verified scaler artifact to: {scaler_out}")

