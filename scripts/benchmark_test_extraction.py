import os
import sys
import json
import time
import numpy as np
import joblib

repo_root = r"c:\Users\imsto\Desktop\Final Year Paper\Old Repo Data"
test_path = r"c:\Users\imsto\Desktop\Final Year Paper\ember_dataset_2018_2\ember2018\test_features.jsonl"
data_dir = os.path.join(repo_root, "experiments", "baseline_correction_2026_10_30", "data")

sys.path.insert(0, os.path.join(repo_root, "scripts"))
import prepare_ember_data_corrected as prep

with open(os.path.join(data_dir, "vocab.json")) as f:
    vocab = json.load(f)

scaler = joblib.load(os.path.join(data_dir, "struct_scaler.joblib"))

extractor = prep.ef.PEFeatureExtractor(feature_version=2, print_feature_warning=False)

def encode_tokens(toks, max_len=256):
    ids = [vocab.get(t, 1) for t in toks[:max_len]]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids

print("Benchmarking feature extraction on 1,000 test lines...")
t0 = time.time()
n_bench = 1000

xb_arr = np.zeros((n_bench, 512), dtype=np.float32)
xt_arr = np.zeros((n_bench, 256), dtype=np.int64)
xs_raw = np.zeros((n_bench, 66), dtype=np.float32)
xe_arr = np.zeros((n_bench, 2381), dtype=np.float32)
y_arr = np.zeros(n_bench, dtype=np.int64)

with open(test_path, "r", encoding="utf-8") as f:
    for i in range(n_bench):
        line = f.readline()
        rec = json.loads(line)
        xb_arr[i] = prep.extract_modality_a(rec)
        xt_arr[i] = encode_tokens(prep.extract_modality_b_tokens(rec))
        xs_raw[i] = prep.extract_modality_c(rec)
        xe_arr[i] = extractor.process_raw_features(rec)
        y_arr[i] = rec.get("label", -1)

xs_scaled = scaler.transform(xs_raw).astype(np.float32)
elapsed = time.time() - t0
print(f"Extracted {n_bench} records in {elapsed:.2f}s ({elapsed/n_bench*1000:.2f} ms/record)")
print(f"Projected time for 200,000 records: {elapsed * 200 / 60:.1f} minutes")
print(f"Label distribution in sample: {np.bincount(y_arr)}")

