import os
import hashlib
import numpy as np

repo = r"c:\Users\imsto\Desktop\Final Year Paper\Old Repo Data"
ckpt_dir = os.path.join(repo, "experiments", "baseline_correction_2026_10_30", "checkpoints")
pred_path = os.path.join(repo, "experiments", "final_test_evaluation_2026_10_31", "prediction_outputs", "test_predictions_all_models.npz")

print("=" * 80)
print("TASK 3: LIGHTGBM IDENTITY & DETERMINISM INVESTIGATION")
print("=" * 80)

# 1. Prediction arrays check
preds = np.load(pred_path)
p42 = preds["p_fresh_lightgbm_seed42"]
p123 = preds["p_fresh_lightgbm_seed123"]
p2026 = preds["p_fresh_lightgbm_seed2026"]

exact_42_123 = np.array_equal(p42, p123)
exact_42_2026 = np.array_equal(p42, p2026)
max_diff_123 = np.max(np.abs(p42 - p123))
max_diff_2026 = np.max(np.abs(p42 - p2026))

print(f"Prediction arrays identical (42 vs 123) : {exact_42_123} (Max abs diff: {max_diff_123})")
print(f"Prediction arrays identical (42 vs 2026): {exact_42_2026} (Max abs diff: {max_diff_2026})")

# 2. Checkpoint files check
ckpts = {
    "seed42": os.path.join(ckpt_dir, "fresh_lightgbm_seed42.txt"),
    "seed123": os.path.join(ckpt_dir, "fresh_lightgbm_seed123.txt"),
    "seed2026": os.path.join(ckpt_dir, "fresh_lightgbm_seed2026.txt"),
}

def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

shas = {}
sizes = {}
for s, p in ckpts.items():
    shas[s] = file_sha(p)
    sizes[s] = os.path.getsize(p)
    print(f"Checkpoint {s:8s} | Path: {p} | Size: {sizes[s]:,d} bytes | SHA-256: {shas[s]}")

# 3. Model file content diff
with open(ckpts["seed42"], "r") as f:
    lines42 = f.readlines()
with open(ckpts["seed123"], "r") as f:
    lines123 = f.readlines()
with open(ckpts["seed2026"], "r") as f:
    lines2026 = f.readlines()

print(f"\nLine count: 42={len(lines42)}, 123={len(lines123)}, 2026={len(lines2026)}")
diff_lines_123 = [i for i, (l1, l2) in enumerate(zip(lines42, lines123)) if l1 != l2]
diff_lines_2026 = [i for i, (l1, l2) in enumerate(zip(lines42, lines2026)) if l1 != l2]
print(f"Differing lines between 42 and 123 : {len(diff_lines_123)}")
if diff_lines_123:
    for idx in diff_lines_123[:5]:
        print(f"  Line {idx}: 42='{lines42[idx].strip()}' vs 123='{lines123[idx].strip()}'")

print(f"Differing lines between 42 and 2026: {len(diff_lines_2026)}")
if diff_lines_2026:
    for idx in diff_lines_2026[:5]:
        print(f"  Line {idx}: 42='{lines42[idx].strip()}' vs 2026='{lines2026[idx].strip()}'")

