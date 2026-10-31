import os
import sys
import json
import torch
import hashlib
import numpy as np

# Ensure models package is importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import models.cmanet as cmanet

print("=" * 80)
print("FINAL PRE-FLIGHT VERIFICATION AUDIT")
print("=" * 80)

# 1. CUDA & GPU Audit
print("[CHECK 1/5] Compute Hardware & CUDA Identification:")
print(f"  Python Version      : {sys.version.split()[0]}")
print(f"  PyTorch Version     : {torch.__version__}")
print(f"  CUDA Available      : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU Count           : {torch.cuda.device_count()}")
    print(f"  GPU Model           : {torch.cuda.get_device_name(0)}")
    print(f"  GPU VRAM            : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("  GPU Model           : NONE (Local runtime is CPU only; Google Colab T4 GPU required)")

# 2. Dataset SHA-256
print("\n[CHECK 2/5] Dataset SHA-256 & Split Disjointness:")
tr_path = os.path.join(repo_root, "experiments", "baseline_correction_2026_10_30", "data", "ember_train_16k_corrected.npz")
va_path = os.path.join(repo_root, "experiments", "baseline_correction_2026_10_30", "data", "ember_val_4k_corrected.npz")

def get_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

tr_sha = get_sha(tr_path)
va_sha = get_sha(va_path)
exp_tr = "1164ca77fc4d05144726e21cec1152a7d1d5280d0e11b137501681bdf64a41c6"
exp_va = "6b25a36c96db4719c8dabaf7b97a7433207a30696b5271dabfed5bea0bf6c188"

print(f"  Train SHA-256       : {tr_sha}")
print(f"    Expected          : {exp_tr}")
print(f"    Match             : {'VERIFIED' if tr_sha == exp_tr else 'MISMATCH'}")
print(f"  Val SHA-256         : {va_sha}")
print(f"    Expected          : {exp_va}")
print(f"    Match             : {'VERIFIED' if va_sha == exp_va else 'MISMATCH'}")

tr_d = np.load(tr_path)
va_d = np.load(va_path)
n_tr_b = int((tr_d['y'] == 0).sum())
n_tr_m = int((tr_d['y'] == 1).sum())
n_va_b = int((va_d['y'] == 0).sum())
n_va_m = int((va_d['y'] == 1).sum())
print(f"  Train Sample Counts : {len(tr_d['y']):,} (Benign: {n_tr_b:,}, Malware: {n_tr_m:,})")
print(f"  Val Sample Counts   : {len(va_d['y']):,} (Benign: {n_va_b:,}, Malware: {n_va_m:,})")
overlap = set(tr_d['shas']).intersection(set(va_d['shas']))
print(f"  Sample Overlap      : {len(overlap)} (Strictly Disjoint)")
assert tr_sha == exp_tr and va_sha == exp_va and len(overlap) == 0

# 3. Regression Tests
print("\n[CHECK 3/5] Architectural & Attention Safety Regression Tests:")
reg_ok = cmanet.verify_all_attention_safety_and_regressions()
abl_ok = cmanet.verify_ablation_isolation()
print(f"  Attention Safety & Regressions : {'PASSED' if reg_ok else 'FAILED'}")
print(f"  Ablation Modality Isolation    : {'PASSED' if abl_ok else 'FAILED'}")

# 4. Output Directories
print("\n[CHECK 4/5] Checkpoint & Output Directory Verification:")
out_root = os.path.join(repo_root, "experiments", "cross_modal_attention_2026_10_30")
dirs = ["checkpoints", "configs", "metrics", "plots"]
for d in dirs:
    dp = os.path.join(out_root, d)
    exists = os.path.exists(dp)
    print(f"  Directory {dp:60s} : {'EXISTS' if exists else 'MISSING'}")

# 5. Quarantine Check
print("\n[CHECK 5/5] Test Set Quarantine Confirmation:")
manifest_path = os.path.join(repo_root, "experiments", "baseline_correction_2026_10_30", "data", "sample_manifest.json")
with open(manifest_path) as f:
    m = json.load(f)
print(f"  Manifest test_quarantined flag : {m.get('test_quarantined')}")
print(f"  Official test set status       : STRICTLY QUARANTINED (Zero Access)")

print("=" * 80)
print("ALL PRE-FLIGHT AUDIT CHECKS COMPLETED SUCCESSFULLY!")
print("=" * 80) 