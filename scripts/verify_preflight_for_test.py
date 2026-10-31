import os
import sys
import json
import hashlib
import torch
import lightgbm as lgb
import numpy as np
import joblib

repo_root = r"c:\Users\imsto\Desktop\Final Year Paper\Old Repo Data"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import models.cmanet as cmanet
import scripts.run_baseline_correction_experiments as base_exp

print("=" * 80)
print("PRE-EXECUTION AUDIT & SMOKE VERIFICATION FOR FINAL TEST EVALUATION")
print("=" * 80)

data_dir = os.path.join(repo_root, "experiments", "baseline_correction_2026_10_30", "data")
cma_ckpt_dir = os.path.join(repo_root, "experiments", "cross_modal_attention_2026_10_30", "checkpoints")
base_ckpt_dir = os.path.join(repo_root, "experiments", "baseline_correction_2026_10_30", "checkpoints")

def file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

# 1. Artifacts existence
vocab_path = os.path.join(data_dir, "vocab.json")
scaler_path = os.path.join(data_dir, "struct_scaler.joblib")
assert os.path.exists(vocab_path), f"Missing vocab: {vocab_path}"
assert os.path.exists(scaler_path), f"Missing scaler: {scaler_path}"

with open(vocab_path) as f:
    vocab = json.load(f)
scaler = joblib.load(scaler_path)
print(f"[OK] Vocabulary loaded: {len(vocab):,} tokens (pad={vocab.get('<PAD>')}, unk={vocab.get('<UNK>')})")
print(f"[OK] StandardScaler loaded: mean shape {scaler.mean_.shape}, scale shape {scaler.scale_.shape}")

# 2. Checkpoints manifest
checkpoints = {
    # Primary model (from cross_modal_attention_2026_10_30)
    "MultimodalFusionNet_seed42": (cmanet.MultimodalFusionNet, os.path.join(cma_ckpt_dir, "best_MultimodalFusionNet_seed42.pt")),
    "MultimodalFusionNet_seed123": (cmanet.MultimodalFusionNet, os.path.join(cma_ckpt_dir, "best_MultimodalFusionNet_seed123.pt")),
    "MultimodalFusionNet_seed2026": (cmanet.MultimodalFusionNet, os.path.join(cma_ckpt_dir, "best_MultimodalFusionNet_seed2026.pt")),
    
    # Baselines (from baseline_correction_2026_10_30)
    "ByteMLP_seed42": (base_exp.ByteMLP, os.path.join(base_ckpt_dir, "best_ByteMLP_seed42.pt")),
    "ByteMLP_seed123": (base_exp.ByteMLP, os.path.join(base_ckpt_dir, "best_ByteMLP_seed123.pt")),
    "ByteMLP_seed2026": (base_exp.ByteMLP, os.path.join(base_ckpt_dir, "best_ByteMLP_seed2026.pt")),
    
    "MaskedTextCNN_seed42": (base_exp.MaskedTextCNN, os.path.join(base_ckpt_dir, "best_MaskedTextCNN_seed42.pt")),
    "MaskedTextCNN_seed123": (base_exp.MaskedTextCNN, os.path.join(base_ckpt_dir, "best_MaskedTextCNN_seed123.pt")),
    "MaskedTextCNN_seed2026": (base_exp.MaskedTextCNN, os.path.join(base_ckpt_dir, "best_MaskedTextCNN_seed2026.pt")),
    
    "StructMLP_seed42": (base_exp.StructMLP, os.path.join(base_ckpt_dir, "best_StructMLP_seed42.pt")),
    "StructMLP_seed123": (base_exp.StructMLP, os.path.join(base_ckpt_dir, "best_StructMLP_seed123.pt")),
    "StructMLP_seed2026": (base_exp.StructMLP, os.path.join(base_ckpt_dir, "best_StructMLP_seed2026.pt")),
    
    # Negative Reference (from cross_modal_attention_2026_10_30)
    "CMANet_Full_seed42": (cmanet.CrossModalAttentionNet, os.path.join(cma_ckpt_dir, "best_CMANet_Full_seed42.pt")),
    "CMANet_Full_seed123": (cmanet.CrossModalAttentionNet, os.path.join(cma_ckpt_dir, "best_CMANet_Full_seed123.pt")),
    "CMANet_Full_seed2026": (cmanet.CrossModalAttentionNet, os.path.join(cma_ckpt_dir, "best_CMANet_Full_seed2026.pt")),
    
    # Fresh LightGBM (from baseline_correction_2026_10_30)
    "Fresh_LightGBM_seed42": ("lightgbm", os.path.join(base_ckpt_dir, "fresh_lightgbm_seed42.txt")),
    "Fresh_LightGBM_seed123": ("lightgbm", os.path.join(base_ckpt_dir, "fresh_lightgbm_seed123.txt")),
    "Fresh_LightGBM_seed2026": ("lightgbm", os.path.join(base_ckpt_dir, "fresh_lightgbm_seed2026.txt")),
}

print(f"\nVerifying {len(checkpoints)} required model checkpoints:")
ckpt_hashes = {}
for name, (cls, path) in checkpoints.items():
    assert os.path.exists(path), f"MISSING CHECKPOINT: {path}"
    sha = file_sha256(path)
    sz = os.path.getsize(path)
    ckpt_hashes[name] = {"path": path, "sha256": sha, "size_bytes": sz}
    print(f"  [FOUND] {name:28s} | Size: {sz:>8,d} bytes | SHA: {sha[:16]}...")

# 3. Validation Slice Smoke Test
print("\nRunning smoke test inference on 5 validation samples...")
val_npz_path = os.path.join(data_dir, "ember_val_4k_corrected.npz")
val_data = np.load(val_npz_path)

xb = torch.tensor(val_data["X_byte"][:5], dtype=torch.float32)
xt = torch.tensor(val_data["X_text"][:5], dtype=torch.long)
xs = torch.tensor(val_data["X_struct"][:5], dtype=torch.float32)
xe = val_data["X_ember2381"][:5]

for name, (cls, path) in checkpoints.items():
    if cls == "lightgbm":
        bst = lgb.Booster(model_file=path)
        p = bst.predict(xe)
        assert len(p) == 5
        assert np.all((p >= 0.0) & (p <= 1.0))
        print(f"  [SMOKE OK] {name:28s} -> Probs: {np.round(p, 4)}")
    else:
        m = cls()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        with torch.no_grad():
            if "ByteMLP" in name:
                logits = m(xb)
            elif "MaskedTextCNN" in name:
                logits = m(xt)
            elif "StructMLP" in name:
                logits = m(xs)
            else:
                logits = m(xb, xt, xs)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            assert len(probs) == 5
            assert np.all((probs >= 0.0) & (probs <= 1.0))
            assert np.all(np.isfinite(probs))
            print(f"  [SMOKE OK] {name:28s} -> Probs: {np.round(probs, 4)}")

print("\n" + "=" * 80)
print("ALL PRE-FLIGHT VERIFICATIONS PASSED: 18 CHECKPOINTS + 2 ARTIFACTS VERIFIED")
print("=" * 80)

