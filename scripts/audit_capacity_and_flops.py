import os
import sys
import torch
import torch.nn as nn
import numpy as np

repo_root = r"c:\Users\imsto\Desktop\Final Year Paper\Old Repo Data"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import models.cmanet as cmanet
import scripts.run_baseline_correction_experiments as base_exp

print("=" * 80)
print("INDEPENDENT RE-CALCULATION OF PARAMETERS, FLOPs, AND MACs")
print("=" * 80)

def count_params(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = sum(p.numel() for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters() if p.requires_grad)
    return total, total - emb, emb

# Test all models
models_to_test = {
    "CMANet_Full": cmanet.CrossModalAttentionNet,
    "CMANet_NoByte": cmanet.CMANet_NoByte,
    "CMANet_NoStruct": cmanet.CMANet_NoStruct,
    "CMANet_NoText": cmanet.CMANet_NoText,
    "CMANet_NoCrossAttn": cmanet.CMANet_NoCrossAttn,
    "MultimodalFusionNet (cmanet.py)": cmanet.MultimodalFusionNet,
    "MultimodalFusionNet (base_exp)": base_exp.MultimodalFusionNet,
    "ByteMLP": base_exp.ByteMLP,
    "MaskedTextCNN": base_exp.MaskedTextCNN,
    "StructMLP": base_exp.StructMLP,
}

print(f"{'Model Name':35s} | {'Total Params':12s} | {'Non-Emb Params':14s} | {'Emb Params':10s} | {'Calculated MACs':15s} | {'Calculated MFLOPs':17s}")
print("-" * 115)

for name, cls in models_to_test.items():
    m = cls()
    tot, non_emb, emb = count_params(m)
    
    # Calculate MACs / FLOPs
    if hasattr(cmanet, "compute_model_capacity"):
        cap = cmanet.compute_model_capacity(m, name)
        macs = cap.get("macs", 0)
        mflops = cap.get("mflops", 0.0)
    else:
        macs = 0
        mflops = 0.0
    print(f"{name:35s} | {tot:12,d} | {non_emb:14,d} | {emb:10,d} | {macs:15,d} | {mflops:17.2f}")
