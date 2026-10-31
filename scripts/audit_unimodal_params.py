import os
import sys
import torch
import torch.nn as nn

repo_root = r"c:\Users\imsto\Desktop\Final Year Paper\Old Repo Data"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import scripts.run_baseline_correction_experiments as base_exp

def count_params(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = sum(p.numel() for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters() if p.requires_grad)
    return total, total - emb, emb

unimodals = {
    "ByteMLP": (base_exp.ByteMLP(), (1, 512)),
    "MaskedTextCNN": (base_exp.MaskedTextCNN(), (1, 256)),
    "StructMLP": (base_exp.StructMLP(), (1, 66)),
}

for name, (m, in_shape) in unimodals.items():
    tot, non_emb, emb = count_params(m)
    print(f"{name:15s} | Total: {tot:10,d} | Non-Emb: {non_emb:10,d} | Emb: {emb:10,d}")

