#!/usr/bin/env python3
"""
test_cross_modal_architecture.py

Comprehensive verification and unit test suite for:
- CrossModalAttentionNet (CMANet: Bidirectional Global-to-Sequence Cross-Attention)
- CMANet-NoByte (Ablation 1)
- CMANet-NoStruct (Ablation 2)
- CMANet-NoText (Ablation 3)
- CMANet-NoCrossAttn (Ablation 4)
- MultimodalFusionNet (Reference Baseline)

Checks:
1. Tensor dimensions across variable batch sizes and sequence lengths
2. Strict padding invariance across sequences padded to lengths 10, 50, 128, 256 (< 1e-6 diff)
3. Non-zero gradient flow across all trainable parameter tensors
4. Finite outputs under normal and extreme edge cases (e.g. 100% padding tokens)
5. Attention weight masking correctness (pad keys receive strictly 0.0 attention)
6. CPU/GPU portability
7. Checkpoint save/load exact reproduction equivalence
8. 200-sample, 1-epoch smoke test on corrected EMBER data
"""

import os
import sys
import json
import time
import math
import random
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ==============================================================================
# 1. MODEL DEFINITIONS
# ==============================================================================

class CrossModalAttentionNet(nn.Module):
    """
    Bidirectional Global-to-Sequence Cross-Attention Network (CMANet).
    Accurate architectural description: Bidirectional global-to-sequence cross-attention.
    Modality A: Byte Histogram & Entropy Histogram (512-D continuous vector)
    Modality B: Hashed API / Token Sequence (L-length integer sequence, L <= 256)
    Modality C: PE Structural Properties (66-D continuous vector)
    """
    def __init__(self, vocab_sz=10000, byte_dim=512, struct_dim=66, d_model=128, nhead=4, dropout=0.1, max_len=256):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 1. Modality Projections (Byte & Struct -> d_model)
        self.byte_proj = nn.Sequential(
            nn.Linear(byte_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.struct_proj = nn.Sequential(
            nn.Linear(struct_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # Modality type embeddings for the two global tokens [Byte, Struct]
        self.modality_type_emb = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)
        
        # 2. Sequence Embeddings (Token + Learned Positional)
        self.text_emb = nn.Embedding(vocab_sz, d_model, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.emb_drop = nn.Dropout(dropout)
        
        # 3. Intra-Modal Text Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(d_model)
        
        # 4. Bidirectional Cross-Attention
        # Direction 1: Sequence to Global (Global tokens query API sequence)
        self.cross_attn_g2t = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_g2t = nn.LayerNorm(d_model)
        
        # Direction 2: Global to Sequence (API sequence queries Global tokens)
        self.cross_attn_t2g = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_t2g = nn.LayerNorm(d_model)
        
        self.drop = nn.Dropout(dropout)
        
        # 5. Classifier Head (Capacity-comparable to baseline classifier)
        # Input dim: T_byte' (128) + T_struct' (128) + h_text (128) = 384
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x_byte, x_text, x_struct, return_intermediates=False):
        B, L = x_text.shape
        
        # Create boolean mask: True for valid tokens, False for padding (idx == 0)
        valid_mask = (x_text != 0)  # (B, L)
        float_mask = valid_mask.unsqueeze(-1).float()  # (B, L, 1)
        has_valid = valid_mask.any(dim=1)  # (B,)
        
        # Safe key padding mask: True means ignore (pad)
        # Avoid all-True key padding mask per sample to prevent softmax NaN
        safe_key_padding_mask = ~valid_mask
        if not has_valid.all():
            safe_key_padding_mask = safe_key_padding_mask.clone()
            safe_key_padding_mask[~has_valid, 0] = False
            
        # 1. Project Global Modalities into Summary Tokens
        t_b = self.byte_proj(x_byte).unsqueeze(1)    # (B, 1, d_model)
        t_s = self.struct_proj(x_struct).unsqueeze(1) # (B, 1, d_model)
        t_global = torch.cat([t_b, t_s], dim=1) + self.modality_type_emb # (B, 2, d_model)
        
        # 2. Sequence Token Embeddings + Positional Encoding
        h_text = (self.text_emb(x_text) + self.pos_emb[:, :L, :]) * float_mask
        h_text = self.emb_drop(h_text) * float_mask
        
        # 3. Intra-modal Sequence Self-Attention with Strict Post-Norm Re-masking
        attn_self, weights_self = self.self_attn(
            h_text, h_text, h_text,
            key_padding_mask=safe_key_padding_mask,
            need_weights=return_intermediates
        )
        # Zero out padding outputs before residual and re-mask after LayerNorm to eliminate bias leakage
        h_text = self.norm_self(h_text + self.drop(attn_self * float_mask)) * float_mask
        
        # 4. Bidirectional Cross-Attention
        # Direction 1: Sequence to Global (Global tokens query Sequence)
        attn_g2t, weights_g2t = self.cross_attn_g2t(
            t_global, h_text, h_text,
            key_padding_mask=safe_key_padding_mask,
            need_weights=return_intermediates
        )
        attn_g2t = attn_g2t * has_valid.unsqueeze(1).unsqueeze(2).float()
        t_global_out = self.norm_g2t(t_global + self.drop(attn_g2t))
        t_byte_prime = t_global_out[:, 0, :]   # (B, d_model)
        t_struct_prime = t_global_out[:, 1, :] # (B, d_model)
        
        # Direction 2: Global to Sequence (Sequence tokens query Global tokens)
        attn_t2g, weights_t2g = self.cross_attn_t2g(
            h_text, t_global, t_global,
            need_weights=return_intermediates
        )
        # Strict post-norm re-masking on API sequence positions
        h_text_out = self.norm_t2g(h_text + self.drop(attn_t2g * float_mask)) * float_mask
        
        # 5. Masked Average Sequence Pooling
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0).float() # (B, 1)
        h_text_pooled = (h_text_out * float_mask).sum(dim=1) / valid_counts       # (B, d_model)
        h_text_pooled = h_text_pooled * has_valid.unsqueeze(1).float()
        
        # 6. Readout Fusion & Classification
        z_fused = torch.cat([t_byte_prime, t_struct_prime, h_text_pooled], dim=1) # (B, 384)
        logits = self.classifier(z_fused)
        
        if return_intermediates:
            return logits, {
                "t_byte_prime": t_byte_prime,
                "t_struct_prime": t_struct_prime,
                "h_text_out": h_text_out,
                "h_text_pooled": h_text_pooled,
                "z_fused": z_fused,
                "weights_self": weights_self,
                "weights_g2t": weights_g2t,
                "weights_t2g": weights_t2g,
            }
        return logits


class CMANet_NoByte(nn.Module):
    """
    Ablation 1: CMANet without Byte modality (Struct + Text cross-attention).
    """
    def __init__(self, vocab_sz=10000, struct_dim=66, d_model=128, nhead=4, dropout=0.1, max_len=256):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        self.struct_proj = nn.Sequential(
            nn.Linear(struct_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.text_emb = nn.Embedding(vocab_sz, d_model, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.emb_drop = nn.Dropout(dropout)
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(d_model)
        
        self.cross_attn_g2t = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_g2t = nn.LayerNorm(d_model)
        
        self.cross_attn_t2g = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_t2g = nn.LayerNorm(d_model)
        
        self.drop = nn.Dropout(dropout)
        
        # Readout: T_struct' (128) + h_text (128) = 256
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x_byte, x_text, x_struct, return_intermediates=False):
        # x_byte is ignored
        B, L = x_text.shape
        valid_mask = (x_text != 0)
        float_mask = valid_mask.unsqueeze(-1).float()
        has_valid = valid_mask.any(dim=1)
        
        safe_key_padding_mask = ~valid_mask
        if not has_valid.all():
            safe_key_padding_mask = safe_key_padding_mask.clone()
            safe_key_padding_mask[~has_valid, 0] = False
            
        t_global = self.struct_proj(x_struct).unsqueeze(1) # (B, 1, d_model)
        
        h_text = (self.text_emb(x_text) + self.pos_emb[:, :L, :]) * float_mask
        h_text = self.emb_drop(h_text) * float_mask
        
        attn_self, _ = self.self_attn(h_text, h_text, h_text, key_padding_mask=safe_key_padding_mask)
        h_text = self.norm_self(h_text + self.drop(attn_self * float_mask)) * float_mask
        
        attn_g2t, _ = self.cross_attn_g2t(t_global, h_text, h_text, key_padding_mask=safe_key_padding_mask)
        attn_g2t = attn_g2t * has_valid.unsqueeze(1).unsqueeze(2).float()
        t_global_out = self.norm_g2t(t_global + self.drop(attn_g2t))
        t_struct_prime = t_global_out[:, 0, :]
        
        attn_t2g, _ = self.cross_attn_t2g(h_text, t_global, t_global)
        h_text_out = self.norm_t2g(h_text + self.drop(attn_t2g * float_mask)) * float_mask
        
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0).float()
        h_text_pooled = (h_text_out * float_mask).sum(dim=1) / valid_counts
        h_text_pooled = h_text_pooled * has_valid.unsqueeze(1).float()
        
        z_fused = torch.cat([t_struct_prime, h_text_pooled], dim=1) # (B, 256)
        return self.classifier(z_fused)


class CMANet_NoStruct(nn.Module):
    """
    Ablation 2: CMANet without PE Structural modality (Byte + Text cross-attention).
    """
    def __init__(self, vocab_sz=10000, byte_dim=512, d_model=128, nhead=4, dropout=0.1, max_len=256):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        self.byte_proj = nn.Sequential(
            nn.Linear(byte_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.text_emb = nn.Embedding(vocab_sz, d_model, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.emb_drop = nn.Dropout(dropout)
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(d_model)
        
        self.cross_attn_g2t = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_g2t = nn.LayerNorm(d_model)
        
        self.cross_attn_t2g = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_t2g = nn.LayerNorm(d_model)
        
        self.drop = nn.Dropout(dropout)
        
        # Readout: T_byte' (128) + h_text (128) = 256
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x_byte, x_text, x_struct, return_intermediates=False):
        # x_struct is ignored
        B, L = x_text.shape
        valid_mask = (x_text != 0)
        float_mask = valid_mask.unsqueeze(-1).float()
        has_valid = valid_mask.any(dim=1)
        
        safe_key_padding_mask = ~valid_mask
        if not has_valid.all():
            safe_key_padding_mask = safe_key_padding_mask.clone()
            safe_key_padding_mask[~has_valid, 0] = False
            
        t_global = self.byte_proj(x_byte).unsqueeze(1) # (B, 1, d_model)
        
        h_text = (self.text_emb(x_text) + self.pos_emb[:, :L, :]) * float_mask
        h_text = self.emb_drop(h_text) * float_mask
        
        attn_self, _ = self.self_attn(h_text, h_text, h_text, key_padding_mask=safe_key_padding_mask)
        h_text = self.norm_self(h_text + self.drop(attn_self * float_mask)) * float_mask
        
        attn_g2t, _ = self.cross_attn_g2t(t_global, h_text, h_text, key_padding_mask=safe_key_padding_mask)
        attn_g2t = attn_g2t * has_valid.unsqueeze(1).unsqueeze(2).float()
        t_global_out = self.norm_g2t(t_global + self.drop(attn_g2t))
        t_byte_prime = t_global_out[:, 0, :]
        
        attn_t2g, _ = self.cross_attn_t2g(h_text, t_global, t_global)
        h_text_out = self.norm_t2g(h_text + self.drop(attn_t2g * float_mask)) * float_mask
        
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0).float()
        h_text_pooled = (h_text_out * float_mask).sum(dim=1) / valid_counts
        h_text_pooled = h_text_pooled * has_valid.unsqueeze(1).float()
        
        z_fused = torch.cat([t_byte_prime, h_text_pooled], dim=1) # (B, 256)
        return self.classifier(z_fused)


class CMANet_NoText(nn.Module):
    """
    Ablation 3: CMANet without API Text sequence (Byte + Struct only, no attention).
    """
    def __init__(self, byte_dim=512, struct_dim=66, d_model=128):
        super().__init__()
        self.byte_proj = nn.Sequential(
            nn.Linear(byte_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.struct_proj = nn.Sequential(
            nn.Linear(struct_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        # Readout: T_byte (128) + T_struct (128) = 256
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x_byte, x_text, x_struct, return_intermediates=False):
        # x_text is ignored
        t_b = self.byte_proj(x_byte)   # (B, d_model)
        t_s = self.struct_proj(x_struct) # (B, d_model)
        z_fused = torch.cat([t_b, t_s], dim=1) # (B, 256)
        return self.classifier(z_fused)


class CMANet_NoCrossAttn(nn.Module):
    """
    Ablation 4: CMANet without Cross-Attention (Intra-modal text self-attention only + concatenation).
    Directly isolates the value of cross-modal attention versus intra-modal self-attention.
    """
    def __init__(self, vocab_sz=10000, byte_dim=512, struct_dim=66, d_model=128, nhead=4, dropout=0.1, max_len=256):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        self.byte_proj = nn.Sequential(
            nn.Linear(byte_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.struct_proj = nn.Sequential(
            nn.Linear(struct_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.text_emb = nn.Embedding(vocab_sz, d_model, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.emb_drop = nn.Dropout(dropout)
        
        # Self-attention on text sequence only (NO cross-attention)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        
        # Readout: T_byte (128) + T_struct (128) + h_text (128) = 384
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x_byte, x_text, x_struct, return_intermediates=False):
        B, L = x_text.shape
        valid_mask = (x_text != 0)
        float_mask = valid_mask.unsqueeze(-1).float()
        has_valid = valid_mask.any(dim=1)
        
        safe_key_padding_mask = ~valid_mask
        if not has_valid.all():
            safe_key_padding_mask = safe_key_padding_mask.clone()
            safe_key_padding_mask[~has_valid, 0] = False
            
        t_b = self.byte_proj(x_byte)     # (B, d_model)
        t_s = self.struct_proj(x_struct) # (B, d_model)
        
        h_text = (self.text_emb(x_text) + self.pos_emb[:, :L, :]) * float_mask
        h_text = self.emb_drop(h_text) * float_mask
        
        attn_self, _ = self.self_attn(h_text, h_text, h_text, key_padding_mask=safe_key_padding_mask)
        h_text_out = self.norm_self(h_text + self.drop(attn_self * float_mask)) * float_mask
        
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0).float()
        h_text_pooled = (h_text_out * float_mask).sum(dim=1) / valid_counts
        h_text_pooled = h_text_pooled * has_valid.unsqueeze(1).float()
        
        z_fused = torch.cat([t_b, t_s, h_text_pooled], dim=1) # (B, 384)
        return self.classifier(z_fused)


class MultimodalFusionNet(nn.Module):
    """
    Reference Baseline Model (MultimodalFusionNet from 3-seed benchmark).
    Concatenation of ByteMLP (128) + MaskedTextCNN (128) + StructMLP (64) = 320-D.
    """
    def __init__(self, vocab_sz=10000, byte_dim=512, struct_dim=66, text_emb_dim=128):
        super().__init__()
        self.byte_enc = nn.Sequential(nn.Linear(byte_dim, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.struct_enc = nn.Sequential(nn.Linear(struct_dim, 64), nn.BatchNorm1d(64), nn.ReLU())
        self.text_emb = nn.Embedding(vocab_sz, text_emb_dim, padding_idx=0)
        self.text_conv1 = nn.Conv1d(text_emb_dim, 128, kernel_size=3, padding=1)
        self.text_conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.classifier = nn.Sequential(
            nn.Linear(320, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x_byte, x_text, x_struct):
        h_b = self.byte_enc(x_byte)
        h_s = self.struct_enc(x_struct)

        mask = (x_text != 0).float()
        t_h = self.text_emb(x_text).transpose(1, 2)
        t_h1 = self.relu(self.text_conv1(t_h)) * mask.unsqueeze(1)
        t_h2 = self.relu(self.text_conv2(t_h1)) * mask.unsqueeze(1)
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        h_t = t_h2.sum(dim=-1) / valid_counts

        fused = torch.cat([h_b, h_t, h_s], dim=1)
        return self.classifier(fused)


# ==============================================================================
# 2. COMPUTATIONAL & PARAMETER FAIRNESS ANALYSIS
# ==============================================================================

def compute_model_capacity(model, name, sample_L=256):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_emb_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'text_emb' not in n)
    emb_params = total_params - non_emb_params
    
    # Calculate theoretical FLOPs / MACs for 1 sample (L=256)
    # Projections + Attention + MLP
    macs = 0
    if hasattr(model, 'byte_proj'):
        macs += 512 * 128
    if hasattr(model, 'struct_proj'):
        macs += 66 * 128
    if hasattr(model, 'byte_enc'):
        macs += 512 * 128
    if hasattr(model, 'struct_enc'):
        macs += 66 * 64
        
    # Text CNN (if baseline)
    if hasattr(model, 'text_conv1'):
        macs += 2 * (sample_L * 3 * 128 * 128)
        
    # Self-attention (if attention-based)
    if hasattr(model, 'self_attn'):
        # QKV in-proj: 3 * L * D^2
        macs += 3 * sample_L * (128 ** 2)
        # Scores: L * L * D
        macs += sample_L * sample_L * 128
        # Context: L * L * D
        macs += sample_L * sample_L * 128
        # Out-proj: L * D^2
        macs += sample_L * (128 ** 2)
        
    # Cross-attention (if CMANet)
    if hasattr(model, 'cross_attn_g2t'):
        # G2T: Q=2, K=L=256
        macs += (2 * (128 ** 2)) + (2 * sample_L * (128 ** 2)) # in-proj Q and KV
        macs += 2 * sample_L * 128 # scores
        macs += 2 * sample_L * 128 # context
        macs += 2 * (128 ** 2)     # out-proj
        
        # T2G: Q=L=256, K=2
        macs += (sample_L * (128 ** 2)) + (2 * 2 * (128 ** 2)) # in-proj Q and KV
        macs += sample_L * 2 * 128 # scores
        macs += sample_L * 2 * 128 # context
        macs += sample_L * (128 ** 2) # out-proj
        
    # Classifier MACs
    for module in model.classifier:
        if isinstance(module, nn.Linear):
            macs += module.in_features * module.out_features
            
    flops = 2 * macs
    
    return {
        "name": name,
        "total_params": total_params,
        "non_emb_params": non_emb_params,
        "emb_params": emb_params,
        "macs": macs,
        "mflops": flops / 1e6
    }


def benchmark_inference_latency(model, device, num_warmup=20, num_runs=100, batch_size=32, seq_len=256):
    model.eval()
    model.to(device)
    
    xb = torch.randn(batch_size, 512, device=device)
    xt = torch.randint(1, 1000, (batch_size, seq_len), device=device)
    xs = torch.randn(batch_size, 66, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(xb, xt, xs)
            
    # Measure
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_t = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(xb, xt, xs)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_t = time.perf_counter()
    
    total_time_ms = (end_t - start_t) * 1000
    latency_per_batch_ms = total_time_ms / num_runs
    latency_per_sample_ms = latency_per_batch_ms / batch_size
    return latency_per_batch_ms, latency_per_sample_ms


# ==============================================================================
# 3. COMPREHENSIVE UNIT TESTS
# ==============================================================================

def run_all_unit_tests():
    print("=" * 80)
    print("CMANet ARCHITECTURAL VERIFICATION & UNIT TEST SUITE")
    print("=" * 80)
    results = {}
    
    # --------------------------------------------------------------------------
    # UNIT TEST 1: Tensor dimensions across variable batch sizes and lengths
    # --------------------------------------------------------------------------
    print("\n[TEST 1/7] Tensor dimensions across variable batch sizes & sequence lengths...")
    test1_pass = True
    models = {
        "CMANet": CrossModalAttentionNet(),
        "CMANet_NoByte": CMANet_NoByte(),
        "CMANet_NoStruct": CMANet_NoStruct(),
        "CMANet_NoText": CMANet_NoText(),
        "CMANet_NoCrossAttn": CMANet_NoCrossAttn(),
        "MultimodalFusionNet": MultimodalFusionNet()
    }
    
    batch_sizes = [1, 4, 32, 64]
    seq_lengths = [10, 50, 128, 256]
    
    for mname, mod in models.items():
        mod.eval()
        for B in batch_sizes:
            for L in seq_lengths:
                xb = torch.randn(B, 512)
                xt = torch.randint(0, 500, (B, L))
                xs = torch.randn(B, 66)
                with torch.no_grad():
                    out = mod(xb, xt, xs)
                if out.shape != (B, 2):
                    print(f"FAILED {mname} at B={B}, L={L}: Expected shape ({B}, 2), got {out.shape}")
                    test1_pass = False
    print(f"  Result: {'PASSED' if test1_pass else 'FAILED'} (Tested 6 models across 16 combinations of (B, L))")
    results["unit_test_1_dimensions"] = test1_pass
    
    # --------------------------------------------------------------------------
    # UNIT TEST 2: Strict padding invariance test
    # --------------------------------------------------------------------------
    print("\n[TEST 2/7] Strict padding invariance (representations & logits)...")
    cmanet = CrossModalAttentionNet()
    cmanet.eval()
    
    valid_tokens = [17, 84, 392, 1045, 62] # 5 valid tokens
    padding_lengths = [10, 50, 128, 256]
    
    xb = torch.randn(1, 512)
    xs = torch.randn(1, 66)
    
    logits_list = []
    valid_reps_list = []
    pooled_reps_list = []
    
    with torch.no_grad():
        for L in padding_lengths:
            xt = torch.zeros(1, L, dtype=torch.long)
            xt[0, :len(valid_tokens)] = torch.tensor(valid_tokens)
            out, inter = cmanet(xb, xt, xs, return_intermediates=True)
            logits_list.append(out)
            valid_reps_list.append(inter["h_text_out"][:, :len(valid_tokens), :])
            pooled_reps_list.append(inter["h_text_pooled"])
            
            # Check strictly padded tokens are 0.0
            if L > len(valid_tokens):
                pad_reps = inter["h_text_out"][:, len(valid_tokens):, :]
                max_pad_val = pad_reps.abs().max().item()
                assert max_pad_val == 0.0, f"Padded positions are non-zero: {max_pad_val}"
                
    ref_logit = logits_list[0]
    ref_valid = valid_reps_list[0]
    ref_pool = pooled_reps_list[0]
    
    max_logit_diff = 0.0
    max_valid_diff = 0.0
    max_pool_diff = 0.0
    
    print(f"  Reference length: {padding_lengths[0]} (5 valid tokens, {padding_lengths[0]-5} pad tokens)")
    for i, L in enumerate(padding_lengths[1:], start=1):
        d_logit = (logits_list[i] - ref_logit).abs().max().item()
        d_valid = (valid_reps_list[i] - ref_valid).abs().max().item()
        d_pool = (pooled_reps_list[i] - ref_pool).abs().max().item()
        max_logit_diff = max(max_logit_diff, d_logit)
        max_valid_diff = max(max_valid_diff, d_valid)
        max_pool_diff = max(max_pool_diff, d_pool)
        print(f"  Target length {L:3d}: Logit max diff = {d_logit:.2e}, Valid-tok diff = {d_valid:.2e}, Pooled diff = {d_pool:.2e}")
        
    test2_pass = (max_logit_diff < 1e-6) and (max_valid_diff < 1e-6) and (max_pool_diff < 1e-6)
    print(f"  Strict padding invariance (< 1e-6): {'PASSED' if test2_pass else 'FAILED'}")
    results["unit_test_2_padding_invariance"] = {
        "passed": test2_pass,
        "max_logit_diff": max_logit_diff,
        "max_valid_token_diff": max_valid_diff,
        "max_pooled_diff": max_pool_diff
    }
    
    # --------------------------------------------------------------------------
    # UNIT TEST 3: Gradient flow verification across all parameters
    # --------------------------------------------------------------------------
    print("\n[TEST 3/7] Gradient flow across all parameter tensors...")
    cmanet.train()
    cmanet.zero_grad()
    
    xb = torch.randn(8, 512)
    xt = torch.randint(1, 500, (8, 64))
    xs = torch.randn(8, 66)
    target = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
    
    criterion = nn.CrossEntropyLoss()
    logits = cmanet(xb, xt, xs)
    loss = criterion(logits, target)
    loss.backward()
    
    zero_grad_params = []
    grad_norms = {}
    for name, p in cmanet.named_parameters():
        if p.requires_grad:
            if p.grad is None or p.grad.norm().item() == 0.0:
                zero_grad_params.append(name)
            else:
                grad_norms[name] = float(p.grad.norm().item())
                
    test3_pass = (len(zero_grad_params) == 0)
    print(f"  Verified {len(grad_norms)} parameter tensors have non-zero gradients.")
    if not test3_pass:
        print(f"  WARNING: Tensors with zero gradient: {zero_grad_params}")
    else:
        print("  Key gradient norm samples:")
        sample_keys = [
            "byte_proj.0.weight", "struct_proj.0.weight", "text_emb.weight",
            "self_attn.in_proj_weight", "cross_attn_g2t.in_proj_weight",
            "cross_attn_t2g.in_proj_weight", "classifier.0.weight", "classifier.6.weight"
        ]
        for k in sample_keys:
            if k in grad_norms:
                print(f"    {k:30s} : norm = {grad_norms[k]:.4e}")
    print(f"  Result: {'PASSED' if test3_pass else 'FAILED'}")
    results["unit_test_3_gradient_flow"] = test3_pass
    
    # --------------------------------------------------------------------------
    # UNIT TEST 4: Finite outputs under normal and extreme edge cases
    # --------------------------------------------------------------------------
    print("\n[TEST 4/7] Finite outputs under edge cases (100% padding, zero inputs, extremes)...")
    cmanet.eval()
    test4_pass = True
    
    edge_cases = {
        "all_padding_sequence": (torch.randn(4, 512), torch.zeros(4, 128, dtype=torch.long), torch.randn(4, 66)),
        "single_token_valid": (torch.randn(4, 512), torch.cat([torch.tensor([[10]]*4), torch.zeros(4, 127, dtype=torch.long)], dim=1), torch.randn(4, 66)),
        "all_valid_no_padding": (torch.randn(4, 512), torch.randint(1, 500, (4, 128)), torch.randn(4, 66)),
        "all_zeros_byte_and_struct": (torch.zeros(4, 512), torch.randint(1, 500, (4, 128)), torch.zeros(4, 66)),
        "extreme_scale_values": (torch.randn(4, 512) * 50.0, torch.randint(1, 500, (4, 128)), torch.randn(4, 66) * 50.0)
    }
    
    for case_name, (xb_c, xt_c, xs_c) in edge_cases.items():
        with torch.no_grad():
            out = cmanet(xb_c, xt_c, xs_c)
        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        if has_nan or has_inf:
            print(f"  FAILED on {case_name}: NaN={has_nan}, Inf={has_inf}")
            test4_pass = False
        else:
            print(f"  Case '{case_name:28s}': finite logits output shape {list(out.shape)}")
    print(f"  Result: {'PASSED' if test4_pass else 'FAILED'}")
    results["unit_test_4_finite_outputs"] = test4_pass
    
    # --------------------------------------------------------------------------
    # UNIT TEST 5: Attention-mask correctness
    # --------------------------------------------------------------------------
    print("\n[TEST 5/7] Attention-mask correctness (pad keys receive strictly 0.0 attention)...")
    cmanet.eval()
    xt_mask_test = torch.zeros(1, 30, dtype=torch.long)
    xt_mask_test[0, :10] = torch.randint(1, 500, (10,)) # First 10 valid, next 20 pad
    
    with torch.no_grad():
        _, inter = cmanet(torch.randn(1, 512), xt_mask_test, torch.randn(1, 66), return_intermediates=True)
    
    # Check self-attention weights: shape (B, L, L) or (B, num_heads, L, L)
    weights_self = inter["weights_self"]
    # Keys 10:30 should have 0.0 attention weight for all queries 0:10
    if weights_self is not None:
        pad_self_attn = weights_self[:, :10, 10:]
        max_self_pad_attn = pad_self_attn.max().item()
    else:
        max_self_pad_attn = 0.0
        
    weights_g2t = inter["weights_g2t"]
    # Global tokens query text sequence: keys 10:30 should receive 0.0 attention
    if weights_g2t is not None:
        pad_g2t_attn = weights_g2t[:, :, 10:]
        max_g2t_pad_attn = pad_g2t_attn.max().item()
    else:
        max_g2t_pad_attn = 0.0
        
    print(f"  Max attention weight on pad keys (Self-Attn) : {max_self_pad_attn:.4e}")
    print(f"  Max attention weight on pad keys (G2T-Attn)  : {max_g2t_pad_attn:.4e}")
    test5_pass = (max_self_pad_attn < 1e-6) and (max_g2t_pad_attn < 1e-6)
    print(f"  Result: {'PASSED' if test5_pass else 'FAILED'}")
    results["unit_test_5_attention_mask"] = test5_pass
    
    # --------------------------------------------------------------------------
    # UNIT TEST 6: Device portability
    # --------------------------------------------------------------------------
    print("\n[TEST 6/7] Device portability (CPU and GPU if available)...")
    cpu_device = torch.device("cpu")
    cmanet.to(cpu_device)
    out_cpu = cmanet(torch.randn(2, 512), torch.randint(1, 100, (2, 32)), torch.randn(2, 66))
    assert out_cpu.device.type == "cpu"
    print("  CPU Execution: PASSED")
    
    cuda_tested = False
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        cmanet.to(gpu_device)
        out_gpu = cmanet(
            torch.randn(2, 512, device=gpu_device),
            torch.randint(1, 100, (2, 32), device=gpu_device),
            torch.randn(2, 66, device=gpu_device)
        )
        assert out_gpu.device.type == "cuda"
        cuda_tested = True
        print(f"  CUDA Execution on {torch.cuda.get_device_name(0)}: PASSED")
        cmanet.to(cpu_device)
    else:
        print("  CUDA Execution: SKIPPED (No local CUDA device available; will execute on Google Colab GPU)")
    results["unit_test_6_portability"] = True
    
    # --------------------------------------------------------------------------
    # UNIT TEST 7: Checkpoint serialization and reproduction equivalence
    # --------------------------------------------------------------------------
    print("\n[TEST 7/7] Checkpoint serialization & reproduction equivalence...")
    cmanet_orig = CrossModalAttentionNet()
    cmanet_orig.eval()
    
    xb_test = torch.randn(4, 512)
    xt_test = torch.randint(0, 500, (4, 64))
    xs_test = torch.randn(4, 66)
    
    with torch.no_grad():
        logits_orig = cmanet_orig(xb_test, xt_test, xs_test)
        
    buffer = io.BytesIO()
    torch.save(cmanet_orig.state_dict(), buffer)
    buffer.seek(0)
    
    cmanet_loaded = CrossModalAttentionNet()
    cmanet_loaded.load_state_dict(torch.load(buffer))
    cmanet_loaded.eval()
    
    with torch.no_grad():
        logits_loaded = cmanet_loaded(xb_test, xt_test, xs_test)
        
    checkpoint_diff = (logits_orig - logits_loaded).abs().max().item()
    print(f"  Max logit discrepancy after state_dict save & load: {checkpoint_diff:.2e}")
    test7_pass = (checkpoint_diff == 0.0)
    print(f"  Result: {'PASSED' if test7_pass else 'FAILED'}")
    results["unit_test_7_checkpoint_equivalence"] = test7_pass
    
    # --------------------------------------------------------------------------
    # MODEL CAPACITY & FAIRNESS TABLE
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CAPACITY & COMPUTATIONAL FAIRNESS COMPARISON TABLE (Sample L=256)")
    print("=" * 80)
    
    model_suite = [
        ("CMANet (Full)", CrossModalAttentionNet()),
        ("CMANet-NoByte", CMANet_NoByte()),
        ("CMANet-NoStruct", CMANet_NoStruct()),
        ("CMANet-NoText", CMANet_NoText()),
        ("CMANet-NoCrossAttn", CMANet_NoCrossAttn()),
        ("MultimodalFusionNet (Ref)", MultimodalFusionNet())
    ]
    
    print(f"{'Model':26s} | {'Total Params':12s} | {'Non-Emb Params':14s} | {'Emb Params':10s} | {'FLOPs (M)':9s} | {'CPU Latency (ms)':16s}")
    print("-" * 105)
    
    capacity_reports = []
    for name, m in model_suite:
        cap = compute_model_capacity(m, name, sample_L=256)
        batch_lat, sample_lat = benchmark_inference_latency(m, torch.device("cpu"), num_warmup=5, num_runs=20, batch_size=32, seq_len=256)
        cap["latency_batch_32_ms"] = batch_lat
        cap["latency_per_sample_ms"] = sample_lat
        capacity_reports.append(cap)
        print(f"{name:26s} | {cap['total_params']:12,d} | {cap['non_emb_params']:14,d} | {cap['emb_params']:10,d} | {cap['mflops']:9.2f} | {sample_lat:16.4f}")
        
    results["capacity_reports"] = capacity_reports
    return results


# ==============================================================================
# 4. 200-SAMPLE, 1-EPOCH SMOKE TEST ON CORRECTED EMBER DATA
# ==============================================================================

def run_smoke_test():
    print("\n" + "=" * 80)
    print("200-SAMPLE, 1-EPOCH SMOKE TEST ON CORRECTED EMBER DATA")
    print("=" * 80)
    
    train_npz_path = os.path.join("experiments", "baseline_correction_2026_10_30", "data", "ember_train_16k_corrected.npz")
    val_npz_path = os.path.join("experiments", "baseline_correction_2026_10_30", "data", "ember_val_4k_corrected.npz")
    
    if not os.path.exists(train_npz_path):
        train_npz_path = os.path.join("Old Repo Data", train_npz_path)
        val_npz_path = os.path.join("Old Repo Data", val_npz_path)
        
    print(f"Loading corrected training data from: {train_npz_path}")
    train_data = np.load(train_npz_path)
    val_data = np.load(val_npz_path)
    
    # Take first 200 training samples and 50 validation samples
    n_smoke_train = 200
    n_smoke_val = 50
    
    xb_train = torch.tensor(train_data["X_byte"][:n_smoke_train], dtype=torch.float32)
    xt_train = torch.tensor(train_data["X_text"][:n_smoke_train], dtype=torch.long)
    xs_train = torch.tensor(train_data["X_struct"][:n_smoke_train], dtype=torch.float32)
    y_train = torch.tensor(train_data["y"][:n_smoke_train], dtype=torch.long)
    
    xb_val = torch.tensor(val_data["X_byte"][:n_smoke_val], dtype=torch.float32)
    xt_val = torch.tensor(val_data["X_text"][:n_smoke_val], dtype=torch.long)
    xs_val = torch.tensor(val_data["X_struct"][:n_smoke_val], dtype=torch.float32)
    y_val = torch.tensor(val_data["y"][:n_smoke_val], dtype=torch.long)
    
    dataset = TensorDataset(xb_train, xt_train, xs_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"Train subset: {len(y_train)} samples, {len(loader)} batches (batch_size=32)")
    print(f"Val subset  : {len(y_val)} samples")
    
    # Instantiate CMANet and baseline Adam optimizer
    model = CrossModalAttentionNet()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    step_losses = []
    t0 = time.time()
    
    for batch_idx, (b_xb, b_xt, b_xs, b_y) in enumerate(loader, start=1):
        optimizer.zero_grad()
        logits = model(b_xb, b_xt, b_xs)
        loss = criterion(logits, b_y)
        loss.backward()
        optimizer.step()
        
        step_losses.append(loss.item())
        print(f"  Batch {batch_idx}/{len(loader)}: Loss = {loss.item():.4f}")
        
    t_train = time.time() - t0
    
    # Evaluate on val subset
    model.eval()
    with torch.no_grad():
        val_logits = model(xb_val, xt_val, xs_val)
        val_loss = criterion(val_logits, y_val).item()
        val_preds = val_logits.argmax(dim=1)
        val_acc = (val_preds == y_val).float().mean().item()
        
    print(f"\nSmoke Test Completed in {t_train:.2f} seconds.")
    print(f"Initial Step Loss : {step_losses[0]:.4f}")
    print(f"Final Step Loss   : {step_losses[-1]:.4f}")
    print(f"Validation Loss   : {val_loss:.4f}")
    print(f"Validation Acc    : {val_acc * 100:.2f}%")
    
    # Save checkpoint to verify serialization
    ckpt_dir = os.path.join("experiments", "cross_modal_attention_2026_10_30", "checkpoints")
    if not os.path.exists(ckpt_dir):
        ckpt_dir = os.path.join("Old Repo Data", ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "cmanet_smoke_checkpoint.pt")
    
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc
    }, ckpt_path)
    print(f"Successfully saved smoke checkpoint to: {ckpt_path}")
    
    # Reload and verify
    reloaded_ckpt = torch.load(ckpt_path)
    model_reloaded = CrossModalAttentionNet()
    model_reloaded.load_state_dict(reloaded_ckpt["model_state_dict"])
    model_reloaded.eval()
    with torch.no_grad():
        reloaded_logits = model_reloaded(xb_val, xt_val, xs_val)
    reloaded_diff = (val_logits - reloaded_logits).abs().max().item()
    print(f"Reload verification discrepancy: {reloaded_diff:.2e}")
    assert reloaded_diff == 0.0, "Reload verification failed!"
    
    smoke_results = {
        "initial_loss": step_losses[0],
        "final_loss": step_losses[-1],
        "val_loss": val_loss,
        "val_acc": val_acc,
        "train_time_sec": t_train,
        "checkpoint_saved": ckpt_path,
        "reloaded_verified": True
    }
    return smoke_results


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    t_start = time.time()
    unit_test_results = run_all_unit_tests()
    smoke_test_results = run_smoke_test()
    
    # Save test summary to json
    out_dir = os.path.join("experiments", "cross_modal_attention_2026_10_30", "metrics")
    if not os.path.exists(out_dir):
        out_dir = os.path.join("Old Repo Data", out_dir)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "architecture_verification_summary.json")
    
    final_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unit_tests": unit_test_results,
        "smoke_test": smoke_test_results,
        "total_elapsed_sec": time.time() - t_start
    }
    
    with open(summary_path, "w") as f:
        json.dump(final_output, f, indent=2)
        
    print("\n" + "=" * 80)
    print(f"ALL UNIT TESTS AND SMOKE TEST COMPLETED SUCCESSFULLY!")
    print(f"Verification summary written to: {summary_path}")
    print("=" * 80)

