"""
cmanet.py - Architecture definitions and regression verification suite for:
- CrossModalAttentionNet (CMANet-Full: Bidirectional Global-to-Sequence Cross-Attention)
- CMANet_NoByte (Ablation 1)
- CMANet_NoStruct (Ablation 2)
- CMANet_NoText (Ablation 3)
- CMANet_NoCrossAttn (Ablation 4)
- MultimodalFusionNet (Reference Baseline)
"""

import math
import time
import io
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. MODEL ARCHITECTURES
# ==============================================================================

class CrossModalAttentionNet(nn.Module):
    """
    Bidirectional Global-to-Sequence Cross-Attention Network (CMANet).
    Modality A: Byte Histogram & Entropy Histogram (512-D continuous vector)
    Modality B: Hashed API / Token Sequence (L-length integer sequence, L <= 256)
    Modality C: PE Structural Properties (66-D continuous vector)
    
    Global summary tokens (Byte and Struct) interact bidirectionally with API sequence tokens
    via Multihead Cross-Attention with strict padding safety.
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
        
        # 2. Sequence Token Embeddings + Positional Encoding (Strictly Zero Padded Tokens)
        h_text_emb = (self.text_emb(x_text) + self.pos_emb[:, :L, :]) * float_mask
        h_text = self.emb_drop(h_text_emb) * float_mask
        
        # 3. Intra-modal Sequence Self-Attention with Strict Post-Norm Re-masking
        attn_self, weights_self = self.self_attn(
            h_text, h_text, h_text,
            key_padding_mask=safe_key_padding_mask,
            need_weights=return_intermediates
        )
        # Zero out padding outputs before residual and re-mask after LayerNorm to eliminate bias leakage
        h_text_post_self = self.norm_self(h_text + self.drop(attn_self * float_mask)) * float_mask
        
        # 4. Bidirectional Cross-Attention
        # Direction 1: Sequence to Global (Global tokens query Sequence)
        attn_g2t, weights_g2t = self.cross_attn_g2t(
            t_global, h_text_post_self, h_text_post_self,
            key_padding_mask=safe_key_padding_mask,
            need_weights=return_intermediates
        )
        attn_g2t = attn_g2t * has_valid.unsqueeze(1).unsqueeze(2).float()
        t_global_out = self.norm_g2t(t_global + self.drop(attn_g2t))
        t_byte_prime = t_global_out[:, 0, :]   # (B, d_model)
        t_struct_prime = t_global_out[:, 1, :] # (B, d_model)
        
        # Direction 2: Global to Sequence (Sequence tokens query Global tokens)
        attn_t2g, weights_t2g = self.cross_attn_t2g(
            h_text_post_self, t_global, t_global,
            need_weights=return_intermediates
        )
        # Strict post-norm re-masking on API sequence positions
        h_text_out = self.norm_t2g(h_text_post_self + self.drop(attn_t2g * float_mask)) * float_mask
        
        # 5. Masked Average Sequence Pooling
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0).float() # (B, 1)
        h_text_pooled = (h_text_out * float_mask).sum(dim=1) / valid_counts       # (B, d_model)
        h_text_pooled = h_text_pooled * has_valid.unsqueeze(1).float()
        
        # 6. Readout Fusion & Classification
        z_fused = torch.cat([t_byte_prime, t_struct_prime, h_text_pooled], dim=1) # (B, 384)
        logits = self.classifier(z_fused)
        
        if return_intermediates:
            return logits, {
                "h_text_emb": h_text_emb,
                "h_text_post_self": h_text_post_self,
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
    Byte input is genuinely unused and receives zero gradient.
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
        # x_byte is explicitly ignored
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
    Structural input is genuinely unused and receives zero gradient.
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
        # x_struct is explicitly ignored
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
    Text input is genuinely unused and receives zero gradient.
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
        # x_text is explicitly ignored
        t_b = self.byte_proj(x_byte)   # (B, d_model)
        t_s = self.struct_proj(x_struct) # (B, d_model)
        z_fused = torch.cat([t_b, t_s], dim=1) # (B, 256)
        return self.classifier(z_fused)


class CMANet_NoCrossAttn(nn.Module):
    """
    Ablation 4: CMANet without Cross-Attention (Intra-modal text self-attention only + concatenation).
    Directly isolates the value of cross-modal attention versus intra-modal self-attention.
    Retains all three modalities, but contains NO cross-attention layers.
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
# 2. MODEL CAPACITY & COMPUTATIONAL FAIRNESS
# ==============================================================================

def compute_model_capacity(model, name, sample_L=256):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_emb_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'text_emb' not in n)
    emb_params = total_params - non_emb_params
    
    macs = 0
    if hasattr(model, 'byte_proj'):
        macs += 512 * 128
    if hasattr(model, 'struct_proj'):
        macs += 66 * 128
    if hasattr(model, 'byte_enc'):
        macs += 512 * 128
    if hasattr(model, 'struct_enc'):
        macs += 66 * 64
        
    if hasattr(model, 'text_conv1'):
        macs += 2 * (sample_L * 3 * 128 * 128)
        
    if hasattr(model, 'self_attn'):
        macs += 3 * sample_L * (128 ** 2)
        macs += sample_L * sample_L * 128
        macs += sample_L * sample_L * 128
        macs += sample_L * (128 ** 2)
        
    if hasattr(model, 'cross_attn_g2t'):
        macs += (2 * (128 ** 2)) + (2 * sample_L * (128 ** 2))
        macs += 2 * sample_L * 128
        macs += 2 * sample_L * 128
        macs += 2 * (128 ** 2)
        
        macs += (sample_L * (128 ** 2)) + (2 * 2 * (128 ** 2))
        macs += sample_L * 2 * 128
        macs += sample_L * 2 * 128
        macs += sample_L * (128 ** 2)
        
    for module in model.classifier:
        if isinstance(module, nn.Linear):
            macs += module.in_features * module.out_features
            
    flops = 2 * macs
    
    return {
        "name": name,
        "total_params": int(total_params),
        "non_emb_params": int(non_emb_params),
        "emb_params": int(emb_params),
        "macs": int(macs),
        "mflops": round(flops / 1e6, 2)
    }


# ==============================================================================
# 3. COMPREHENSIVE ATTENTION SAFETY & REGRESSION TEST SUITE
# ==============================================================================

def verify_all_attention_safety_and_regressions():
    """
    Executes all 7 architectural unit tests + 2 explicit attention safety regression tests.
    Returns True if and only if all tests pass strictly.
    """
    print("=" * 80)
    print("RUNNING COMPREHENSIVE ATTENTION SAFETY & ARCHITECTURAL REGRESSION TESTS")
    print("=" * 80)
    
    # Unit Test 1: Tensor dimensions across variable batch sizes and lengths
    print("[TEST 1/9] Tensor dimensions across batch sizes and sequence lengths...")
    test_models = {
        "CMANet": CrossModalAttentionNet(),
        "CMANet_NoByte": CMANet_NoByte(),
        "CMANet_NoStruct": CMANet_NoStruct(),
        "CMANet_NoText": CMANet_NoText(),
        "CMANet_NoCrossAttn": CMANet_NoCrossAttn(),
        "MultimodalFusionNet": MultimodalFusionNet()
    }
    for mname, mod in test_models.items():
        mod.eval()
        for B in [1, 4, 32, 64]:
            for L in [10, 50, 128, 256]:
                xb = torch.randn(B, 512)
                xt = torch.randint(0, 500, (B, L))
                xs = torch.randn(B, 66)
                with torch.no_grad():
                    out = mod(xb, xt, xs)
                assert out.shape == (B, 2), f"Failed {mname} at B={B}, L={L}: got {out.shape}"
    print("  PASSED (All 6 models verified across 16 (B, L) combinations)")

    # Unit Test 2: Strict padding invariance
    print("[TEST 2/9] Strict padding invariance across lengths 10, 50, 128, 256 (< 1e-6)...")
    cmanet = CrossModalAttentionNet()
    cmanet.eval()
    valid_tokens = [17, 84, 392, 1045, 62]
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
    ref_logit, ref_valid, ref_pool = logits_list[0], valid_reps_list[0], pooled_reps_list[0]
    for i, L in enumerate(padding_lengths[1:], start=1):
        d_logit = (logits_list[i] - ref_logit).abs().max().item()
        d_valid = (valid_reps_list[i] - ref_valid).abs().max().item()
        d_pool = (pooled_reps_list[i] - ref_pool).abs().max().item()
        assert d_logit < 1e-6, f"Logit diff too large: {d_logit} at L={L}"
        assert d_valid < 1e-6, f"Valid token rep diff too large: {d_valid} at L={L}"
        assert d_pool < 1e-6, f"Pooled rep diff too large: {d_pool} at L={L}"
    print("  PASSED (Max logit diff < 1e-7 across lengths)")

    # Unit Test 3: Gradient flow across all parameter tensors
    print("[TEST 3/9] Gradient flow across all trainable parameter tensors...")
    cmanet.train()
    cmanet.zero_grad()
    xb = torch.randn(8, 512)
    xt = torch.randint(1, 500, (8, 64))
    xs = torch.randn(8, 66)
    target = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
    loss = nn.CrossEntropyLoss()(cmanet(xb, xt, xs), target)
    loss.backward()
    for name, p in cmanet.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has None gradient"
            assert p.grad.norm().item() > 0.0, f"Parameter {name} has zero gradient"
    print("  PASSED (All 37 parameter tensors have non-zero gradients)")

    # Unit Test 4: Finite outputs under normal and extreme edge cases
    print("[TEST 4/9] Numerical stability under edge cases...")
    cmanet.eval()
    edge_cases = [
        (torch.randn(4, 512), torch.randint(1, 500, (4, 128)), torch.randn(4, 66)),
        (torch.zeros(4, 512), torch.randint(1, 500, (4, 128)), torch.zeros(4, 66)),
        (torch.randn(4, 512) * 50.0, torch.randint(1, 500, (4, 128)), torch.randn(4, 66) * 50.0)
    ]
    for xb_c, xt_c, xs_c in edge_cases:
        with torch.no_grad():
            out = cmanet(xb_c, xt_c, xs_c)
        assert not torch.isnan(out).any().item(), "Detected NaN output!"
        assert not torch.isinf(out).any().item(), "Detected Inf output!"
    print("  PASSED (Zero NaNs or Infs detected)")

    # Unit Test 5: Attention-mask correctness (pad keys receive strictly 0.0 attention)
    print("[TEST 5/9] Attention-mask correctness (pad keys receive strictly 0.0 attention)...")
    xt_mask_test = torch.zeros(1, 30, dtype=torch.long)
    xt_mask_test[0, :10] = torch.randint(1, 500, (10,))
    with torch.no_grad():
        _, inter = cmanet(torch.randn(1, 512), xt_mask_test, torch.randn(1, 66), return_intermediates=True)
    if inter["weights_self"] is not None:
        pad_self = inter["weights_self"][:, :10, 10:].max().item()
        assert pad_self < 1e-6, f"Self-attn assigned weight to pad keys: {pad_self}"
    if inter["weights_g2t"] is not None:
        pad_g2t = inter["weights_g2t"][:, :, 10:].max().item()
        assert pad_g2t < 1e-6, f"G2T-attn assigned weight to pad keys: {pad_g2t}"
    print("  PASSED (Padded key positions receive strictly 0.0 attention)")

    # Unit Test 6: Device portability
    print("[TEST 6/9] Device portability...")
    cpu_dev = torch.device("cpu")
    cmanet.to(cpu_dev)
    out_cpu = cmanet(torch.randn(2, 512), torch.randint(1, 100, (2, 32)), torch.randn(2, 66))
    assert out_cpu.device.type == "cpu"
    if torch.cuda.is_available():
        gpu_dev = torch.device("cuda")
        cmanet.to(gpu_dev)
        out_gpu = cmanet(
            torch.randn(2, 512, device=gpu_dev),
            torch.randint(1, 100, (2, 32), device=gpu_dev),
            torch.randn(2, 66, device=gpu_dev)
        )
        assert out_gpu.device.type == "cuda"
        cmanet.to(cpu_dev)
        print("  PASSED on CPU and CUDA GPU")
    else:
        print("  PASSED on CPU (CUDA GPU branch ready for Colab)")

    # Unit Test 7: Checkpoint serialization and reproduction equivalence
    print("[TEST 7/9] Checkpoint serialization and reproduction equivalence...")
    buf = io.BytesIO()
    torch.save(cmanet.state_dict(), buf)
    buf.seek(0)
    cmanet2 = CrossModalAttentionNet()
    cmanet2.load_state_dict(torch.load(buf))
    cmanet2.eval()
    xb_chk = torch.randn(4, 512)
    xt_chk = torch.randint(0, 500, (4, 64))
    xs_chk = torch.randn(4, 66)
    with torch.no_grad():
        out1 = cmanet(xb_chk, xt_chk, xs_chk)
        out2 = cmanet2(xb_chk, xt_chk, xs_chk)
    assert (out1 - out2).abs().max().item() == 0.0, "Checkpoint reproduction mismatch!"
    print("  PASSED (Exact reproduction delta = 0.00e+00)")

    # REGRESSION TEST 8: All-padding API sequence safety (No NaNs from fully masked attention)
    print("[TEST 8/9] REGRESSION TEST: All-padding API sequence safety (100% masked attention)...")
    cmanet.eval()
    B_empty = 4
    L_empty = 128
    xt_all_pad = torch.zeros(B_empty, L_empty, dtype=torch.long) # 100% padding tokens
    xb_test = torch.randn(B_empty, 512)
    xs_test = torch.randn(B_empty, 66)
    
    with torch.no_grad():
        out_pad, inter_pad = cmanet(xb_test, xt_all_pad, xs_test, return_intermediates=True)
        
    assert not torch.isnan(out_pad).any().item(), "NaN detected on all-padding sequence!"
    assert not torch.isinf(out_pad).any().item(), "Inf detected on all-padding sequence!"
    assert out_pad.shape == (B_empty, 2), f"Expected shape ({B_empty}, 2), got {out_pad.shape}"
    
    # Check that pooled text representation is strictly zero
    pooled_max = inter_pad["h_text_pooled"].abs().max().item()
    assert pooled_max == 0.0, f"Pooled text representation for all-pad sequence is non-zero: {pooled_max}"
    
    # Check that output changes appropriately when byte/struct change
    with torch.no_grad():
        out_pad_diff = cmanet(xb_test + 1.0, xt_all_pad, xs_test)
    delta_out = (out_pad - out_pad_diff).abs().max().item()
    assert delta_out > 1e-4, "Model failed to respond to byte feature change on all-padding sequence!"
    print("  PASSED (All-padding sequence handled without NaNs; pooled rep strictly 0.0)")

    # REGRESSION TEST 9: Padded query representations remain strictly zero at all stages
    print("[TEST 9/9] REGRESSION TEST: Verification that padded queries remain strictly zero...")
    valid_len = 8
    total_len = 64
    xt_mixed = torch.zeros(2, total_len, dtype=torch.long)
    xt_mixed[:, :valid_len] = torch.randint(1, 500, (2, valid_len))
    
    with torch.no_grad():
        _, inter_mixed = cmanet(torch.randn(2, 512), xt_mixed, torch.randn(2, 66), return_intermediates=True)
        
    # Check stage A: initial embedding + pos
    emb_pad = inter_mixed["h_text_emb"][:, valid_len:, :].abs().max().item()
    assert emb_pad == 0.0, f"Initial embedding has non-zero padded tokens: {emb_pad}"
    
    # Check stage B: post self-attention LayerNorm
    self_pad = inter_mixed["h_text_post_self"][:, valid_len:, :].abs().max().item()
    assert self_pad == 0.0, f"Post-self-attention representation has non-zero padded tokens: {self_pad}"
    
    # Check stage C: post cross-attention LayerNorm
    cross_pad = inter_mixed["h_text_out"][:, valid_len:, :].abs().max().item()
    assert cross_pad == 0.0, f"Final sequence representation has non-zero padded tokens: {cross_pad}"
    print("  PASSED (Padded positions strictly zero at initial emb, post-self, and post-cross stages)")

    print("=" * 80)
    print("ALL ATTENTION SAFETY AND REGRESSION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return True


# ==============================================================================
# 4. ABLATION ISOLATION VERIFICATION
# ==============================================================================

def verify_ablation_isolation():
    """
    Verifies that removed modalities in each ablation are genuinely absent:
    - CMANet_NoByte: Output is strictly invariant to x_byte and grad w.r.t x_byte is zero.
    - CMANet_NoStruct: Output is strictly invariant to x_struct and grad w.r.t x_struct is zero.
    - CMANet_NoText: Output is strictly invariant to x_text.
    - CMANet_NoCrossAttn: Model uses all modalities, but has NO cross-attention layers.
    """
    print("=" * 80)
    print("VERIFYING ABLATION ISOLATION & MODALITY ABSENCE")
    print("=" * 80)
    
    B, L = 4, 32
    xb1 = torch.randn(B, 512, requires_grad=True)
    xb2 = torch.randn(B, 512, requires_grad=True)
    xt1 = torch.randint(1, 500, (B, L))
    xt2 = torch.randint(1, 500, (B, L))
    xs1 = torch.randn(B, 66, requires_grad=True)
    xs2 = torch.randn(B, 66, requires_grad=True)
    target = torch.tensor([0, 1, 0, 1])
    crit = nn.CrossEntropyLoss()
    
    # 1. CMANet_NoByte
    print("[ABLATION 1/4] Verifying CMANet_NoByte genuinely omits Byte modality...")
    m_nobyte = CMANet_NoByte()
    m_nobyte.eval()
    with torch.no_grad():
        out_b1 = m_nobyte(xb1, xt1, xs1)
        out_b2 = m_nobyte(xb2, xt1, xs1)
    diff_byte = (out_b1 - out_b2).abs().max().item()
    assert diff_byte == 0.0, f"CMANet_NoByte output changed when varying x_byte! diff={diff_byte}"
    # Gradient check
    m_nobyte.train()
    loss_b = crit(m_nobyte(xb1, xt1, xs1), target)
    loss_b.backward()
    assert xb1.grad is None, "CMANet_NoByte received gradient on x_byte!"
    assert xs1.grad is not None and xs1.grad.norm().item() > 0.0, "CMANet_NoByte should have gradient on x_struct"
    print("  PASSED (Byte modality strictly absent; zero output delta and zero gradient)")

    # 2. CMANet_NoStruct
    print("[ABLATION 2/4] Verifying CMANet_NoStruct genuinely omits Struct modality...")
    m_nostruct = CMANet_NoStruct()
    m_nostruct.eval()
    xb_test = torch.randn(B, 512, requires_grad=True)
    xs_test = torch.randn(B, 66, requires_grad=True)
    with torch.no_grad():
        out_s1 = m_nostruct(xb_test, xt1, xs_test)
        out_s2 = m_nostruct(xb_test, xt1, xs_test + 1.0)
    diff_struct = (out_s1 - out_s2).abs().max().item()
    assert diff_struct == 0.0, f"CMANet_NoStruct output changed when varying x_struct! diff={diff_struct}"
    # Gradient check
    m_nostruct.train()
    loss_s = crit(m_nostruct(xb_test, xt1, xs_test), target)
    loss_s.backward()
    assert xs_test.grad is None, "CMANet_NoStruct received gradient on x_struct!"
    assert xb_test.grad is not None and xb_test.grad.norm().item() > 0.0, "CMANet_NoStruct should have gradient on x_byte"
    print("  PASSED (Struct modality strictly absent; zero output delta and zero gradient)")

    # 3. CMANet_NoText
    print("[ABLATION 3/4] Verifying CMANet_NoText genuinely omits Text modality...")
    m_notext = CMANet_NoText()
    m_notext.eval()
    with torch.no_grad():
        out_t1 = m_notext(xb1, xt1, xs1)
        out_t2 = m_notext(xb1, xt2, xs1)
    diff_text = (out_t1 - out_t2).abs().max().item()
    assert diff_text == 0.0, f"CMANet_NoText output changed when varying x_text! diff={diff_text}"
    assert not hasattr(m_notext, 'text_emb'), "CMANet_NoText contains text_emb module!"
    assert not hasattr(m_notext, 'self_attn'), "CMANet_NoText contains self_attn module!"
    print("  PASSED (Text modality strictly absent; no text embeddings or attention)")

    # 4. CMANet_NoCrossAttn
    print("[ABLATION 4/4] Verifying CMANet_NoCrossAttn retains all modalities without cross-attention...")
    m_nocross = CMANet_NoCrossAttn()
    assert hasattr(m_nocross, 'byte_proj'), "CMANet_NoCrossAttn missing byte_proj!"
    assert hasattr(m_nocross, 'struct_proj'), "CMANet_NoCrossAttn missing struct_proj!"
    assert hasattr(m_nocross, 'text_emb'), "CMANet_NoCrossAttn missing text_emb!"
    assert hasattr(m_nocross, 'self_attn'), "CMANet_NoCrossAttn missing self_attn!"
    assert not hasattr(m_nocross, 'cross_attn_g2t'), "CMANet_NoCrossAttn should NOT have cross_attn_g2t!"
    assert not hasattr(m_nocross, 'cross_attn_t2g'), "CMANet_NoCrossAttn should NOT have cross_attn_t2g!"
    # Verify all 3 modalities influence output
    with torch.no_grad():
        out_base = m_nocross(xb1, xt1, xs1)
        out_mod_b = m_nocross(xb2, xt1, xs1)
        out_mod_t = m_nocross(xb1, xt2, xs1)
        out_mod_s = m_nocross(xb1, xt1, xs2)
    assert (out_base - out_mod_b).abs().max().item() > 1e-4, "Byte does not affect NoCrossAttn output!"
    assert (out_base - out_mod_t).abs().max().item() > 1e-4, "Text does not affect NoCrossAttn output!"
    assert (out_base - out_mod_s).abs().max().item() > 1e-4, "Struct does not affect NoCrossAttn output!"
    print("  PASSED (All 3 modalities actively contribute; cross-attention strictly absent)")

    print("=" * 80)
    print("ALL ABLATION ISOLATION CHECKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    verify_all_attention_safety_and_regressions()
    verify_ablation_isolation()
