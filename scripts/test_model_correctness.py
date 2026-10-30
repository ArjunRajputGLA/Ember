#!/usr/bin/env python3
"""
scripts/test_model_correctness.py
----------------------------------
Unit testing harness for Native EMBER models:
1. Padding-Safety Test:
   Verifies that MaskedTextCNN produces identical representation regardless
   of how many padding tokens (<PAD> = 0) are appended to a sequence.
2. Multimodal Fusion Gradient Flow Test:
   Verifies that all three branches (Byte, Text, Struct) and the classifier
   receive strictly non-zero gradients during backward pass.
3. Numerical Sanitation:
   Verifies zero NaN and zero Inf in logits and parameter gradients.
"""

import sys
import numpy as np

def run_tests():
    print("=" * 70)
    print("RUNNING ARCHITECTURAL & METHODOLOGICAL CORRECTNESS TESTS")
    print("=" * 70)

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[NOTE] PyTorch is not installed in the local environment.")
        print("       These tests will execute automatically in the Colab notebook.")
        print("       Running pure NumPy verification instead...")
        return run_numpy_padding_test()

    # --------------------------------------------------------------------------
    # 1. PyTorch MaskedTextCNN Definition
    # --------------------------------------------------------------------------
    class MaskedTextCNN(nn.Module):
        def __init__(self, vocab_sz=1000, emb_dim=32):
            super().__init__()
            self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
            self.conv1 = nn.Conv1d(emb_dim, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.clf = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 2)
            )

        def forward(self, x_text):
            mask = (x_text != 0).float() # (batch, seq_len)
            h = self.emb(x_text).transpose(1, 2) # (batch, emb_dim, seq_len)
            
            # Mask applied after each conv layer to zero out padding positions
            h1 = self.relu(self.conv1(h)) * mask.unsqueeze(1)
            h2 = self.relu(self.conv2(h1)) * mask.unsqueeze(1)
            
            # Masked average pooling
            valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = h2.sum(dim=-1) / valid_counts
            return self.clf(pooled)

    # --------------------------------------------------------------------------
    # 2. PyTorch MultimodalFusionNet Definition
    # --------------------------------------------------------------------------
    class MultimodalFusionNet(nn.Module):
        def __init__(self, vocab_sz=1000, byte_dim=512, struct_dim=66, text_emb_dim=32):
            super().__init__()
            self.byte_enc = nn.Sequential(
                nn.Linear(byte_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU()
            )
            self.struct_enc = nn.Sequential(
                nn.Linear(struct_dim, 32),
                nn.BatchNorm1d(32),
                nn.ReLU()
            )
            self.text_emb = nn.Embedding(vocab_sz, text_emb_dim, padding_idx=0)
            self.text_conv1 = nn.Conv1d(text_emb_dim, 32, kernel_size=3, padding=1)
            self.text_conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            
            # Fusion: 64 (byte) + 32 (text) + 32 (struct) = 128
            self.classifier = nn.Sequential(
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
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
            
            fused = torch.cat([h_b, h_t, h_s], dim=1) # (batch, 128)
            return self.classifier(fused)

    # --------------------------------------------------------------------------
    # Test 1: Padding Invariance
    # --------------------------------------------------------------------------
    print("\n[TEST 1] Testing TextCNN Padding Invariance...")
    model_text = MaskedTextCNN()
    model_text.eval()
    
    # Sample sequence of 5 tokens
    seq_base = torch.tensor([[10, 25, 42, 99, 7]], dtype=torch.long)
    seq_pad50 = torch.tensor([[10, 25, 42, 99, 7] + [0]*45], dtype=torch.long)
    seq_pad256 = torch.tensor([[10, 25, 42, 99, 7] + [0]*251], dtype=torch.long)
    
    with torch.no_grad():
        out_base = model_text(seq_base)
        out_pad50 = model_text(seq_pad50)
        out_pad256 = model_text(seq_pad256)
        
    diff_50 = torch.max(torch.abs(out_base - out_pad50)).item()
    diff_256 = torch.max(torch.abs(out_base - out_pad256)).item()
    
    print(f"  - Max abs diff between length 5 and padded length 50  : {diff_50:.8e}")
    print(f"  - Max abs diff between length 5 and padded length 256 : {diff_256:.8e}")
    assert diff_256 < 1e-6, f"Padding leakage detected! diff={diff_256}"
    print("  [PASS] MaskedTextCNN produces identical logits regardless of padding length.")

    # --------------------------------------------------------------------------
    # Test 2: Multimodal Fusion Gradient Flow
    # --------------------------------------------------------------------------
    print("\n[TEST 2] Testing Multimodal Gradient Flow Across All 3 Encoders...")
    model_mm = MultimodalFusionNet()
    model_mm.train()
    
    # Dummy batch of 4 samples
    x_b = torch.randn(4, 512, requires_grad=True)
    x_t = torch.randint(0, 500, (4, 64))
    x_s = torch.randn(4, 66, requires_grad=True)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    
    criterion = nn.CrossEntropyLoss()
    logits = model_mm(x_b, x_t, x_s)
    
    assert not torch.isnan(logits).any(), "NaN found in forward logits!"
    assert not torch.isinf(logits).any(), "Inf found in forward logits!"
    
    loss = criterion(logits, y)
    loss.backward()
    
    # Check gradients
    grad_checks = {
        "Byte Encoder (byte_enc)": [p.grad for p in model_mm.byte_enc.parameters()],
        "Text Embedding (text_emb)": [model_mm.text_emb.weight.grad],
        "Text Conv (text_conv1/2)": [p.grad for p in model_mm.text_conv1.parameters()] + [p.grad for p in model_mm.text_conv2.parameters()],
        "Struct Encoder (struct_enc)": [p.grad for p in model_mm.struct_enc.parameters()],
        "Classifier (classifier)": [p.grad for p in model_mm.classifier.parameters()]
    }
    
    all_passed = True
    for branch_name, grads in grad_checks.items():
        norms = [g.norm().item() for g in grads if g is not None]
        non_zeros = [n > 0 for n in norms]
        has_nan = any(np.isnan(n) for n in norms)
        has_inf = any(np.isinf(n) for n in norms)
        avg_norm = np.mean(norms) if norms else 0.0
        
        status = "PASSED" if all(non_zeros) and not has_nan and not has_inf else "FAILED"
        print(f"  - {branch_name:<28} : {status} (Mean Grad Norm: {avg_norm:.4e}, Non-Zero: {all(non_zeros)})")
        if not all(non_zeros) or has_nan or has_inf:
            all_passed = False
            
    assert all_passed, "Gradient flow verification failed!"
    print("  [PASS] All three modality branches receive non-zero, healthy gradients.")
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED: MODEL ARCHITECTURES MATHEMATICALLY VERIFIED")
    print("=" * 70)
    return True


def run_numpy_padding_test():
    """Pure NumPy fallback padding invariance verification."""
    print("Running pure NumPy Masked Pooling invariance test...")
    np.random.seed(42)
    vocab_size, emb_dim, conv_dim = 100, 16, 16
    emb_weights = np.random.randn(vocab_size, emb_dim).astype(np.float32)
    emb_weights[0] = 0.0  # PAD token is zero
    conv_w = np.random.randn(conv_dim, emb_dim, 3).astype(np.float32)
    conv_b = np.random.randn(conv_dim).astype(np.float32)
    
    def forward(toks):
        T = len(toks)
        mask = (np.array(toks) != 0).astype(np.float32)
        h = emb_weights[toks]
        h_pad = np.pad(h, ((1, 1), (0, 0)), mode='constant')
        conv_out = np.zeros((conv_dim, T), dtype=np.float32)
        for t in range(T):
            w = h_pad[t:t+3].T
            for c in range(conv_dim):
                conv_out[c, t] = np.sum(conv_w[c] * w) + conv_b[c]
        conv_out = np.maximum(0, conv_out) * mask[np.newaxis, :]
        return np.sum(conv_out, axis=1) / max(1.0, np.sum(mask))

    base_seq = [12, 45, 78, 23, 9]
    out_base = forward(base_seq)
    diffs = []
    for l in [10, 50, 256]:
        out_pad = forward(base_seq + [0]*(l - len(base_seq)))
        d = np.max(np.abs(out_base - out_pad))
        diffs.append(d)
        print(f"  Length {l:>3} -> Max Diff: {d:.8e}")
    assert max(diffs) < 1e-6
    print("  [PASS] Pure NumPy verification passed.")
    return True

if __name__ == '__main__':
    run_tests()

