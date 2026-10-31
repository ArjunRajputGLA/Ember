# Comprehensive Research Audit: Bidirectional Cross-Modal Attention (CMANet) vs. Multimodal Concatenation on EMBER 2018

**Project:** Ember – Malware Detection Using Native Multimodal Learning  
**Working Branch:** `experiment/native-multimodal-transformer`  
**Dataset:** EMBER 2018 v2 (Corrected 16k Train / 4k Validation Disjoint Splits)  
**Execution Environment:** Windows 11, AMD64 Family 25 Model 80 (16 Logical Threads, 15.4 GB RAM), PyTorch 2.15.0+cpu  
**Audit Date:** October 31, 2026  
**Status:** Post-Experiment Audit Complete (All 18 Benchmark Runs Analyzed)  

---

## Executive Summary & Core Research Finding

### The Core Research Question
> *Why did CMANet-Full fail to outperform MultimodalFusionNet and CMANet-NoCrossAttn despite its additional cross-modal attention layers?*

### Direct Empirical Finding
Across 18 fully executed, methodologically controlled training runs evaluated on identical 4,000 validation samples across three random seeds (`[42, 123, 2026]`), **the central cross-modal attention hypothesis was NOT supported by empirical evidence**:

1. **CMANet-Full vs. MultimodalFusionNet (Baseline Concatenation):**
   - **CMANet-Full:** $92.38 \pm 0.30\%$ Accuracy, $0.9789 \pm 0.0012$ ROC-AUC, $154.7 \pm 21.4$ False Negatives.
   - **MultimodalFusionNet:** $92.96 \pm 0.48\%$ Accuracy, $0.9801 \pm 0.0012$ ROC-AUC, $154.3 \pm 8.1$ False Negatives.
   - **Statistical Result:** Across all three seeds, the 95% bootstrap confidence intervals for ROC-AUC difference cross zero ($\Delta \text{AUC} \in [-0.0040, +0.0013], p \ge 0.27$). For accuracy, MultimodalFusionNet slightly outperforms CMANet-Full ($\Delta \text{Acc} \approx -0.58\%$). MultimodalFusionNet achieves this while requiring **$8.7\times$ less training time** and **$50\%$ fewer FLOPs**.
2. **CMANet-Full vs. CMANet-NoCrossAttn (Cross-Attention Isolation):**
   - **CMANet-NoCrossAttn:** $92.42 \pm 0.25\%$ Accuracy, $0.9785 \pm 0.0014$ ROC-AUC, $162.3 \pm 23.6$ False Negatives.
   - **Statistical Result:** The difference between full cross-modal attention and self-attention + concatenation is **statistically indistinguishable from zero** ($\Delta \text{Acc} = -0.04\%, 95\%\text{ CI: } [-0.77\%, +0.63\%], p = 0.8696$ on Seed 42; $p = 0.2990$ on Seed 123; $p = 0.2404$ on Seed 2026).
3. **Multimodal vs. Unimodal Baseline:**
   - Both CMANet-Full and MultimodalFusionNet statistically significantly outperform the top unimodal baseline `StructMLP` ($89.99 \pm 0.58\%$ Acc, $p < 0.0002$, $95\%\text{ CI}$ strictly above zero). Multimodal learning itself provides genuine value, but **cross-modal attention adds no benefit over simple concatenation**.

---

## Phase 1 — Statistical Audit & Paired Bootstrap Analysis

### 1.1 Complete 18-Run Experimental Telemetry

Every model was trained with strictly identical hyperparameters: Adam optimizer ($\text{lr}=0.001, \text{weight decay}=0.0001$), $\text{batch size}=32$, 10 epochs, and best validation loss checkpointing. All runs were evaluated on the exact same 4,000 validation samples.

| Model Name | Seed | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC | FPR@99%TPR (%) | Malware FN | Training Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CMANet_Full** | 42 | 92.17 | 94.69 | 89.35 | 91.94 | 0.9786 | 29.80 | 146 | 1569.7 |
| **CMANet_Full** | 123 | 92.73 | 94.94 | 90.25 | 92.53 | 0.9802 | 30.50 | 139 | 1561.2 |
| **CMANet_Full** | 2026 | 92.25 | 93.30 | 91.05 | 92.16 | 0.9778 | 32.45 | 179 | 1678.8 |
| **CMANet_NoByte** | 42 | 91.93 | 95.03 | 88.50 | 91.65 | 0.9752 | 34.75 | 165 | 1446.5 |
| **CMANet_NoByte** | 123 | 92.45 | 95.14 | 89.45 | 92.21 | 0.9758 | 34.05 | 150 | 1732.4 |
| **CMANet_NoByte** | 2026 | 92.27 | 93.63 | 90.70 | 92.14 | 0.9739 | 42.75 | 155 | 1879.8 |
| **CMANet_NoStruct** | 42 | 89.50 | 89.95 | 88.95 | 89.45 | 0.9670 | 36.55 | 210 | 1444.4 |
| **CMANet_NoStruct** | 123 | 90.25 | 93.99 | 86.05 | 89.84 | 0.9678 | 40.80 | 279 | 1789.8 |
| **CMANet_NoStruct** | 2026 | 89.70 | 90.95 | 88.10 | 89.50 | 0.9656 | 44.80 | 238 | 2115.2 |
| **CMANet_NoText** | 42 | 90.97 | 92.05 | 89.70 | 90.86 | 0.9717 | 35.90 | 203 | 15.5 |
| **CMANet_NoText** | 123 | 91.07 | 94.61 | 87.10 | 90.70 | 0.9712 | 35.60 | 148 | 18.5 |
| **CMANet_NoText** | 2026 | 91.60 | 90.91 | 92.45 | 91.67 | 0.9740 | 34.50 | 145 | 22.5 |
| **CMANet_NoCrossAttn** | 42 | 92.25 | 93.85 | 90.45 | 92.12 | 0.9776 | 33.00 | 144 | 1155.9 |
| **CMANet_NoCrossAttn** | 123 | 92.30 | 94.13 | 90.20 | 92.12 | 0.9778 | 33.30 | 154 | 1437.6 |
| **CMANet_NoCrossAttn** | 2026 | 92.70 | 94.67 | 90.55 | 92.56 | 0.9801 | 35.10 | 189 | 1540.2 |
| **MultimodalFusionNet** | 42 | 92.80 | 93.44 | 92.05 | 92.74 | 0.9800 | 27.10 | 158 | 158.1 |
| **MultimodalFusionNet** | 123 | 93.50 | 94.19 | 92.75 | 93.46 | 0.9813 | 29.90 | 145 | 183.5 |
| **MultimodalFusionNet** | 2026 | 92.57 | 93.20 | 91.95 | 92.57 | 0.9790 | 33.00 | 160 | 207.5 |

### 1.2 Multi-Seed Aggregate Summary (Mean $\pm$ Sample Std, $ddof=1$)

| Model Name | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC | FPR@99%TPR (%) | Malware FN | Mean Train Time |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CMANet_Full** | **$92.38 \pm 0.30$** | $94.31 \pm 0.89$ | $90.22 \pm 0.85$ | **$92.21 \pm 0.30$** | **$0.9789 \pm 0.0012$** | **$30.92 \pm 1.37$** | **$154.7 \pm 21.4$** | 1,603.2 s (26.7 min) |
| **CMANet_NoByte** | $92.22 \pm 0.26$ | $94.60 \pm 0.84$ | $89.55 \pm 1.10$ | $92.00 \pm 0.31$ | $0.9750 \pm 0.0010$ | $37.18 \pm 4.83$ | $156.7 \pm 7.6$ | 1,686.2 s (28.1 min) |
| **CMANet_NoStruct** | $89.82 \pm 0.39$ | $91.63 \pm 2.10$ | $87.70 \pm 1.49$ | $89.60 \pm 0.21$ | $0.9668 \pm 0.0011$ | $40.72 \pm 4.13$ | $242.3 \pm 34.7$ | 1,783.1 s (29.7 min) |
| **CMANet_NoText** | $91.21 \pm 0.34$ | $92.52 \pm 1.89$ | $89.75 \pm 2.68$ | $91.08 \pm 0.52$ | $0.9723 \pm 0.0015$ | $35.33 \pm 0.74$ | $165.3 \pm 32.7$ | 18.8 s (0.3 min) |
| **CMANet_NoCrossAttn** | $92.42 \pm 0.25$ | $94.22 \pm 0.42$ | $90.40 \pm 0.18$ | $92.27 \pm 0.25$ | $0.9785 \pm 0.0014$ | $33.80 \pm 1.14$ | $162.3 \pm 23.6$ | 1,377.9 s (23.0 min) |
| **MultimodalFusionNet** | **$92.96 \pm 0.48$** | $93.61 \pm 0.52$ | $92.25 \pm 0.44$ | **$92.92 \pm 0.47$** | **$0.9801 \pm 0.0012$** | **$30.00 \pm 2.95$** | **$154.3 \pm 8.1$** | 183.1 s (3.1 min) |

### 1.3 Paired Bootstrap Statistical Significance (10,000 Resamples)

To eliminate sample-ordering and selection variance, paired bootstrap resampling was executed across all 4,000 validation samples for each seed independently:

#### A. CMANet-Full vs. MultimodalFusionNet
$$\Delta = \text{Metric}(\text{CMANet-Full}) - \text{Metric}(\text{MultimodalFusionNet})$$

| Seed | Metric | Point Difference ($\Delta_{\text{obs}}$) | Bootstrap Mean Diff | 95% Confidence Interval | Two-Tailed $p$-value | Statistically Significant ($\alpha=0.05$)? |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **42** | Accuracy | $-0.0063$ ($-0.63\%$) | $-0.0063$ | $[-0.0140, +0.0012]$ | $p = 0.1144$ | **No** (CI includes zero) |
| | F1-Score | $-0.0053$ ($-0.53\%$) | $-0.0053$ | $[-0.0130, +0.0023]$ | $p = 0.1718$ | **No** (CI includes zero) |
| | ROC-AUC | $-0.0014$ | $-0.0014$ | $[-0.0040, +0.0012]$ | $p = 0.2730$ | **No** (CI includes zero) |
| | Malware FN | $-12.0$ | $-12.0$ | $[-34.0, +10.0]$ | $p = 0.3124$ | **No** (CI includes zero) |
| **123** | Accuracy | $-0.0078$ ($-0.78\%$) | $-0.0077$ | $[-0.0150, -0.0005]$ | $p = 0.0368$ | Yes (FusionNet slightly better) |
| | F1-Score | $-0.0070$ ($-0.70\%$) | $-0.0070$ | $[-0.0143, +0.0001]$ | $p = 0.0562$ | **No** (CI includes zero) |
| | ROC-AUC | $-0.0011$ | $-0.0011$ | $[-0.0036, +0.0015]$ | $p = 0.4074$ | **No** (CI includes zero) |
| | Malware FN | $-6.0$ | $-6.0$ | $[-26.0, +14.0]$ | $p = 0.5838$ | **No** (CI includes zero) |
| **2026**| Accuracy | $-0.0032$ ($-0.32\%$) | $-0.0033$ | $[-0.0108, +0.0043]$ | $p = 0.4284$ | **No** (CI includes zero) |
| | F1-Score | $-0.0038$ ($-0.38\%$) | $-0.0038$ | $[-0.0115, +0.0040]$ | $p = 0.3458$ | **No** (CI includes zero) |
| | ROC-AUC | $-0.0012$ | $-0.0012$ | $[-0.0038, +0.0013]$ | $p = 0.3686$ | **No** (CI includes zero) |
| | Malware FN | $+19.0$ | $+19.0$ | $[-3.0, +42.0]$ | $p = 0.1048$ | **No** (CI includes zero) |

#### B. CMANet-Full vs. CMANet-NoCrossAttn
$$\Delta = \text{Metric}(\text{CMANet-Full}) - \text{Metric}(\text{CMANet-NoCrossAttn})$$

| Seed | Metric | Point Difference ($\Delta_{\text{obs}}$) | Bootstrap Mean Diff | 95% Confidence Interval | Two-Tailed $p$-value | Statistically Significant ($\alpha=0.05$)? |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **42** | Accuracy | $-0.0008$ ($-0.08\%$) | $-0.0008$ | $[-0.0077, +0.0063]$ | $p = 0.8696$ | **No** (Null hypothesis retained) |
| | F1-Score | $-0.0008$ ($-0.08\%$) | $-0.0008$ | $[-0.0078, +0.0062]$ | $p = 0.8448$ | **No** (Null hypothesis retained) |
| | ROC-AUC | $+0.0010$ | $+0.0010$ | $[-0.0013, +0.0033]$ | $p = 0.4138$ | **No** (Null hypothesis retained) |
| | Malware FN | $+2.0$ | $+2.0$ | $[-18.0, +23.0]$ | $p = 0.8956$ | **No** (Null hypothesis retained) |
| **123** | Accuracy | $+0.0042$ ($+0.42\%$) | $+0.0043$ | $[-0.0033, +0.0120]$ | $p = 0.2990$ | **No** (Null hypothesis retained) |
| | F1-Score | $+0.0045$ ($+0.45\%$) | $+0.0045$ | $[-0.0031, +0.0122]$ | $p = 0.2586$ | **No** (Null hypothesis retained) |
| | ROC-AUC | $+0.0024$ | $+0.0024$ | $[-0.0004, +0.0052]$ | $p = 0.0890$ | **No** (Null hypothesis retained) |
| | Malware FN | $-15.0$ | $-15.0$ | $[-37.0, +6.0]$ | $p = 0.1802$ | **No** (Null hypothesis retained) |
| **2026**| Accuracy | $-0.0045$ ($-0.45\%$) | $-0.0045$ | $[-0.0117, +0.0027]$ | $p = 0.2404$ | **No** (Null hypothesis retained) |
| | F1-Score | $-0.0039$ ($-0.39\%$) | $-0.0039$ | $[-0.0115, +0.0037]$ | $p = 0.3024$ | **No** (Null hypothesis retained) |
| | ROC-AUC | $-0.0023$ | $-0.0023$ | $[-0.0048, +0.0001]$ | $p = 0.0612$ | **No** (Null hypothesis retained) |
| | Malware FN | $-10.0$ | $-10.0$ | $[-31.0, +12.0]$ | $p = 0.3892$ | **No** (Null hypothesis retained) |

#### C. CMANet-Full vs. Unimodal StructMLP
$$\Delta = \text{Metric}(\text{CMANet-Full}) - \text{Metric}(\text{StructMLP})$$

| Seed | Metric | Point Difference ($\Delta_{\text{obs}}$) | Bootstrap Mean Diff | 95% Confidence Interval | Two-Tailed $p$-value | Statistically Significant ($\alpha=0.05$)? |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **42** | Accuracy | $+0.0202$ ($+2.02\%$) | $+0.0202$ | $[+0.0117, +0.0290]$ | $\mathbf{p < 0.0001}$ | **Yes (Multimodal Superior)** |
| | F1-Score | $+0.0207$ ($+2.07\%$) | $+0.0207$ | $[+0.0122, +0.0296]$ | $\mathbf{p < 0.0001}$ | **Yes (Multimodal Superior)** |
| | ROC-AUC | $+0.0120$ | $+0.0120$ | $[+0.0088, +0.0153]$ | $\mathbf{p < 0.0001}$ | **Yes (Multimodal Superior)** |
| **123** | Accuracy | $+0.0338$ ($+3.38\%$) | $+0.0337$ | $[+0.0247, +0.0432]$ | $\mathbf{p < 0.0001}$ | **Yes (Multimodal Superior)** |
| | F1-Score | $+0.0317$ ($+3.17\%$) | $+0.0317$ | $[+0.0228, +0.0409]$ | $\mathbf{p < 0.0001}$ | **Yes (Multimodal Superior)** |
| | ROC-AUC | $+0.0162$ | $+0.0162$ | $[+0.0125, +0.0201]$ | $\mathbf{p < 0.0001}$ | **Yes (Multimodal Superior)** |
| **2026**| Accuracy | $+0.0177$ ($+1.77\%$) | $+0.0178$ | $[+0.0088, +0.0268]$ | $\mathbf{p = 0.0002}$ | **Yes (Multimodal Superior)** |
| | F1-Score | $+0.0148$ ($+1.48\%$) | $+0.0148$ | $[+0.0057, +0.0237]$ | $\mathbf{p = 0.0016}$ | **Yes (Multimodal Superior)** |
| | ROC-AUC | $+0.0093$ | $+0.0093$ | $[+0.0058, +0.0129]$ | $\mathbf{p < 0.0001}$ | **Yes (Multimodal Superior)** |

---

## Phase 2 — Attention Mechanism Empirical Audit

To determine why cross-attention failed to add value, we conducted a direct mathematical audit on the trained CMANet checkpoints and intermediate activation matrices across the validation set.

### 2.1 Attention Weight Entropy (The Diffusion Problem)
In cross-attention direction 1 (Global-to-Sequence, $G \leftarrow T$), the Byte and Struct summary tokens query the API sequence tokens.
- We measured the **Normalized Shannon Attention Entropy**:
  $$\bar{H} = \frac{-\sum_{i=1}^{L_{\text{valid}}} \alpha_i \ln \alpha_i}{\ln(L_{\text{valid}})}$$
  *(Where $\bar{H} = 1.0$ indicates an entirely uniform distribution over all valid tokens, and $\bar{H} \to 0.0$ indicates sharp focus on a single token).*
- **Empirical Measurement:**  
  $$\text{Mean Normalized Entropy } \bar{H}_{G \leftarrow T} = \mathbf{0.9307} \quad (\text{Std: } 0.0412)$$
- **Interpretation:** Attention weights are **almost completely diffuse**. Rather than learning to identify sparse, critical API invocations (e.g., `VirtualAllocEx`, `WriteProcessMemory`, `RegSetValueEx`), the global query tokens absorb a near-uniform soft average of the entire sequence. Soft average pooling with attention weights of $\sim 1/L$ performs mathematically almost identically to uniform average pooling!

### 2.2 Sequence-to-Global Query Preference
In cross-attention direction 2 (Sequence-to-Global, $T \leftarrow G$), API tokens query the two global summary tokens (Byte vs. Struct).
- **Empirical Attention Distribution:**
  - Attention Weight on Byte Token: $\mathbf{46.84\%}$
  - Attention Weight on Struct Token: $\mathbf{53.16\%}$
- **Interpretation:** The API sequence tokens assign an almost static 50/50 split between Byte and Struct summary representations. There is no dynamic, context-specific routing (e.g., heavily weighting Struct for packed files or heavily weighting Byte for entropy anomalies).

### 2.3 Positional Embedding Utility
- **Learned Positional Embedding Norm:** Mean $\mathbf{0.4271}$ (Min $0.2194$, Max $1.1078$).
- **Comparison to Token Embedding Norms:** Token embeddings have an average $L2$ norm of $\mathbf{\sim 3.48}$. The learned positional embeddings are an order of magnitude smaller ($\sim 12\%$ of token magnitude).
- **Adjacent Position Cosine Similarity:** Mean $\mathbf{0.1124}$ (Min $-0.68$, Max $+0.75$).
- **Interpretation:** In EMBER's hashed API sequences, absolute position within the extracted sequence carries very little consistent semantic information. The positional embeddings remained small and noisy, functioning primarily as regularization noise rather than structural sequence encoding.

### 2.4 Residual Branch Dynamics
In CMANet's residual connections:
$$\mathbf{z}_{\text{out}} = \text{LayerNorm}(\mathbf{z}_{\text{base}} + \text{Dropout}(\mathbf{A}))$$
- **Byte Token:** Baseline Norm $\|\mathbf{z}_{\text{base}}\| = 6.4496$, Update Magnitude $\|\Delta\| = 7.1097 \implies \text{Update Ratio } = \mathbf{1.1023}$.
- **Struct Token:** Baseline Norm $\|\mathbf{z}_{\text{base}}\| = 6.9751$, Update Magnitude $\|\Delta\| = 7.5908 \implies \text{Update Ratio } = \mathbf{1.0883}$.
- **Interpretation:** The attention updates are neither vanishingly small nor exploding. They provide substantial perturbations ($\sim 1.1\times$ baseline norm). However, because the attention weights are diffuse ($\bar{H} = 0.93$), these perturbations essentially inject smoothed sequence averages back into the global vectors without introducing selective non-linear interactions.

---

## Phase 3 — Training Dynamics & Computational Efficiency Audit

### 3.1 Training Trajectory & Generalization Gap

| Model | Avg Epoch 10 Train Loss | Avg Epoch 10 Val Loss | Generalization Gap ($\text{Val} - \text{Train}$) | Avg Best Val Loss | Avg Best Epoch |
|---|:---:|:---:|:---:|:---:|:---:|
| **CMANet_Full** | $\mathbf{0.0972}$ | $\mathbf{0.2177}$ | $\mathbf{0.1206}$ | $0.1896$ | $6.3$ |
| **CMANet_NoCrossAttn** | $0.1230$ | $0.2076$ | $0.0846$ | $0.1926$ | $5.3$ |
| **MultimodalFusionNet** | $0.1195$ | $0.2026$ | $\mathbf{0.0831}$ | $\mathbf{0.1821}$ | $7.0$ |

#### The Overfitting Diagnostic:
- Notice that `CMANet_Full` achieved the **lowest training loss** of all models ($0.0972$), but had the **highest validation loss at epoch 10** ($0.2177$).
- The generalization gap for `CMANet_Full` is **$0.1206$**, which is **$45.1\%$ larger** than `MultimodalFusionNet` ($0.0831$) and `CMANet_NoCrossAttn` ($0.0846$).
- `MultimodalFusionNet` maintained a lower validation loss ($0.1821$ vs $0.1896$) and peaked later (Epoch 7.0 vs 6.3), indicating smoother, more regularized optimization.

### 3.2 Computational Cost & Efficiency Comparison ($L=256$)

| Model Name | Total Parameters | Non-Embedding Parameters | Theoretical FLOPs | Theoretical MACs | Total Training Time (3 Seeds) | Inference Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **CMANet_Full** | 1,644,610 | 364,610 | 101.71 M | 50.85 M | 4,809.7 s (~80.2 min) | 1.83 ms/sample |
| **CMANet_NoCrossAttn** | 1,511,746 | 231,746 | 67.37 M | 33.68 M | 4,133.7 s (~68.9 min) | 1.57 ms/sample |
| **MultimodalFusionNet** | 1,498,626 | 218,626 | **50.57 M** | **25.28 M** | **549.1 s (~9.2 min)** | **0.40 ms/sample** |

- **Training Speedup:** `MultimodalFusionNet` trained in **9.2 minutes total** across all 3 seeds, compared to **80.2 minutes** for `CMANet_Full` (**$8.7\times$ faster**).
- **FLOPs:** `MultimodalFusionNet` requires exactly **half the FLOPs** ($50.57\text{ M}$ vs $101.71\text{ M}$).
- **Inference Latency:** `MultimodalFusionNet` processes samples at **$0.40\text{ ms}$**, which is **$4.6\times$ faster** than CMANet ($1.83\text{ ms}$).

---

## Phase 4 — Modality Contribution & Architectural Dissection

By ranking all 5 CMANet variants, we can isolate the empirical contribution of each individual component:

```
[92.96%] MultimodalFusionNet (Baseline: 1D Conv + Masked Pooling + Concat)
   │  -0.58% accuracy delta, +8.7x training cost
[92.42%] CMANet-NoCrossAttn (Self-Attention + Concat)
   │  -0.04% accuracy delta (Indistinguishable from zero, p = 0.87)
[92.38%] CMANet-Full (Bidirectional Cross-Attention)
   │  -0.16% accuracy delta
[92.22%] CMANet-NoByte (Struct + Text only)
   │  -1.01% accuracy delta
[91.21%] CMANet-NoText (Byte + Struct only; Dense MLP)
   │  -1.39% accuracy delta
[89.82%] CMANet-NoStruct (Byte + Text only)  <-- CATASTROPHIC DROP (-2.56%, +87.6 FN)
```

### Hierarchy of Modality Importance:
1. **PE Structural Features (Modality C) — Absolute Foundation:**
   - Removing PE structure causes the single largest performance collapse in the entire benchmark: accuracy falls from $92.38\% \to 89.82\%$, and false negatives jump from $154.7 \to 242.3$ (+87.6 missed malware samples).
2. **Sequential API Text (Modality B) — Meaningful Incremental Signal:**
   - Removing API sequences drops accuracy from $92.38\% \to 91.21\%$ ($-1.17\%$). API tokens provide essential behavioral evidence that pure tabular byte/PE features cannot fully capture.
3. **Byte Histograms & Entropy (Modality A) — Minor Regularizer:**
   - Removing byte histograms produces only a mild drop from $92.38\% \to 92.22\%$ ($-0.16\%$).
4. **Cross-Modal Attention Mechanism — Redundant Overhead:**
   - Removing cross-modal attention entirely (`CMANet-NoCrossAttn`) changes accuracy from $92.38\% \to 92.42\%$ ($+0.04\%$, $p=0.87$). It contributes no measurable predictive gain over simple concatenation.

---

## Why Did Cross-Modal Attention Fail? Four Grounded Hypotheses

Based on the empirical evidence gathered during this audit, we identify four root causes:

### 1. Inductive Bias Deficit vs. 1D Convolutions (The Locality Advantage)
- **MultimodalFusionNet's Advantage:** In malware analysis, malicious API patterns occur in tight local n-grams (e.g., `VirtualAlloc` $\to$ `WriteProcessMemory` $\to$ `CreateRemoteThread`). `MultimodalFusionNet` uses 1D convolutions with kernel size 3, which has a **strong, hard inductive bias for local contiguous trigrams**.
- **CMANet's Disadvantage:** Multi-Head Attention has **no inductive bias for locality**. It must learn from scratch which token pairs across all 256 positions are relevant. With only 16,000 training samples, the attention mechanism cannot learn these local combinatorial patterns as effectively as a simple 1D convolution.

### 2. Modality Granularity Asymmetry (The Summary Token Bottleneck)
- Cross-attention was designed for sequences where both modalities have token-level granularity (e.g., image patches attending to word tokens in CLIP or ViLT).
- In EMBER, Byte features (512-D) and PE Struct features (66-D) are **whole-file global summary vectors**, not sequences.
- Projecting each into a single 128-D token creates a severe information bottleneck: every API token in the sequence queries the exact same static summary vector. As proven by our attention entropy audit ($\bar{H} = 0.9307$), this collapses into a diffuse, uniform average that cannot capture token-specific structural associations.

### 3. Overparameterization & Optimization Lag on Small Sample Budgets
- On a 16,000-sample training set, adding three Multi-Head Attention blocks (146,000 additional projection parameters and 6 LayerNorms) without local inductive bias caused CMANet to overfit (training loss $0.0972$ vs validation loss $0.2177$, a $45\%$ larger generalization gap).
- Concurrently, Adam with standard settings without an extensive warmup schedule or learning rate decay caused the attention projections to settle into broad, non-selective local minima.

### 4. Dense Structural Prepotency
- The 66 PE structural features are extraordinarily information-dense: a basic MLP on PE structure alone (`StructMLP`) already achieves $\approx 90\%$ accuracy on EMBER.
- Because PE structure resolves the vast majority of easy benign/malware boundaries, the remaining difficult boundary cases require detecting subtle, specific API trigrams or packing anomalies. Diffuse global-to-sequence attention fails to resolve these fine-grained edge cases.

---

## Methodological Limitations

1. **Vocabulary Hashing:** API names are hashed into 10,000 buckets using MurmurHash3. Hashing collisions and lack of subword tokenization (e.g., BPE) prevent the attention mechanism from attending to semantic sub-tokens (e.g., `Create`, `Process`, `Ex`).
2. **Fixed Sequence Window ($L=256$):** PE files with thousands of imports are truncated to the first 256 functions, while shorter files are padded.
3. **Sample Scale:** Training was evaluated on 16,000 samples. While sufficient to establish statistical separation between modalities ($p < 0.0001$ for multimodal vs unimodal), it is insufficient for data-hungry quadratic cross-attention to surpass convolutional inductive biases.

---

## Final Research Recommendations & Next Steps

1. **Do NOT Adopt CMANet-Full for Production:**
   - CMANet-Full adds $2\times$ FLOPs, $8.7\times$ training latency, and increased memory usage with zero statistically significant gain in accuracy, F1, or ROC-AUC over `MultimodalFusionNet`.
2. **Retain MultimodalFusionNet as the Primary Neural Architecture:**
   - `MultimodalFusionNet` remains the superior neural architecture: $92.96\%$ Accuracy, $0.9801$ ROC-AUC, $154.3$ FN, and ultra-fast inference ($0.40\text{ ms}$).
3. **Recommended Next Research Direction (Hierarchical / N-Gram Hybridization):**
   - If attention is to be explored further, it should **replace the global summary bottleneck**:
     - Apply shallow 1D convolutions *first* to extract local API n-grams, followed by self-attention *only* over valid n-grams.
     - Extract section-level sequence tokens (e.g., tokenizing each PE section header) rather than compressing the entire structural space into one 128-D vector.
4. **Official Test Set Evaluation Protocol:**
   - In accordance with research integrity rules, `test_features.jsonl` remains strictly untouched until a model demonstrating confirmed validation superiority is established.

