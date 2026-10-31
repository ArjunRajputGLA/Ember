# Final Controlled Experiment Protocol: EMBER Multimodal Malware Detection

**Project:** Ember – Malware Detection Using Native Multimodal Learning  
**Working Branch:** `experiment/native-multimodal-transformer`  
**Target Paper:** B.Tech Final Year Research Thesis / Conference Submission  
**Auditor / ML Systems Engineer:** Senior ML Research Engineer  
**Date:** October 31, 2026  
**Status:** Protocol Frozen for Final Test Evaluation  

---

## 1. Primary Neural Architecture Specification: `MultimodalFusionNet`

Following the post-experiment research audit and 18-run cross-modal benchmark, `MultimodalFusionNet` is officially **frozen as the primary neural architecture**.

### 1.1 Architectural Blueprint & Tensor Flow
`MultimodalFusionNet` processes three heterogeneous PE modalities through dedicated unimodal encoders before late concatenation and joint non-linear classification:

```
[Modality A: Byte Statistics (512-D)]
   │
   ├─► Linear(512 -> 128) -> BatchNorm1d(128) -> ReLU() ──► h_byte (128-D)
   │
[Modality B: API Token Sequence (L=256)]
   │
   ├─► Embedding(10000 -> 128, padding_idx=0)
   ├─► Conv1d(128 -> 128, kernel=3, padding=1) -> ReLU() * AttentionMask
   ├─► Conv1d(128 -> 128, kernel=3, padding=1) -> ReLU() * AttentionMask
   ├─► Mask-Aware Mean Pooling (sum over valid tokens / valid_counts) ──► h_text (128-D)
   │
[Modality C: PE Structural Header Metrics (66-D)]
   │
   ├─► Linear(66 -> 64) -> BatchNorm1d(64) -> ReLU() ──► h_struct (64-D)
   │
[Late Concatenation Fusion]
   │
   ├─► Fused Representation: [h_byte (128), h_text (128), h_struct (64)] = 320-D
   │
[Joint Non-Linear Classifier]
   │
   ├─► Linear(320 -> 128) -> BatchNorm1d(128) -> ReLU() -> Dropout(p=0.3)
   ├─► Linear(128 -> 64) -> ReLU()
   └─► Linear(64 -> 2) ──► Logits (2-D) -> Softmax
```

### 1.2 Capacity & Efficiency Profile
- **Total Trainable Parameters:** 1,498,626
- **Shared API Embedding Parameters:** 1,280,000 (10,000 vocab $\times$ 128-D)
- **Non-Embedding Trainable Parameters:** 218,626
- **Theoretical MACs:** 25,284,864 per sample ($L=256$)
- **Theoretical FLOPs:** 50.57 MFLOPs per sample
- **CPU Inference Latency:** $0.40\text{ ms/sample}$ (~2,500 samples/sec on 16 threads)
- **GPU Inference Latency:** $< 0.05\text{ ms/sample}$ (on T4 GPU)

---

## 2. Dataset Definition & Preprocessing Standards

### 2.1 Fixed Splits & Cryptographic Manifest
- **Training Subset:** 16,000 samples (8,000 Benign, 8,000 Malware).
  - Array File: `experiments/baseline_correction_2026_10_30/data/ember_train_16k_corrected.npz`
  - SHA-256 Checksum: `1164ca77fc4d05144726e21cec1152a7d1d5280d0e11b137501681bdf64a41c6`
- **Validation Subset:** 4,000 samples (2,000 Benign, 2,000 Malware).
  - Array File: `experiments/baseline_correction_2026_10_30/data/ember_val_4k_corrected.npz`
  - SHA-256 Checksum: `6b25a36c96db4719c8dabaf7b97a7433207a30696b5271dabfed5bea0bf6c188`
- **Disjointness Assertion:** 0 SHA-256 overlap between training and validation sets.

### 2.2 Leakage-Free Feature Processing Rules
1. **Modality A (512-D Byte Histogram & Entropy):**
   - 256-D byte value histogram normalized by its sum.
   - 256-D byte entropy histogram normalized by its sum.
   - Normalized strictly on a per-sample basis ($\epsilon = 10^{-7}$).
2. **Modality B (API Text Sequence, $L=256$):**
   - Extracted from `imports`, `exports`, and `section` names.
   - Vocabulary: 10,000 tokens, fitted **strictly and exclusively** on the 16,000 training partition.
   - Unseen tokens in validation/test mapped to `<UNK>` (index 1). Padding mapped to `<PAD>` (index 0).
3. **Modality C (66-D Structural PE Metrics):**
   - 10 General PE metrics, 13 COFF/Optional Header metrics, 7 Section statistics, 30 Data Directory metrics (15 pairs of size and presence), 8 Strings statistics.
   - Preprocessing: `StandardScaler` fitted **strictly and exclusively** on the 16,000 training partition.
4. **Tabular Features (2,381-D):**
   - Extracted via `PEFeatureExtractor(feature_version=2)`.
   - Patched with Option B `SectionInfo` (`[list(entry_val)]`) to preserve character-level MurmurHash3 parity with official EMBER 2018 (scikit-learn 0.20).

---

## 3. Training & Validation Protocol

To maintain complete experimental comparability with all audited baselines and CMANet variants:
- **Optimizer:** Adam
- **Learning Rate:** $0.001$ ($1\times 10^{-3}$)
- **Weight Decay:** $0.0001$ ($1\times 10^{-4}$)
- **Batch Size:** 32
- **Epochs:** 10
- **Loss Function:** `nn.CrossEntropyLoss()`
- **Learning Rate Scheduler:** None (constant learning rate to isolate architecture from scheduling artifacts)
- **Checkpoint Selection:** Best validation cross-entropy loss checkpoint saved and restored for final evaluation
- **Hardware Agnosticism:** CPU- and CUDA-compatible tensor operations

---

## 4. Multi-Seed Replication & Statistical Standard

Every reported metric must follow strict multi-seed replication:
- **Seeds:** `[42, 123, 2026]`
- **Reported Interval:** Mean $\pm$ Sample Standard Deviation ($s$, $ddof=1$):
  $$s = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2} \quad (N=3)$$
- **Required Primary Evaluation Metrics:**
  1. Classification Accuracy (%)
  2. Precision (%)
  3. Recall (%)
  4. F1-Score (%)
  5. Area Under ROC Curve (ROC-AUC)
  6. False Positive Rate at 99% True Positive Rate (FPR@99%TPR)
  7. Absolute Malware False Negatives (FN count out of positive test population)
- **Pairwise Significance Testing:**
  - 10,000 paired bootstrap resamples on identical sample indices.
  - Report 95% Confidence Interval ($[2.5\%, 97.5\%]$).
  - Report empirical two-tailed p-value.
  - Apply Bonferroni correction across multiple hypothesis comparisons ($\alpha_{\text{adjusted}} = 0.05 / K$).

---

## 5. Official Test Set Evaluation Protocol (Final Execution Phase)

### 5.1 Quarantined Test Dataset Specification
- **Path:** `ember_dataset_2018_2/ember2018/test_features.jsonl`
- **Total Population:** 200,000 JSONL records (100,000 Benign, 100,000 Malware, 0 Unlabeled).
- **File Size:** ~1.87 GB.
- **Quarantine Policy:** Until the final test run is explicitly launched, this file must remain strictly unread by any training script, hyperparameter optimizer, or feature scaler.

### 5.2 Single-Pass Test Execution Standard
To guarantee academic integrity and eliminate test-set overfitting:
1. **Zero Retraining or Tuning:** Test set evaluation is strictly read-only and **single-pass**. No hyperparameter tuning, model adjustment, or feature re-selection is permitted after test set evaluation.
2. **Deterministic Feature Ingestion:**
   - Features from `test_features.jsonl` are pre-extracted using the frozen training vocabulary (`vocab.json`) and the frozen training scaler (`StandardScaler`).
3. **Multi-Seed Ensembling / Reporting:**
   - All three saved checkpoints (`best_MultimodalFusionNet_seed42.pt`, `best_MultimodalFusionNet_seed123.pt`, `best_MultimodalFusionNet_seed2026.pt`) will be evaluated independently on the 200,000 test records.
   - Report Mean $\pm$ Sample Standard Deviation across the three runs.
4. **Mandatory Comparative Benchmark Set:**
   To provide a complete thesis defense, the exact same 200,000 test samples must be evaluated across:
   - `MultimodalFusionNet` (Primary Candidate)
   - `Fresh_LightGBM` (Tabular Upper Bound Reference)
   - `StructMLP` (Top Unimodal Baseline Reference)
   - `MaskedTextCNN` (API Sequence Baseline Reference)
   - `ByteMLP` (Raw Byte Baseline Reference)
   - `CMANet_Full` (Negative Finding Reference)

