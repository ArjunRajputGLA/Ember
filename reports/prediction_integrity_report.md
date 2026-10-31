# Prediction Integrity & Forensic Audit Report

**Audit Target:** Official EMBER 2018 v2 Test Evaluation Prediction Outputs  
**Audit Date:** 2026-10-31  
**Lead Investigator:** Senior ML Research Engineer & Methodology Auditor  
**Dataset Quarantined Hash Manifest:** Verified Bit-for-Bit Isolation  

---

## 1. Executive Forensic Summary

This audit assesses the mathematical validity, cryptographic integrity, and data flow of the raw predictions generated during the official test evaluation of 200,000 EMBER binaries.

### Key Audit Findings:
1. **LightGBM Seed Determinism Fully Explained:** The three LightGBM seeds (42, 123, 2026) produce bit-for-bit identical test predictions. Investigation confirms that LightGBM's histogram binning and tree building algorithms are 100% deterministic when `bagging_fraction=1.0` and `colsample_bytree=1.0`. The checkpoint files are physically distinct with unique cryptographic hashes, but their tree structures and thresholds are identical. No aliasing or caching bugs occurred.
2. **MultimodalFusionNet Ensemble Verified:** The probability vector `p_multimodalfusionnet_ensemble` mathematically equals $\frac{p_{42} + p_{123} + p_{2026}}{3}$ with a maximum absolute residual of **0.0000000000e+00**. The decision threshold is strictly $\ge 0.5$.
3. **Array Completeness & Range:** All 19 prediction vectors in `test_predictions_all_models.npz` contain exactly 200,000 finite values strictly bound in $[0.0, 1.0]$ with zero NaNs, Infs, or missing values.

---

## 2. Cryptographic Asset Verification

| Artifact Description | File Path | File Size | SHA-256 Hash |
|:---|:---|:---:|:---|
| **Official Test NPZ** | `data/ember_test_200k.npz` | 746.8 MB | `50b72f67bd782f83665ef0c3e7656860f7e45d8dc7ac650ec7e1da0b2e09f5e1` |
| **All Test Predictions** | `prediction_outputs/test_predictions_all_models.npz` | 24.0 MB | `3866fa446ed9b59da2f503e81bdc6584e892eb4237d22e3b1c22357ae989cdb5` |
| **LightGBM Checkpoint Seed 42** | `checkpoints/fresh_lightgbm_seed42.txt` | 3,669,803 bytes | `f638b4524595c7ee5e5ed51f13dcdf53607b6c80b6558e8985b9e84aebd6838c` |
| **LightGBM Checkpoint Seed 123** | `checkpoints/fresh_lightgbm_seed123.txt` | 3,669,805 bytes | `c5a2474e4b8653850ad4f4f598fa19d8d1e610d903de707985a16a358e613299` |
| **LightGBM Checkpoint Seed 2026** | `checkpoints/fresh_lightgbm_seed2026.txt` | 3,669,804 bytes | `090843cd93f1da4f7f35d1baad45ffe2a89da20f535118b2fee53ec39eb4e19c` |

---

## 3. LightGBM Identity Investigation (Task 3)

### 3.1 Empirical Observation
During official test evaluation, `Fresh_LightGBM` produced identical metrics across seeds 42, 123, and 2026:
- Accuracy: **93.38%**
- F1-Score: **93.48%**
- ROC-AUC: **0.9847**
- Malware False Negatives: **4,987**
- Benign False Positives: **8,257**

### 3.2 Hypothesis Testing & Verification
We investigated whether this was caused by file caching, improper loop scoping, or mathematical determinism:

1. **Prediction Array Comparison:**
   - $\max |p_{42} - p_{123}| = 0.0$
   - $\max |p_{42} - p_{2026}| = 0.0$
   - Predictions are bit-for-bit identical across all 200,000 test binaries.

2. **Checkpoint File Verification:**
   - The three checkpoint files exist as three distinct files on disk.
   - The file hashes are unique:
     - Seed 42 SHA-256: `f638b4524595c7ee5e5ed51f13dcdf53607b6c80b6558e8985b9e84aebd6838c`
     - Seed 123 SHA-256: `c5a2474e4b8653850ad4f4f598fa19d8d1e610d903de707985a16a358e613299`
     - Seed 2026 SHA-256: `090843cd93f1da4f7f35d1baad45ffe2a89da20f535118b2fee53ec39eb4e19c`

3. **Line-by-Line Model Dissection:**
   - Total lines in each model file: **20,415 lines**.
   - Exactly **7 lines** differ between seed 42 and seed 123 (differing lines: 7).
   - Exactly **7 lines** differ between seed 42 and seed 2026 (differing lines: 7).
   - Dissection of differing lines reveals:
     - Line 20302: `[seed: 42]` vs `[seed: 123]` vs `[seed: 2026]`
     - Line 20314: `[bagging_seed: 400]` vs `[bagging_seed: 19053]` vs `[bagging_seed: 26038]`
     - Line 20318: `[feature_fraction_seed: 30056]` vs `[feature_fraction_seed: 13104]` vs `[feature_fraction_seed: 28694]`
     - Line 20320: `[extra_seed: 12879]` vs `[extra_seed: 3265]` vs `[extra_seed: 4728]`
     - Line 20334: `[drop_seed: 17869]` vs `[drop_seed: 23075]` vs `[drop_seed: 5409]`
   - Every single tree structure, feature split index, numerical threshold, and leaf weight (lines 1 to 20,300) is **100% identical**.

### 3.3 Theoretical Root Cause
LightGBM's default training configuration utilizes:
- `bagging_fraction = 1.0` (no sample subsampling)
- `feature_fraction = 1.0` (no feature subsampling)

Because deterministic histogram construction and greedy split finding are executed over fixed, ordered data without subsampling, the pseudorandom number generator (PRNG) is never invoked during tree construction. Thus, the random seed only alters the metadata header comments. The model parameters are mathematically invariant to seed changes under full-batch non-stochastic gradient boosting.

**Verdict:** The identical test metrics are genuine and scientifically legitimate.

---

## 4. MultimodalFusionNet Ensemble Verification (Task 4)

### 4.1 Ensemble Aggregation Protocol
The ensemble was pre-specified in `reports/final_experiment_protocol.md` as an arithmetic mean soft-voting ensemble across seeds 42, 123, and 2026:
$$p_{\text{ensemble}} = \frac{p_{\text{seed42}} + p_{\text{seed123}} + p_{\text{seed2026}}}{3}$$
$$\hat{y}_{\text{ensemble}} = \mathbb{I}(p_{\text{ensemble}} \ge 0.5)$$

### 4.2 Verification Against Independent Calculation
- Arithmetic residual: $\max |p_{\text{ensemble}} - \frac{p_{42} + p_{123} + p_{2026}}{3}| = \mathbf{0.0000000000e+00}$
- Accuracy of Ensemble: **91.34%** (matches `seed_wise_metrics.csv`)
- F1-Score of Ensemble: **91.45%** (matches `seed_wise_metrics.csv`)
- ROC-AUC of Ensemble: **0.9716** (matches `seed_wise_metrics.csv`)
- False Negatives: **7,413** (matches `seed_wise_metrics.csv`)

**Verdict:** Ensemble calculations adhere strictly to the frozen protocol.
