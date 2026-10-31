# Forensic Log of Unresolved Methodological Issues & Audit Classifications

**Project:** Ember – Malware Detection Using Native Multimodal Learning  
**Working Branch:** `experiment/native-multimodal-transformer`  
**Dataset:** EMBER 2018 v2  
**Auditor:** Senior ML Research Engineer & Research-Methodology Auditor  
**Date:** October 31, 2026  
**Status:** Audit Complete  

---

## 1. Audit Taxonomy & Classification Standards

Every audited item in this investigation is categorized under one of four strict methodological statuses:
- **`VERIFIED`**: Independently reproduced from raw source arrays, checkpoints, or deterministic scripts with zero discrepancies.
- **`NEEDS VERIFICATION`**: Plausible claim supported by partial code or documentation, but lacking full end-to-end cryptographic or execution traces.
- **`METHODOLOGICAL ISSUE`**: An identified flaw, transcription mismatch, procedural divergence, data-leakage risk, or statistical vulnerability that must be documented or resolved.
- **`NOT APPLICABLE`**: Checks or assumptions rendered irrelevant by current project architecture or constraints.

---

## 2. Comprehensive Inventory of Findings & Classifications

### Issue 1: Manual Metric Transcription Error in Post-Experiment Analysis Report
- **Classification:** `METHODOLOGICAL ISSUE`
- **Severity:** High (Reporting Integrity) / Low (Algorithmic / Code Integrity)
- **Evidence:**
  - File: [`reports/cmanet_post_experiment_analysis.md`](file:///c:/Users/imsto/Desktop/Final%20Year%20Paper/Old%20Repo%20Data/reports/cmanet_post_experiment_analysis.md#L40-L42), Lines 40–42 & Line 63.
  - Source File: [`experiments/cross_modal_attention_2026_10_30/metrics/run_CMANet_Full_seed42.json`](file:///c:/Users/imsto/Desktop/Final%20Year%20Paper/Old%20Repo%20Data/experiments/cross_modal_attention_2026_10_30/metrics/run_CMANet_Full_seed42.json#L2-L5).
  - Source File: [`experiments/cross_modal_attention_2026_10_30/metrics/cross_modal_attention_results.json`](file:///c:/Users/imsto/Desktop/Final%20Year%20Paper/Old%20Repo%20Data/experiments/cross_modal_attention_2026_10_30/metrics/cross_modal_attention_results.json#L221-L229).
  - Raw Predictions: [`experiments/cross_modal_attention_2026_10_30/metrics/predictions_seed42.npz`](file:///c:/Users/imsto/Desktop/Final%20Year%20Paper/Old%20Repo%20Data/experiments/cross_modal_attention_2026_10_30/metrics/predictions_seed42.npz).
- **Description:**
  In [`reports/cmanet_post_experiment_analysis.md`](file:///c:/Users/imsto/Desktop/Final%20Year%20Paper/Old%20Repo%20Data/reports/cmanet_post_experiment_analysis.md), Table 1.1 reports the following metrics for `CMANet_Full`:
  - Seed 42: Precision $94.69\%$, Recall $89.35\%$, F1-Score $91.94\%$.
  - Seed 123: Precision $94.94\%$, Recall $90.25\%$, F1-Score $92.53\%$.
  - Aggregate Mean: F1-Score reported as **$92.21 \pm 0.30\%$** against Accuracy **$92.38 \pm 0.30\%$**.
  However, independent recalculation directly from the ground-truth prediction arrays and the original run JSON logs reveals:
  - Seed 42: Precision $91.74\%$, Recall $92.70\%$, F1-Score **$92.22\%$** (TN=1833, FP=167, FN=146, TP=1854).
  - Seed 123: Precision $92.45\%$, Recall $93.05\%$, F1-Score **$92.75\%$** (TN=1848, FP=152, FN=139, TP=1861).
  - Seed 2026: Precision $93.29\%$, Recall $91.05\%$, F1-Score **$92.16\%$** (TN=1869, FP=131, FN=179, TP=1821).
  - Ground-Truth Aggregate Mean: Precision $92.49 \pm 0.78\%$, Recall $92.27 \pm 1.07\%$, **Mean F1-Score = $92.38 \pm 0.32\%$**.
- **Root Cause:**
  A manual transcription error occurred during the compilation of the markdown report table. The numbers $94.69\%$, $89.35\%$, and $91.94\%$ do not exist in any JSON metrics file or prediction array in the repository. Because the validation set is perfectly balanced (2,000 benign, 2,000 malware) and precision/recall are closely matched, the true F1 score is virtually identical to Accuracy ($92.38\%$).
- **Impact on Conclusions:**
  Zero impact on scientific conclusions. Both the true F1 ($92.38\%$) and reported Accuracy ($92.38\%$) remain inferior to `MultimodalFusionNet` ($92.96\%$ Accuracy, $92.92\%$ F1).
- **Required Resolution:**
  Update the transcription in project reports and thesis text to reflect the verified telemetry ($92.38\%$ F1).

---

### Issue 2: Batch Size Asymmetry & StructMLP Bootstrap Comparison Provenance
- **Classification:** `METHODOLOGICAL ISSUE`
- **Severity:** Medium (Experimental Rigor & Hyperparameter Parity)
- **Evidence:**
  - Script: [`scripts/run_baseline_correction_experiments.py`](file:///c:/Users/imsto/Desktop/Final%20Year%20Paper/Old%20Repo%20Data/scripts/run_baseline_correction_experiments.py#L317), Line 317 (`--batch_size 64`).
  - Script: [`scripts/run_cross_modal_experiments.py`](file:///c:/Users/imsto/Desktop/Final%20Year%20Paper/Old%20Repo%20Data/scripts/run_cross_modal_experiments.py#L596), Line 596 (`--batch-size 32`).
  - Script: [`scripts/run_cross_modal_experiments.py`](file:///c:/Users/imsto/Desktop/Final%20Year%20Paper/Old%20Repo%20Data/scripts/run_cross_modal_experiments.py#L741-L779), Lines 741–779.
  - Manifest: [`experiments/baseline_correction_2026_10_30/configs/experiment_manifest.json`](file:///c:/Users/imsto/Desktop/Final%20Year%20Paper/Old%20Repo%20Data/experiments/baseline_correction_2026_10_30/configs/experiment_manifest.json#L300-L375).
- **Description:**
  1. In the baseline correction phase, all models (including `StructMLP` and `MultimodalFusionNet`) were trained using `batch_size = 64`.
  2. In the cross-modal attention benchmark, all six models (including retrained `MultimodalFusionNet`) were trained using `batch_size = 32`.
  3. When conducting paired bootstrap significance testing in [`scripts/run_cross_modal_experiments.py`](file:///c:/Users/imsto/Desktop/Final%20Year%20Paper/Old%20Repo%20Data/scripts/run_cross_modal_experiments.py), `StructMLP` was not retrained at `batch_size = 32`; instead, predictions were loaded from the prior baseline run (`experiments/baseline_correction_2026_10_30/metrics/predictions_seed{seed}.npz`).
- **Audit Verification:**
  - Sample alignment was strictly verified: `y_true` from the baseline run is identical element-by-element to `y_true` in the cross-modal run. The paired bootstrap is valid at the sample level.
  - However, comparing CMANet-Full (trained at `bs=32`) against StructMLP (trained at `bs=64`) introduces a minor hyperparameter discrepancy.
- **Impact on Conclusions:**
  Negligible. StructMLP is a tiny 2-layer MLP (17,218 parameters). Its performance across seeds at `bs=64` is $89.99 \pm 0.58\%$. The delta between CMANet-Full and StructMLP is $+2.39\%$ ($p < 0.0002$), which is far too large to be an artifact of batch size 32 vs 64. Furthermore, MultimodalFusionNet at `bs=64` achieved $92.73\%$ and at `bs=32` achieved $92.96\%$, consistently outperforming StructMLP by $>2.7\%$.
- **Required Resolution:**
  Document the batch size difference explicitly in the thesis methodology chapter. Retraining StructMLP at `bs=32` is not strictly necessary for final submission, but can be documented as a known minor parameter difference.

---

### Issue 3: Exploratory Test Set Touch in Pre-Flight Inspection Script
- **Classification:** `METHODOLOGICAL ISSUE` (Contained / Quarantined)
- **Severity:** Low (Quarantine Violation in Auxiliary Script, but Complete Isolation in Experimental Pipelines)
- **Evidence:**
  - Script: [`scripts/inspect_ember_dataset.py`](file:///c:/Users/imsto/Desktop/Final%20Year%20Paper/Old%20Repo%20Data/scripts/inspect_ember_dataset.py#L187-L205), Lines 187–205.
  - File: `c:\Users\imsto\Desktop\Final Year Paper\ember_dataset_2018_2\ember2018\test_features.jsonl` (Size: 1.87 GB, 200,000 lines).
- **Description:**
  In early project exploration, [`scripts/inspect_ember_dataset.py`](file:///c:/Users/imsto/Desktop/Final%20Year%20Paper/Old%20Repo%20Data/scripts/inspect_ember_dataset.py) read the first 3,500 lines of `test_features.jsonl` to check JSON schema validity and field types.
- **Audit Verification:**
  - We audited every feature extraction, training, validation, and evaluation pipeline (`prepare_ember_data_corrected.py`, `run_baseline_correction_experiments.py`, `run_cross_modal_experiments.py`, `evaluate_and_audit_all_baselines.py`).
  - None of the training, validation, vocabulary-building, or normalization code ever opened `test_features.jsonl`.
  - The official test set remains 100% un-evaluated and quarantined from model selection.
- **Required Resolution:**
  Note clearly in the audit report that `inspect_ember_dataset.py` was an exploratory read-only schema diagnostic, and that all training, hyperparameter selection, and validation splits were derived exclusively from `train_features_0.jsonl` through `train_features_5.jsonl`.

---

### Issue 4: Multiple-Testing Correction in Bootstrap Significance Reporting
- **Classification:** `METHODOLOGICAL ISSUE` (Statistical Nuance)
- **Severity:** Low (Academic Presentation)
- **Evidence:**
  - Script: [`scripts/run_cross_modal_experiments.py`](file:///c:/Users/imsto/Desktop/Final%20Year%20Paper/Old%20Repo%20Data/scripts/run_cross_modal_experiments.py#L424-L437), Lines 424–437.
  - File: [`experiments/cross_modal_attention_2026_10_30/metrics/statistical_comparison.json`](file:///c:/Users/imsto/Desktop/Final%20Year%20Paper/Old%20Repo%20Data/experiments/cross_modal_attention_2026_10_30/metrics/statistical_comparison.json).
- **Description:**
  The experiment evaluated three distinct pairwise comparisons across four metrics and three random seeds (36 individual statistical hypothesis tests). All p-values were evaluated against unadjusted $\alpha = 0.05$.
  - In Seed 123, CMANet-Full vs MultimodalFusionNet showed a nominal p-value for Accuracy of $p = 0.0368$.
  - Under Bonferroni correction for 3 seeds ($\alpha = 0.05 / 3 = 0.0167$), this p-value is **not statistically significant**.
- **Impact on Conclusions:**
  Strengthens the central null finding. Adjusting for multiple testing eliminates the spurious single-seed significance in Seed 123, reinforcing that CMANet-Full is never statistically superior to MultimodalFusionNet.
- **Required Resolution:**
  Include a discussion of family-wise error rate (FWER) and Bonferroni/Holm correction in the final statistical reporting.

---

### Issue 5: Downsampled Training Scale (16k) and Sequence Truncation (L=256)
- **Classification:** `METHODOLOGICAL ISSUE` (Scope Limitation)
- **Severity:** Low (Standard Academic Project Constraint)
- **Evidence:**
  - Script: [`scripts/prepare_ember_data_corrected.py`](file:///c:/Users/imsto/Desktop/Final%20Year%20Paper/Old%20Repo%20Data/scripts/prepare_ember_data_corrected.py#L241-L244), Lines 241–244.
  - Manifest: [`experiments/baseline_correction_2026_10_30/data/sample_manifest.json`](file:///c:/Users/imsto/Desktop/Final%20Year%20Paper/Old%20Repo%20Data/experiments/baseline_correction_2026_10_30/data/sample_manifest.json#L402-L415).
- **Description:**
  1. Training uses 16,000 samples (2.67% of the 600,000 labeled EMBER training set).
  2. Modality B truncates imported API token sequences to 256 tokens.
- **Audit Verification:**
  - The 16,000-sample scale was selected to enable multi-seed, multi-model execution within local CPU RAM and Colab quotas.
  - It is sufficient to achieve clear statistical significance between multimodal models and unimodal baselines ($p < 0.0001$).
  - However, multi-head self-attention and cross-attention architectures are notoriously data-hungry and lack convolutional inductive bias. In the discussion section, this must be explicitly framed as an inductive bias deficit on limited training data.
- **Required Resolution:**
  Highlight as a primary research limitation and future work direction in the final thesis manuscript.

---

## 3. Methodological Items Fully Verified (`VERIFIED`)

| Audit Check | Status | Evidence / Verification Method |
|:---|:---:|:---|
| **Reservoir Sampling (Algorithm R) Parity** | `VERIFIED` | Full 800,000-line streaming traversal confirmed in `prepare_ember_data_corrected.py`; shard quotas verified against population. |
| **Train/Val Split Disjointness** | `VERIFIED` | SHA-256 intersection of 16k train and 4k val hashes confirmed exactly 0 (`overlap == 0`). |
| **Cryptographic Hash Invariance** | `VERIFIED` | Train SHA-256 (`1164ca...`) and Val SHA-256 (`6b25a3...`) matched across all 18 runs. |
| **Pre-Parsed Feature Isolation** | `VERIFIED` | `PEFeatureExtractor.process_raw_features` operates strictly on JSON dictionaries; LIEF runtime is never invoked. |
| **SectionInfo Compatibility Patch** | `VERIFIED` | Option B `[list(entry_val)]` character-level hashing strictly verified against scikit-learn 0.20 behavior. |
| **Data Leakage in Preprocessing** | `VERIFIED` | Scaler fitted only on train; Vocabulary built only on train; Byte statistics normalized per-sample. |
| **18 Benchmark Metric Reproducibility** | `VERIFIED` | All 18 run metrics reproduced to 4 decimal places from raw prediction NPZ arrays. |
| **Model Parameter Counts & FLOPs** | `VERIFIED` | Total params, non-embedding params, FLOPs, and MACs independently computed for all 10 model classes. |
| **Official Test Set Quarantine** | `VERIFIED` | `test_features.jsonl` (200,000 samples) strictly quarantined; 0 models evaluated on test set. |
| **Multimodal Dominance over Unimodal** | `VERIFIED` | FusionNet and CMANet-Full statistically dominate StructMLP across 3/3 seeds ($p < 0.0002$). |

---

## 4. Final Verdict

- **Complete & Audited:** 44 total experimental runs (5 initial, 15 baseline correction, 6 smoke, 18 cross-modal) are fully inventoried and cryptographically grounded.
- **Primary Issue to Note:** The F1 transcription discrepancy in [`reports/cmanet_post_experiment_analysis.md`](file:///c:/Users/imsto/Desktop/Final%20Year%20Paper/Old%20Repo%20Data/reports/cmanet_post_experiment_analysis.md) is resolved: the ground truth F1 is $92.38\%$, not $92.21\%$.
- **Project Readiness:** The repository is methodologically sound, leakage-free, and ready for primary model freezing.

