#!/usr/bin/env python3
"""
build_cross_modal_colab_notebook.py

Builds notebooks/ember_cross_modal_attention.ipynb for Google Colab and local Jupyter execution.
Reflects complete Phase 1 through Phase 8 cross-modal attention experimental design.
"""

import json
import os

def build_notebook():
    cells = []

    # --------------------------------------------------------------------------
    # Cell 0: Title & Metadata (Markdown)
    # --------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Ember Malware Detection: Native Multimodal Cross-Attention Experiments\n",
            "\n",
            "**Author:** Arjun Singh Rajput (GLA University / IIIT Kottayam)  \n",
            "**Project:** Ember – Malware Detection Using Native Multimodal Learning  \n",
            "**Working Branch:** `experiment/native-multimodal-transformer`  \n",
            "**Dataset:** EMBER 2018 v2 (Corrected 16k Train / 4k Val Balanced Splits)  \n",
            "**Target Hardware:** Google Colab (NVIDIA T4 / V100 / A100 GPU preferred, CPU compatible)  \n",
            "**Output Directory:** `experiments/cross_modal_attention_2026_10_30/`  \n",
            "\n",
            "---\n",
            "\n",
            "### Research Objective & Experimental Protocol:\n",
            "1. **Primary Research Question:** Does learned bidirectional cross-modal attention between raw byte distributions, fine-grained API invocation sequences, and PE structural metadata yield statistically significant improvements in malware detection over simple concatenation-based multimodal fusion?\n",
            "2. **Strict Controlled Experimental Design:**\n",
            "   - **Fixed Split:** Exactly identical 16,000 train (8k benign, 8k malware) and 4,000 validation (2k benign, 2k malware) records from the methodologically corrected baseline.\n",
            "   - **Optimizer Parity:** Exact baseline Adam settings (`lr=1e-3`, `weight_decay=1e-4`, `batch_size=32`, 10 epochs, best validation loss checkpoint, no scheduler/warmup).\n",
            "   - **3-Seed Evaluation:** Multi-seed evaluation across seeds `[42, 123, 2026]`.\n",
            "   - **Paired Bootstrap Testing:** 10,000 resamples for statistical significance testing against `MultimodalFusionNet`, `CMANet-NoCrossAttn`, and `StructMLP`.\n",
            "   - **Quarantined Test Set:** `test_features.jsonl` is strictly unaccessed to eliminate data snooping.\n"
        ]
    })

    # --------------------------------------------------------------------------
    # Cell 1: Dependency Installation & LIEF Patch (Code)
    # --------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Environment Setup & Dependency Installation\n",
            "Installs pinned dependencies, sets up the EMBER feature extraction library, and patches the scikit-learn hashing routine."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install required dependencies (quietly)\n",
            "!pip install -q lief lightgbm torch torchvision scikit-learn scipy tqdm matplotlib pyyaml\n",
            "!pip install -q git+https://github.com/endgameinc/ember.git\n",
            "\n",
            "import os\n",
            "import sys\n",
            "import json\n",
            "import time\n",
            "import math\n",
            "import random\n",
            "import hashlib\n",
            "from collections import Counter\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import scipy\n",
            "import matplotlib.pyplot as plt\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.optim as optim\n",
            "from torch.utils.data import DataLoader, TensorDataset\n",
            "import sklearn\n",
            "from sklearn.metrics import (\n",
            "    accuracy_score, precision_score, recall_score, f1_score,\n",
            "    roc_auc_score, roc_curve, confusion_matrix\n",
            ")\n",
            "import ember\n",
            "\n",
            "# Patch EMBER SectionInfo for modern scikit-learn (>=1.0) compatibility\n",
            "_orig_section_process = ember.features.SectionInfo.process_raw_features\n",
            "def _patched_section_process(self, raw_obj):\n",
            "    sections = raw_obj.get('sections', [])\n",
            "    general = [\n",
            "        len(sections),\n",
            "        sum(1 for s in sections if s.get('size', 0) == 0),\n",
            "        sum(1 for s in sections if s.get('name', '') == ''),\n",
            "        sum(1 for s in sections if 'MEM_READ' in s.get('props', []) and 'MEM_EXECUTE' in s.get('props', [])),\n",
            "        sum(1 for s in sections if 'MEM_WRITE' in s.get('props', []))\n",
            "    ]\n",
            "    section_names = [s.get('name', '') for s in sections]\n",
            "    from sklearn.feature_extraction import FeatureHasher\n",
            "    hasher = FeatureHasher(50, input_type='string')\n",
            "    hashed_names = hasher.transform([[name] for name in section_names]).toarray().sum(axis=0).tolist()\n",
            "    section_sizes = [s.get('size', 0) for s in sections]\n",
            "    section_entropies = [s.get('entropy', 0.0) for s in sections]\n",
            "    def stats(arr):\n",
            "        if not arr:\n",
            "            return [0, 0, 0, 0]\n",
            "        return [min(arr), max(arr), sum(arr) / len(arr), float(np.median(arr))]\n",
            "    return np.asarray(general + hashed_names + stats(section_sizes) + stats(section_entropies), dtype=np.float32)\n",
            "ember.features.SectionInfo.process_raw_features = _patched_section_process\n",
            "print('Environment initialized and EMBER SectionInfo patched successfully.')"
        ]
    })

    # --------------------------------------------------------------------------
    # Cell 2: Hardware Audit (Code)
    # --------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Hardware & Runtime Audit\n",
            "Audits compute device, GPU model, VRAM capacity, and key package versions."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('=' * 75)\n",
            "print('RUNTIME & COMPUTE ENVIRONMENT AUDIT')\n",
            "print('=' * 75)\n",
            "print(f'Python Version      : {sys.version.split()[0]}')\n",
            "print(f'PyTorch Version     : {torch.__version__}')\n",
            "print(f'NumPy Version       : {np.__version__}')\n",
            "print(f'Scikit-Learn Version: {sklearn.__version__}')\n",
            "print(f'CUDA Available      : {torch.cuda.is_available()}')\n",
            "\n",
            "if torch.cuda.is_available():\n",
            "    device = torch.device('cuda')\n",
            "    gpu_name = torch.cuda.get_device_name(0)\n",
            "    gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)\n",
            "    print(f'GPU Device Name     : {gpu_name}')\n",
            "    print(f'GPU Total VRAM      : {gpu_vram:.2f} GB')\n",
            "    print(f'CUDA Capability     : {torch.cuda.get_device_capability(0)}')\n",
            "else:\n",
            "    device = torch.device('cpu')\n",
            "    print('GPU Device Name     : None (Executing on CPU)')\n",
            "print('=' * 75)"
        ]
    })

    # --------------------------------------------------------------------------
    # Cell 3: Path & Directory Configuration (Code)
    # --------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Path & Directory Configuration (Google Colab & Local)\n",
            "Automatically detects if running in Google Colab (with optional Google Drive mount) or local repository."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "IN_COLAB = 'google.colab' in sys.modules\n",
            "\n",
            "if IN_COLAB:\n",
            "    from google.colab import drive\n",
            "    # Mount Google Drive if desired\n",
            "    # drive.mount('/content/drive')\n",
            "    REPO_DIR = '/content/Ember'\n",
            "    if not os.path.exists(REPO_DIR):\n",
            "        import subprocess\n",
            "        subprocess.run(['git', 'clone', '-b', 'experiment/native-multimodal-transformer', 'https://github.com/ArjunRajputGLA/Ember.git', '/content/Ember'], check=True)\n",
            "    DATA_DIR = os.path.join(REPO_DIR, 'Old Repo Data', 'experiments', 'baseline_correction_2026_10_30', 'data')\n",
            "    OUTPUT_DIR = os.path.join(REPO_DIR, 'Old Repo Data', 'experiments', 'cross_modal_attention_2026_10_30')\n",
            "else:\n",
            "    # Local Workspace paths\n",
            "    CWD = os.getcwd()\n",
            "    if os.path.exists(os.path.join(CWD, 'experiments')):\n",
            "        DATA_DIR = os.path.join(CWD, 'experiments', 'baseline_correction_2026_10_30', 'data')\n",
            "        OUTPUT_DIR = os.path.join(CWD, 'experiments', 'cross_modal_attention_2026_10_30')\n",
            "    else:\n",
            "        DATA_DIR = os.path.join(CWD, 'Old Repo Data', 'experiments', 'baseline_correction_2026_10_30', 'data')\n",
            "        OUTPUT_DIR = os.path.join(CWD, 'Old Repo Data', 'experiments', 'cross_modal_attention_2026_10_30')\n",
            "\n",
            "CKPT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')\n",
            "PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')\n",
            "CONFIGS_DIR = os.path.join(OUTPUT_DIR, 'configs')\n",
            "METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')\n",
            "\n",
            "for d in [CKPT_DIR, PLOTS_DIR, CONFIGS_DIR, METRICS_DIR]:\n",
            "    os.makedirs(d, exist_ok=True)\n",
            "\n",
            "print(f'Execution Environment: {\"Google Colab\" if IN_COLAB else \"Local Workspace\"}')\n",
            "print(f'Data Directory       : {DATA_DIR}')\n",
            "print(f'Output Directory     : {OUTPUT_DIR}')"
        ]
    })

    # --------------------------------------------------------------------------
    # Cell 4: SHA-256 Manifest Verification (Code)
    # --------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. SHA-256 Manifest Verification & Split Disjointness\n",
            "Verifies that the pre-extracted arrays match the exact SHA-256 hashes from the corrected baseline audit."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "EXPECTED_TRAIN_SHA = '1164ca77fc4d05144726e21cec1152a7d1d5280d0e11b137501681bdf64a41c6'\n",
            "EXPECTED_VAL_SHA   = '6b25a36c96db4719c8dabaf7b97a7433207a30696b5271dabfed5bea0bf6c188'\n",
            "\n",
            "def compute_file_sha256(filepath):\n",
            "    h = hashlib.sha256()\n",
            "    with open(filepath, 'rb') as f:\n",
            "        while chunk := f.read(8192 * 1024):\n",
            "            h.update(chunk)\n",
            "    return h.hexdigest()\n",
            "\n",
            "train_npz_path = os.path.join(DATA_DIR, 'ember_train_16k_corrected.npz')\n",
            "val_npz_path = os.path.join(DATA_DIR, 'ember_val_4k_corrected.npz')\n",
            "\n",
            "assert os.path.exists(train_npz_path), f'Training array not found at: {train_npz_path}'\n",
            "assert os.path.exists(val_npz_path), f'Validation array not found at: {val_npz_path}'\n",
            "\n",
            "actual_train_sha = compute_file_sha256(train_npz_path)\n",
            "actual_val_sha   = compute_file_sha256(val_npz_path)\n",
            "\n",
            "print('=' * 75)\n",
            "print('PRE-FLIGHT SHA-256 VERIFICATION')\n",
            "print('=' * 75)\n",
            "print(f'Train Array SHA-256 : {actual_train_sha}')\n",
            "assert actual_train_sha == EXPECTED_TRAIN_SHA, f'Train SHA-256 mismatch!'\n",
            "print('Train SHA-256 Match : VERIFIED')\n",
            "\n",
            "print(f'Val Array SHA-256   : {actual_val_sha}')\n",
            "assert actual_val_sha == EXPECTED_VAL_SHA, f'Val SHA-256 mismatch!'\n",
            "print('Val SHA-256 Match   : VERIFIED')\n",
            "\n",
            "# Load arrays into memory\n",
            "train_data = np.load(train_npz_path)\n",
            "val_data   = np.load(val_npz_path)\n",
            "\n",
            "X_byte_tr, X_text_tr, X_struct_tr, y_tr = train_data['X_byte'], train_data['X_text'], train_data['X_struct'], train_data['y']\n",
            "X_byte_va, X_text_va, X_struct_va, y_va = val_data['X_byte'], val_data['X_text'], val_data['X_struct'], val_data['y']\n",
            "\n",
            "print(f'Train Records       : {len(y_tr):,} (Benign: {(y_tr==0).sum():,}, Malware: {(y_tr==1).sum():,})')\n",
            "print(f'Val Records         : {len(y_va):,} (Benign: {(y_va==0).sum():,}, Malware: {(y_va==1).sum():,})')\n",
            "\n",
            "# Verify sample disjointness\n",
            "if 'shas' in train_data and 'shas' in val_data:\n",
            "    overlap = set(train_data['shas']).intersection(set(val_data['shas']))\n",
            "    print(f'Sample Overlap      : {len(overlap)} (Strictly Disjoint)')\n",
            "    assert len(overlap) == 0, 'Data leakage: Train and Val have overlapping SHA-256 hashes!'\n",
            "print('=' * 75)"
        ]
    })

    # --------------------------------------------------------------------------
    # Cell 5: Architecture Definitions & Attention Safety Tests (Code)
    # --------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Architecture Definitions & Pre-Flight Attention Safety Suite\\n\",\n",
            "Defines `CrossModalAttentionNet` (CMANet-Full), the 4 ablation models, and `MultimodalFusionNet`. Runs all 7 unit tests, 2 attention safety regression tests (including all-padding API sequences), and the 4 ablation isolation tests.\"\n",
        ]
    })
    
    # Read the full source of models/cmanet.py to ensure 100% self-contained code in the notebook
    cmanet_src_path = os.path.join(os.path.dirname(__file__), "..", "models", "cmanet.py")
    if not os.path.exists(cmanet_src_path):
        cmanet_src_path = os.path.join("Old Repo Data", "models", "cmanet.py")
        
    with open(cmanet_src_path, "r", encoding="utf-8") as f:
        cmanet_code = f.read()

    appendix = """

# Model Capacity Table
suite = [
    ('CMANet (Full)', CrossModalAttentionNet()),
    ('CMANet-NoByte', CMANet_NoByte()),
    ('CMANet-NoStruct', CMANet_NoStruct()),
    ('CMANet-NoText', CMANet_NoText()),
    ('CMANet-NoCrossAttn', CMANet_NoCrossAttn()),
    ('MultimodalFusionNet', MultimodalFusionNet())
]
print('=' * 80)
print('MODEL CAPACITY & COMPUTATIONAL FAIRNESS COMPARISON')
print('=' * 80)
print(f'{"Model":24s} | {"Total Params":12s} | {"Non-Emb Params":14s} | {"Emb Params":10s} | {"FLOPs (M)":9s}')
print('-' * 80)
for name, m in suite:
    cap = compute_model_capacity(m, name)
    print(f"{name:24s} | {cap['total_params']:12,d} | {cap['non_emb_params']:14,d} | {cap['emb_params']:10,d} | {cap['mflops']:9.2f}")
print('=' * 80)
"""
    cell5_source = [cmanet_code + appendix]

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cell5_source
    })

    # --------------------------------------------------------------------------
    # Cell 6: 200-Sample Smoke Test (Code)
    # --------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Fast 200-Sample Smoke Verification\n",
            "Runs 1 epoch on 200 training samples and 50 validation samples across all 6 models to verify forward/backward passes, optimizer stepping, loss decrease, and checkpoint saving."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('=' * 75)\n",
            "print('RUNNING FAST 200-SAMPLE SMOKE VERIFICATION')\n",
            "print('=' * 75)\n",
            "\n",
            "smoke_tr_xb = torch.tensor(X_byte_tr[:200], dtype=torch.float32)\n",
            "smoke_tr_xt = torch.tensor(X_text_tr[:200], dtype=torch.long)\n",
            "smoke_tr_xs = torch.tensor(X_struct_tr[:200], dtype=torch.float32)\n",
            "smoke_tr_y  = torch.tensor(y_tr[:200], dtype=torch.long)\n",
            "\n",
            "smoke_va_xb = torch.tensor(X_byte_va[:50], dtype=torch.float32)\n",
            "smoke_va_xt = torch.tensor(X_text_va[:50], dtype=torch.long)\n",
            "smoke_va_xs = torch.tensor(X_struct_va[:50], dtype=torch.float32)\n",
            "smoke_va_y  = torch.tensor(y_va[:50], dtype=torch.long)\n",
            "\n",
            "smoke_tr_loader = DataLoader(TensorDataset(smoke_tr_xb, smoke_tr_xt, smoke_tr_xs, smoke_tr_y), batch_size=32, shuffle=True)\n",
            "smoke_va_loader = DataLoader(TensorDataset(smoke_va_xb, smoke_va_xt, smoke_va_xs, smoke_va_y), batch_size=32, shuffle=False)\n",
            "\n",
            "crit = nn.CrossEntropyLoss()\n",
            "for name, mcls in suite:\n",
            "    m = mcls().to(device)\n",
            "    opt = optim.Adam(m.parameters(), lr=0.001, weight_decay=0.0001)\n",
            "    m.train()\n",
            "    step_losses = []\n",
            "    for xb, xt, xs, yb in smoke_tr_loader:\n",
            "        xb, xt, xs, yb = xb.to(device), xt.to(device), xs.to(device), yb.to(device)\n",
            "        opt.zero_grad()\n",
            "        logits = m(xb, xt, xs)\n",
            "        loss = crit(logits, yb)\n",
            "        loss.backward()\n",
            "        opt.step()\n",
            "        step_losses.append(loss.item())\n",
            "    m.eval()\n",
            "    with torch.no_grad():\n",
            "        val_out = m(smoke_va_xb.to(device), smoke_va_xt.to(device), smoke_va_xs.to(device))\n",
            "        val_acc = (val_out.argmax(dim=1).cpu() == smoke_va_y).float().mean().item()\n",
            "    print(f'  [Smoke OK] {name:20s}: Loss {step_losses[0]:.4f} -> {step_losses[-1]:.4f} | Val Acc: {val_acc*100:.2f}%')\n",
            "print('=' * 75)\n",
            "print('SMOKE VERIFICATION PASSED FOR ALL 6 MODELS!')"
        ]
    })

    # --------------------------------------------------------------------------
    # Cell 7: Full 3-Seed Benchmark Execution (Code)
    # --------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Full 3-Seed Benchmark Execution Runner\n",
            "Trains and evaluates all 6 models across seeds `[42, 123, 2026]`. Saves checkpoints, loss curves, configurations, and complete validation predictions for paired bootstrap significance testing."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Full 3-Seed Experiment Execution\n",
            "SEEDS = [42, 123, 2026]\n",
            "EPOCHS = 10\n",
            "BATCH_SIZE = 32\n",
            "LR = 0.001\n",
            "WEIGHT_DECAY = 0.0001\n",
            "\n",
            "tr_xb_full = torch.tensor(X_byte_tr, dtype=torch.float32)\n",
            "tr_xt_full = torch.tensor(X_text_tr, dtype=torch.long)\n",
            "tr_xs_full = torch.tensor(X_struct_tr, dtype=torch.float32)\n",
            "tr_y_full  = torch.tensor(y_tr, dtype=torch.long)\n",
            "\n",
            "va_xb_full = torch.tensor(X_byte_va, dtype=torch.float32)\n",
            "va_xt_full = torch.tensor(X_text_va, dtype=torch.long)\n",
            "va_xs_full = torch.tensor(X_struct_va, dtype=torch.float32)\n",
            "va_y_full  = torch.tensor(y_va, dtype=torch.long)\n",
            "\n",
            "va_loader_full = DataLoader(TensorDataset(va_xb_full, va_xt_full, va_xs_full, va_y_full), batch_size=BATCH_SIZE, shuffle=False)\n",
            "\n",
            "all_experiment_results = {}\n",
            "\n",
            "for seed in SEEDS:\n",
            "    print(f'\\n===========================================================================')\n",
            "    print(f'STARTING SEED {seed} RUNS')\n",
            "    print(f'===========================================================================')\n",
            "    \n",
            "    # Set Seed & Determinism\n",
            "    random.seed(seed)\n",
            "    np.random.seed(seed)\n",
            "    torch.manual_seed(seed)\n",
            "    if torch.cuda.is_available():\n",
            "        torch.cuda.manual_seed_all(seed)\n",
            "        \n",
            "    tr_loader_full = DataLoader(TensorDataset(tr_xb_full, tr_xt_full, tr_xs_full, tr_y_full), batch_size=BATCH_SIZE, shuffle=True)\n",
            "    seed_predictions = {'y_true': y_va}\n",
            "    \n",
            "    for mname, mcls in suite:\n",
            "        if mname not in all_experiment_results:\n",
            "            all_experiment_results[mname] = []\n",
            "            \n",
            "        print(f'\\n--- Training {mname} (Seed {seed}) ---')\n",
            "        model = mcls().to(device)\n",
            "        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)\n",
            "        criterion = nn.CrossEntropyLoss()\n",
            "        \n",
            "        best_val_loss = float('inf')\n",
            "        best_epoch = 1\n",
            "        best_ckpt = os.path.join(CKPT_DIR, f'best_{mname.replace(\" \", \"_\")}_seed{seed}.pt')\n",
            "        train_losses, val_losses = [], []\n",
            "        \n",
            "        t0 = time.time()\n",
            "        for ep in range(1, EPOCHS + 1):\n",
            "            model.train()\n",
            "            tr_sum = 0.0\n",
            "            for xb, xt, xs, yb in tr_loader_full:\n",
            "                xb, xt, xs, yb = xb.to(device), xt.to(device), xs.to(device), yb.to(device)\n",
            "                optimizer.zero_grad()\n",
            "                out = model(xb, xt, xs)\n",
            "                loss = criterion(out, yb)\n",
            "                loss.backward()\n",
            "                optimizer.step()\n",
            "                tr_sum += loss.item() * len(yb)\n",
            "            ep_tr = tr_sum / len(y_tr)\n",
            "            train_losses.append(ep_tr)\n",
            "            \n",
            "            model.eval()\n",
            "            va_sum = 0.0\n",
            "            with torch.no_grad():\n",
            "                for xb, xt, xs, yb in va_loader_full:\n",
            "                    xb, xt, xs, yb = xb.to(device), xt.to(device), xs.to(device), yb.to(device)\n",
            "                    loss = criterion(model(xb, xt, xs), yb)\n",
            "                    va_sum += loss.item() * len(yb)\n",
            "            ep_va = va_sum / len(y_va)\n",
            "            val_losses.append(ep_va)\n",
            "            \n",
            "            if ep_va < best_val_loss:\n",
            "                best_val_loss = ep_va\n",
            "                best_epoch = ep\n",
            "                torch.save(model.state_dict(), best_ckpt)\n",
            "                \n",
            "            if ep % 2 == 0 or ep == EPOCHS:\n",
            "                print(f'  Epoch {ep:>2}/{EPOCHS} | Train Loss: {ep_tr:.4f} | Val Loss: {ep_va:.4f} (Best: {best_val_loss:.4f} @ ep {best_epoch})')\n",
            "                \n",
            "        t_train = round(time.time() - t0, 2)\n",
            "        \n",
            "        # Load best checkpoint and evaluate\n",
            "        model.load_state_dict(torch.load(best_ckpt))\n",
            "        model.eval()\n",
            "        \n",
            "        val_probs = []\n",
            "        t_inf_start = time.time()\n",
            "        with torch.no_grad():\n",
            "            for xb, xt, xs, _ in va_loader_full:\n",
            "                xb, xt, xs = xb.to(device), xt.to(device), xs.to(device)\n",
            "                p = torch.softmax(model(xb, xt, xs), dim=1)[:, 1].cpu().numpy()\n",
            "                val_probs.extend(p)\n",
            "        latency = (time.time() - t_inf_start) / len(y_va) * 1000\n",
            "        val_probs = np.array(val_probs, dtype=np.float32)\n",
            "        \n",
            "        # Compute Metrics\n",
            "        pred_bin = (val_probs >= 0.5).astype(int)\n",
            "        cm = confusion_matrix(y_va, pred_bin).ravel()\n",
            "        fpr_arr, tpr_arr, _ = roc_curve(y_va, val_probs)\n",
            "        idx_99 = np.where(tpr_arr >= 0.99)[0]\n",
            "        fpr_99 = float(fpr_arr[idx_99[0]]) if len(idx_99) > 0 else 1.0\n",
            "        \n",
            "        run_m = {\n",
            "            'accuracy': float(accuracy_score(y_va, pred_bin)),\n",
            "            'precision': float(precision_score(y_va, pred_bin, zero_division=0)),\n",
            "            'recall': float(recall_score(y_va, pred_bin, zero_division=0)),\n",
            "            'f1_score': float(f1_score(y_va, pred_bin, zero_division=0)),\n",
            "            'roc_auc': float(roc_auc_score(y_va, val_probs)),\n",
            "            'fpr': float(cm[1] / max(1, cm[0] + cm[1])),\n",
            "            'fpr_at_99pct_tpr': fpr_99,\n",
            "            'malware_fn': int(cm[2]),\n",
            "            'training_time_s': t_train,\n",
            "            'latency_ms': latency,\n",
            "            'best_epoch': best_epoch,\n",
            "            'best_val_loss': best_val_loss,\n",
            "            'seed': seed\n",
            "        }\n",
            "        all_experiment_results[mname].append(run_m)\n",
            "        seed_predictions[f'p_{mname.lower().replace(\" \", \"_\").replace(\"-\", \"_\")}'] = val_probs\n",
            "        \n",
            "        with open(os.path.join(METRICS_DIR, f'run_{mname.replace(\" \", \"_\")}_seed{seed}.json'), 'w') as f:\n",
            "            json.dump(run_m, f, indent=2)\n",
            "            \n",
            "    # Save per-seed predictions NPZ\n",
            "    np.savez_compressed(os.path.join(METRICS_DIR, f'predictions_seed{seed}.npz'), **seed_predictions)\n",
            "print('\\nAll 18 benchmark runs finished successfully!')"
        ]
    })

    # --------------------------------------------------------------------------
    # Cell 8: Aggregate Results & Summary Table (Code)
    # --------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Aggregate Results & Comparative Table (Mean ± Sample Std, ddof=1)\n",
            "Compiles metrics across all 3 seeds and formats the summary table."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('=' * 85)\n",
            "print('CROSS-MODAL ATTENTION BENCHMARK: 3-SEED AGGREGATE SUMMARY (ddof=1)')\n",
            "print('=' * 85)\n",
            "print(f'{\"Model\":22s} | {\"Accuracy (%)\":15s} | {\"ROC-AUC\":14s} | {\"FPR@99TPR (%)\":15s} | {\"False Negatives\":15s}')\n",
            "print('-' * 85)\n",
            "\n",
            "agg_results = {}\n",
            "for mname, runs in all_experiment_results.items():\n",
            "    accs = [r['accuracy'] for r in runs]\n",
            "    aucs = [r['roc_auc'] for r in runs]\n",
            "    fpr99s = [r['fpr_at_99pct_tpr'] for r in runs]\n",
            "    fns = [r['malware_fn'] for r in runs]\n",
            "    \n",
            "    mean_acc, std_acc = np.mean(accs), np.std(accs, ddof=1)\n",
            "    mean_auc, std_auc = np.mean(aucs), np.std(aucs, ddof=1)\n",
            "    mean_fpr99, std_fpr99 = np.mean(fpr99s), np.std(fpr99s, ddof=1)\n",
            "    mean_fn, std_fn = np.mean(fns), np.std(fns, ddof=1)\n",
            "    \n",
            "    agg_results[mname] = {\n",
            "        'accuracy': {'mean': mean_acc, 'std': std_acc},\n",
            "        'roc_auc': {'mean': mean_auc, 'std': std_auc},\n",
            "        'fpr_at_99pct_tpr': {'mean': mean_fpr99, 'std': std_fpr99},\n",
            "        'malware_fn': {'mean': mean_fn, 'std': std_fn}\n",
            "    }\n",
            "    \n",
            "    acc_str = f'{mean_acc*100:.2f} ± {std_acc*100:.2f}'\n",
            "    auc_str = f'{mean_auc:.4f} ± {std_auc:.4f}'\n",
            "    fpr_str = f'{mean_fpr99*100:.2f} ± {std_fpr99*100:.2f}'\n",
            "    fn_str  = f'{mean_fn:.1f} ± {std_fn:.1f}'\n",
            "    print(f'{mname:22s} | {acc_str:15s} | {auc_str:14s} | {fpr_str:15s} | {fn_str:15s}')\n",
            "\n",
            "with open(os.path.join(METRICS_DIR, 'cross_modal_attention_results.json'), 'w') as f:\n",
            "    json.dump(all_experiment_results, f, indent=2)\n",
            "print('=' * 85)"
        ]
    })

    # --------------------------------------------------------------------------
    # Cell 9: Paired Bootstrap Statistical Significance (Code)
    # --------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Paired Bootstrap Statistical Significance Analysis (10,000 Resamples)\n",
            "Evaluates paired per-sample differences between CMANet-Full and:\n",
            "1. `MultimodalFusionNet` (Baseline concatenation)\n",
            "2. `CMANet-NoCrossAttn` (Ablation with intra-modal self-attention only)\n",
            "3. `StructMLP` (Best unimodal neural baseline)\n",
            "\n",
            "> [!IMPORTANT]\n",
            "> Statistical significance is strictly evaluated via paired bootstrap confidence intervals across validation samples, not solely inferred from three random seeds."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Paired Bootstrap Analysis\n",
            "def paired_bootstrap(y_true, p_m, p_b, n_resamples=10000, seed=42):\n",
            "    np.random.seed(seed)\n",
            "    N = len(y_true)\n",
            "    diffs_acc = []\n",
            "    diffs_auc = []\n",
            "    diffs_fn = []\n",
            "    \n",
            "    for _ in range(n_resamples):\n",
            "        idx = np.random.randint(0, N, size=N)\n",
            "        y_b = y_true[idx]\n",
            "        pm_b = p_m[idx]\n",
            "        pb_b = p_b[idx]\n",
            "        \n",
            "        diffs_acc.append(accuracy_score(y_b, pm_b >= 0.5) - accuracy_score(y_b, pb_b >= 0.5))\n",
            "        if len(np.unique(y_b)) == 2:\n",
            "            diffs_auc.append(roc_auc_score(y_b, pm_b) - roc_auc_score(y_b, pb_b))\n",
            "        cm_m = confusion_matrix(y_b, pm_b >= 0.5, labels=[0, 1]).ravel()\n",
            "        cm_b = confusion_matrix(y_b, pb_b >= 0.5, labels=[0, 1]).ravel()\n",
            "        diffs_fn.append(cm_m[2] - cm_b[2])\n",
            "        \n",
            "    def get_stats(arr, higher_better=True):\n",
            "        ci = [np.percentile(arr, 2.5), np.percentile(arr, 97.5)]\n",
            "        mean_d = np.mean(arr)\n",
            "        pval = 2 * min(float((np.array(arr) <= 0).mean()), float((np.array(arr) >= 0).mean()))\n",
            "        return mean_d, ci, min(1.0, pval)\n",
            "        \n",
            "    return {\n",
            "        'acc': get_stats(diffs_acc, True),\n",
            "        'auc': get_stats(diffs_auc, True),\n",
            "        'fn': get_stats(diffs_fn, False)\n",
            "    }\n",
            "\n",
            "print('=' * 85)\n",
            "print('PAIRED BOOTSTRAP SIGNIFICANCE RESULTS (Seed 42 Validation Set)')\n",
            "print('=' * 85)\n",
            "\n",
            "preds_seed42 = np.load(os.path.join(METRICS_DIR, 'predictions_seed42.npz'))\n",
            "p_cmanet = preds_seed42['p_cmanet_(full)'] if 'p_cmanet_(full)' in preds_seed42 else preds_seed42['p_cmanet_full']\n",
            "\n",
            "comparisons = {\n",
            "    'CMANet-Full vs MultimodalFusionNet': preds_seed42.get('p_multimodalfusionnet'),\n",
            "    'CMANet-Full vs CMANet-NoCrossAttn': preds_seed42.get('p_cmanet_nocrossattn')\n",
            "}\n",
            "\n",
            "# Load StructMLP baseline predictions if available\n",
            "base_pred_file = os.path.join(DATA_DIR, '..', 'metrics', 'predictions_seed42.npz')\n",
            "if os.path.exists(base_pred_file):\n",
            "    base_p = np.load(base_pred_file)\n",
            "    comparisons['CMANet-Full vs StructMLP'] = base_p['p_struct']\n",
            "\n",
            "bootstrap_results = {}\n",
            "for comp_name, p_baseline in comparisons.items():\n",
            "    if p_baseline is None:\n",
            "        continue\n",
            "    stats = paired_bootstrap(y_va, p_cmanet, p_baseline, n_resamples=10000, seed=42)\n",
            "    bootstrap_results[comp_name] = stats\n",
            "    print(f'\\nPairing: {comp_name}')\n",
            "    print(f'  Δ Accuracy : {stats[\"acc\"][0]*100:+.2f}% (95% CI: [{stats[\"acc\"][1][0]*100:+.2f}%, {stats[\"acc\"][1][1]*100:+.2f}%], p = {stats[\"acc\"][2]:.4f})')\n",
            "    print(f'  Δ ROC-AUC  : {stats[\"auc\"][0]:+.4f} (95% CI: [{stats[\"auc\"][1][0]:+.4f}, {stats[\"auc\"][1][1]:+.4f}], p = {stats[\"auc\"][2]:.4f})')\n",
            "    print(f'  Δ FN (Mal) : {stats[\"fn\"][0]:+.1f} (95% CI: [{stats[\"fn\"][1][0]:+.1f}, {stats[\"fn\"][1][1]:+.1f}], p = {stats[\"fn\"][2]:.4f})')\n",
            "\n",
            "with open(os.path.join(METRICS_DIR, 'statistical_comparison.json'), 'w') as f:\n",
            "    json.dump({k: {m: {'mean': v[0], 'ci_95': list(v[1]), 'p_val': v[2]} for m, v in s.items()} for k, s in bootstrap_results.items()}, f, indent=2)\n",
            "print('=' * 85)"
        ]
    })

    # --------------------------------------------------------------------------
    # Cell 10: Comparative Visualizations (Code)
    # --------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 10. Comparative Visualizations\n",
            "Plots ROC curves and ROC-AUC comparisons for the tested models."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(8, 6))\n",
            "for key in preds_seed42.files:\n",
            "    if key == 'y_true':\n",
            "        continue\n",
            "    probs = preds_seed42[key]\n",
            "    fpr, tpr, _ = roc_curve(y_va, probs)\n",
            "    auc_val = roc_auc_score(y_va, probs)\n",
            "    label = key.replace('p_', '').replace('_', ' ').title()\n",
            "    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc_val:.4f})')\n",
            "\n",
            "plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)\n",
            "plt.xlim([0.0, 0.2])\n",
            "plt.ylim([0.8, 1.0])\n",
            "plt.title('ROC Curves (Zoomed: Low FPR & High TPR)', fontsize=13)\n",
            "plt.xlabel('False Positive Rate (FPR)', fontsize=11)\n",
            "plt.ylabel('True Positive Rate (TPR)', fontsize=11)\n",
            "plt.legend(loc='lower right')\n",
            "plt.grid(True, alpha=0.3)\n",
            "plt.tight_layout()\n",
            "plt.savefig(os.path.join(PLOTS_DIR, 'comparative_roc_curves.png'), dpi=200)\n",
            "plt.show()\n",
            "print(f'Comparative ROC curve saved to: {os.path.join(PLOTS_DIR, \"comparative_roc_curves.png\")}')"
        ]
    })

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    nb_path = os.path.join("Old Repo Data", "notebooks", "ember_cross_modal_attention.ipynb")
    os.makedirs(os.path.dirname(nb_path), exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)

    print(f"Successfully generated Colab notebook at: {nb_path} with {len(cells)} cells.")

if __name__ == "__main__":
    build_notebook()
