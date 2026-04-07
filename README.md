# PATHQ — Quantum Digital Pathology

**Quantum-Hybrid AI for Whole Slide Image Analysis**  
*VQC + Graph Neural Network + Three-Layer XAI on Gigapixel Tissue Slides*

---
## What PATHQ Does

PATHQ detects cancer in whole slide tissue biopsy images (WSIs) using a three-layer architecture:

1. **Classical layer** — ResNet-50 extracts 512-dim features from 256×256 patches
2. **Quantum layer** — VQC encodes features into quantum state space for enriched representations  
3. **Graph layer** — GCN + ABMIL aggregates patch-level features to slide-level prediction
4. **XAI layer** — Three-level explanation: Grad-CAM++ + attention weights + quantum circuit sensitivity

**Research novelty:** Zero prior papers combine VQC quantum encoding + WSI-scale GNN + three-layer XAI on pathology data (confirmed gap, 2025 literature review).

---

## Quick Start

### 1. Create Environment

```bash
conda create -n pathq python=3.10 -y
conda activate pathq
conda install -c conda-forge openslide -y
```

### 2. Install PyTorch (RTX 5060 — CUDA 12.1)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install All Dependencies

```bash
pip install torch-geometric
pip install pennylane pennylane-qiskit qiskit qiskit-aer
pip install openslide-python monai torchstain timm
pip install medmnist datasets grad-cam wandb qutip
pip install matplotlib seaborn scikit-learn opencv-python tqdm scipy
```

### 4. Verify Setup

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "import pennylane as qml; print(qml.__version__)"
python -c "import torch_geometric; print('PyG OK')"
```

---

## Repository Structure

```
pathq_project/
├── notebooks/
│   ├── week1_setup_and_data.ipynb      ← START HERE
│   ├── week2_feature_extraction.ipynb
│   ├── week3_gnn_baseline.ipynb        (coming Week 3)
│   ├── week4_vqc_prototype.ipynb       (coming Week 4)
│   ├── week5_hybrid_training.ipynb     (coming Week 5)
│   └── week6_xai_complete.ipynb        (coming Week 6)
├── pathq/
│   ├── model.py        ← Full model: VQCEncoder + GNNEncoder + ABMILAggregator
│   ├── train.py        ← Training loop with wandb logging
│   └── xai.py          ← Three-layer XAI: Grad-CAM++ + attention + param-shift
├── data/
│   ├── patches/        ← Extracted 256×256 patches (.pkl per slide)
│   ├── features/       ← ResNet-50 feature vectors (.pt per slide)
│   └── camelyon16/     ← Raw WSI slides (.tif) — put them here
├── checkpoints/        ← Saved model weights
├── outputs/            ← Figures, heatmaps, XAI reports
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Size | Where | Registration |
|---|---|---|---|
| PathMNIST | 200 MB | `pip install medmnist` | None |
| PatchCamelyon | 7.5 GB | HuggingFace | None |
| CAMELYON16 | ~700 GB (use 50-slide subset) | camelyon16.grand-challenge.org | Free |
| BRACS | 60 GB | bracs-dataset.github.io | Free |

Download in this order. Start with PathMNIST + PatchCamelyon on Day 1.  
Register for CAMELYON16 on Day 3. Don't download all 700 GB — start with 50 slides.

---

## Training

```bash
# Classical GNN baseline (no VQC)
python pathq/train.py --mode classical --features_dir ./data/features --epochs 50

# Quantum hybrid (with VQC)
python pathq/train.py --mode quantum --features_dir ./data/features --n_qubits 3 --vqc_layers 2

# Quick test run (10 slides, 5 epochs)
python pathq/train.py --mode classical --max_slides 10 --epochs 5 --no_wandb
```

---

## 8-Week Timeline

| Week | Goal | Hardware |
|---|---|---|
| 1 | Environment + data pipeline | Laptop |
| 2 | Feature extraction + graph building | Laptop |
| 3 | Classical GNN baseline (AUC > 0.90 on PCam) | Laptop |
| 4 | VQC design + PennyLane basics | Laptop (CPU sim) |
| 5 | Hybrid VQC+GNN training + ablations | Laptop |
| 6 | All three XAI layers implemented | Laptop |
| 7 | Full CAMELYON16 training on RunPod | RunPod RTX 4090 |
| 8 | Final evaluation + inference script | Laptop |

---

## Hardware Notes (RTX 5060 8GB)

- **Weeks 1–6:** Everything runs on your laptop. Pre-extract features to keep VRAM under 6 GB.
- **Week 7:** Rent RunPod RTX 4090 (24 GB) for 2 sessions (~$5 total). Upload pre-extracted `.pt` files.
- **VQC:** Runs on CPU via `qiskit-aer` simulator. No GPU needed for quantum circuits.

---

## Key Results (Targets)

| Model | Dataset | AUC | F1 |
|---|---|---|---|
| Classical GNN (baseline) | PatchCamelyon | > 0.90 | > 0.88 |
| Quantum VQC+GNN | PatchCamelyon | >= Classical | >= Classical |
| Classical GNN | CAMELYON16 (full) | > 0.85 | > 0.80 |
| Quantum VQC+GNN | CAMELYON16 (full) | > 0.85 | > 0.80 |
| Quantum vs Classical (20% labels) | CAMELYON16 | delta > +0.03 | — |

---

## Author

Kabi — kabilash0108@gmail.com
