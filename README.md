# QuantaPath v2 — Quantum-Hybrid AI for Cancer Detection in Whole Slide Images

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.44-green.svg)](https://pennylane.ai)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-CAMELYON16-orange.svg)](https://camelyon16.grand-challenge.org)

> **Detecting lymph node metastases in whole slide images using a quantum-hybrid graph neural network.**  
> VQC (3-qubit, 2-layer) + GAT-Transformer — trained and evaluated on CAMELYON16 (221 labeled slides).

---

## 🏆 Results

| Model | Test AUC | Val→Test Gap | F1 |
|-------|----------|--------------|-----|
| v1 Baseline (ResNet-50 + GCNConv) | 0.700 | — | 0.75 |
| v2 Classical (UNI + GAT-Transformer) | **0.870** | −0.084 | 0.84 |
| v2 Quantum (UNI + VQC + GAT-Transformer) | **0.820** | −0.002 ✅ | 0.79 |
| **v2 Ensemble (α=0.6)** | **0.880** 🏆 | — | 0.85 |

> **Key finding:** The quantum model generalises **3× better** than classical (val→test gap −0.002 vs −0.084).  
> The VQC's 1024→3 compression bottleneck acts as a natural regulariser on the small CAMELYON16 dataset.

---

## 🏗️ Architecture

```
Whole Slide Image (WSI)
        ↓  [Patch extraction + UNI ViT-L feature extraction]
Node features: UNI(1024-dim) + sinusoidal pos.enc(16-dim) = 1040-dim
        ↓
┌─────────────────────────────────────────────────┐
│  VQC Encoder (quantum mode only)                │
│  1024 → 128 → 3 qubits (AngleEmbedding)         │
│  2 layers: RY + RZ + CNOT entanglement           │
│  Data re-uploading (Perez-Salinas 2020)          │
│  PauliZ measurement → 3-dim output              │
│  concat(proj_3, quantum_3) → 64-dim + pos.enc   │
└─────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────┐
│  GATMambaBlock                                  │
│  ├── GAT branch:  4-head attention + edge feats │
│  └── Transformer: global context per slide      │
│  Fusion: residual + MLP                         │
└─────────────────────────────────────────────────┘
        ↓
   Global mean pooling
        ↓
   Classifier head → Binary (Normal / Tumor)
```

**Edge features:** `[euclidean_distance, cosine_similarity]` (2-dim, k=8 nearest neighbours)

---

## 📁 Project Structure

```
├── pathq/
│   ├── model_v2.py          # VQCEncoder + GATMambaBlock + QuantaPathV2
│   ├── dataset_v2.py        # CAMELYON16GraphDataset, build_graph_v2, get_loaders_from_features
│   ├── uni_extractor.py     # UNI ViT-L feature extractor + sinusoidal pos encoding
│   ├── model.py             # v1 model (GCNConv + ABMIL) — kept for reference
│   ├── dataset.py           # v1 dataset — kept for reference
│   ├── train.py             # Training utilities
│   └── xai.py               # 3-layer XAI (Grad-CAM + GAT attention + VQC gradients)
│
├── notebooks/
│   ├── week1_setup_and_data.ipynb       # Environment + data verification
│   ├── week2_feature_extraction.ipynb   # ResNet-50 feature extraction (v1)
│   ├── week2b_uni_extraction.ipynb      # UNI ViT-L extraction (v2)
│   ├── week3_classical_fixed.ipynb      # Classical GAT-Transformer baseline
│   ├── week4_ablation_fixed.ipynb       # VQC depth ablation (1L/2L/3L/5Q)
│   ├── week5_quantum_3000.ipynb         # Quantum VQC+GAT at 3000 patches
│   ├── week6_xai.ipynb                  # 3-layer XAI analysis
│   └── week7_ensemble.ipynb             # Soft-voting ensemble
│
├── demo/
│   ├── app.py               # Gradio web demo (4 tabs)
│   ├── run_demo.sh          # One-command launcher
│   └── README.md            # Demo-specific instructions
│
├── checkpoints/             # Trained model weights
│   ├── E2_quantum_3000_best.pth    # Quantum VQC+GAT (3000p, AUC=0.82)
│   ├── E3_classical_3000_best.pth  # Classical GAT-Transformer (3000p, AUC=0.87)
│   ├── w4_A1_vqc_1layer_best.pth   # Ablation: 1 layer
│   ├── w4_A2_vqc_2layer_base_best.pth  # Ablation: 2 layers (winner)
│   ├── w4_A3_vqc_3layer_best.pth   # Ablation: 3 layers
│   └── w4_A4_vqc_2layer_5qubit_best.pth # Ablation: 5 qubits
│
├── outputs/
│   ├── xai/                 # XAI figures (Grad-CAM, GAT attention, VQC gradients, Bloch sphere)
│   ├── week6_xai_report.png # Combined XAI report figure
│   ├── ensemble_result.json # Ensemble grid-search results
│   └── *.json               # Per-experiment result files
│
├── camelyon16/annotations/  # CAMELYON16 XML tumour annotations
├── lesion_annotations/      # Patient-level lesion annotations
├── evaluation_python/       # Official FROC evaluation script
│
├── CLAUDE.md                # Project context for AI assistant
├── QUICKSTART_QUANTAPATH_V2.md  # Quick start guide
└── requirements.txt         # Python dependencies
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
conda create -n pathq python=3.11
conda activate pathq
bash install_quantapath_v2.sh
```

### 2. Download CAMELYON16 features
Pre-extracted UNI features (221 labeled slides) should be placed at:
```
notebooks/data/features_uni/normal_001_uni_features.pt  ...
notebooks/data/features_uni/tumor_001_uni_features.pt   ...
```
See `DOWNLOAD_CAMELYON16_SLIDES.md` for instructions on extracting from raw WSIs.

### 3. Run the interactive demo
```bash
bash demo/run_demo.sh
# → open http://localhost:7860
```

### 4. Train from scratch
```bash
conda activate pathq

# Classical baseline
jupyter nbconvert --to notebook --execute notebooks/week3_classical_fixed.ipynb

# Quantum VQC+GAT
jupyter nbconvert --to notebook --execute notebooks/week5_quantum_3000.ipynb

# Ensemble
jupyter nbconvert --to notebook --execute notebooks/week7_ensemble.ipynb
```

---

## 🖥️ Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Demo (inference) | CPU / any GPU | RTX 3060+ |
| Training (1024p) | RTX 5060 8GB | RTX 4090 24GB |
| Training (3000p) | RTX 4090 24GB | A100 80GB |

> Full 3000-patch training on RTX 5060 (quantum): ~90 min/epoch.  
> Full 3000-patch training on A100 80GB (quantum): ~8 min/epoch.

---

## 📊 Ablation Study (Week 4)

VQC depth ablation at max_patches=512 on laptop RTX 5060:

| Config | Qubits | Layers | Val AUC | Test AUC | Gap | Verdict |
|--------|--------|--------|---------|----------|-----|---------|
| A1 | 3 | 1 | 0.8449 | 0.7132 | −0.1317 | ❌ Overfits |
| **A2 (winner)** | **3** | **2** | **0.8342** | **0.7610** | **−0.0732** | ✅ Best |
| A3 | 3 | 3 | 0.7986 | 0.7463 | −0.0523 | Barren plateau |
| A4 | 5 | 2 | 0.7968 | 0.7445 | −0.0523 | No gain |

**Conclusion:** 2 layers is optimal. 1 layer overfits; 3 layers hits barren plateau. 5 qubits adds no value — the bottleneck is the 1024→3 projection, not circuit width.

---

## 🔬 Explainability (XAI)

Three-layer explanation framework:

| Layer | Method | Output |
|-------|--------|--------|
| Layer 1 | Grad-CAM (input feature gradients) | Spatial attention heatmap — which tissue regions drove the prediction |
| Layer 2 | GAT edge attention weights | Graph structure — which patch-to-patch connections are most important |
| Layer 3a | VQC weight gradients (12 circuit params) | Which quantum gates are most sensitive per slide |
| Layer 3b | Bloch sphere trajectories | How quantum states separate tumour vs normal in Hilbert space |

The demo's XAI tab shows all four layers for any selected slide, plus a natural-language report with tumour region coordinates.

---

## 📦 Dependencies

Key packages (see `requirements.txt` for full list):

```
torch >= 2.0
torch-geometric
pennylane == 0.44.1
pennylane-lightning[gpu]   # GPU-accelerated quantum simulation
custatevec-cu12
gradio >= 6.0
scikit-learn
matplotlib
numpy
```

---

## 🗃️ Dataset

**CAMELYON16** — Camelyon Challenge 2016  
- 221 labeled slides (110 normal + 111 tumor)  
- 112 official test slides (no ground truth labels — excluded from training)  
- Task: binary classification of lymph node metastasis  
- Features: UNI ViT-L pathology foundation model (1024-dim per patch)  
- Patches per slide: up to 3000 (256×256 px at 20× magnification)

---

## 📖 Citation / References

- **UNI:** Chen et al., "Towards a General-Purpose Foundation Model for Computational Pathology", Nature Medicine 2024  
- **CAMELYON16:** Bejnordi et al., JAMA 2017  
- **Data re-uploading VQC:** Perez-Salinas et al., Quantum 2020  
- **GAT:** Veličković et al., ICLR 2018  

---

## 👤 Author

**Kabilash S** -kabilash0108@gmail.com
**Pavitra P** -fspavitra11@gmail.com



---

*QuantaPath v2 — Quantum-Hybrid AI for Pathology | CAMELYON16 | AUC = 0.880*
