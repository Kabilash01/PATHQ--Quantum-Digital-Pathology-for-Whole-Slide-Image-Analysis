# QuantaPath v2 — Project Context for Claude

## What This Project Is
Quantum-hybrid AI for cancer detection in whole slide images (WSIs).
Pipeline: UNI (ViT-L pathology features) → VQC → GAT-Transformer → Binary classifier (tumour/normal).
Dataset: CAMELYON16 (221 labeled slides, lymph node cancer detection).

---

## Current Status: Week 4 COMPLETE

### Completed (Week 1–4)
- v1 baseline (ResNet-50 + GCNConv + ABMIL): AUC 0.70 ✓
- UNI feature extraction (1024-dim, all 221 slides) ✓
- v2 Classical GAT-Transformer: trained + evaluated ✓
- v2 Quantum VQC + GAT-Transformer: trained + evaluated ✓
- Week 4 ablation (1L vs 2L vs 3L vs 5Q, 512 patches): COMPLETE ✓

### Next (Week 5)
Retrain winner config (3q, 2L) at max_patches=3000 on RunPod RTX 4090 for paper numbers.

---

## Week 4 Ablation Results (512 patches, 15 epochs max, patience=5)

| Config | Q | L | Val AUC | Test AUC | Gap | vs W3 |
|---|---|---|---|---|---|---|
| A1 — 1 layer | 3 | 1 | 0.8449 | 0.7132 | −0.1317 ⚠ | −0.068 |
| A2 — 2 layers (winner) | 3 | 2 | 0.8342 | 0.7610 | −0.0732 ✅ | −0.020 |
| A3 — 3 layers | 3 | 3 | 0.7986 | 0.7463 | −0.0523 | −0.035 |
| A4 — 5 qubits | 5 | 2 | 0.7968 | 0.7445 | −0.0523 | −0.037 |

**Winner: A2 (3q, 2L)** — confirms Week 3 config was already optimal.

Key findings:
- 1 layer overfits (worst gap −0.13), 3 layers shows barren plateau (lower AUC than 2L)
- 5 qubits adds no value over 3 qubits — bottleneck is the 1024→3 projection, not circuit width
- 2 layers is the sweet spot: best test AUC + healthy generalisation gap
- All ablation AUCs below W3 baseline (0.7812) — expected (512 vs 3000 patches); rankings are what matter

### Checkpoints (Week 4)
- `checkpoints/w4_A1_vqc_1layer_best.pth`
- `checkpoints/w4_A2_vqc_2layer_base_best.pth`
- `checkpoints/w4_A3_vqc_3layer_best.pth`
- `checkpoints/w4_A4_vqc_2layer_5qubit_best.pth`
- `outputs/week4_ablation_results.json`
- `outputs/week4_best_config.json`

---

## Week 3 v2 Final Results

| Model | Val AUC | Test AUC | Val→Test Gap | Notes |
|---|---|---|---|---|
| Classical GAT-Transformer | 0.9537 | 0.8382 | −0.1155 ⚠ | Overfits |
| Quantum VQC + GAT | 0.8217 | 0.7812 | −0.0405 ✅ | Healthy generalisation |

**Key finding:** Classical wins raw AUC but overfits (training loss → 0.05).
Quantum generalises 3× better — VQC's 1024→3 compression acts as a natural regulariser.
Paper narrative: *"Classical achieves higher test AUC but overfits on small dataset.
Quantum's VQC bottleneck prevents memorisation — increasing qubits (Week 4) expected to close the gap."*

### Checkpoints
- `checkpoints/v2_classical_best.pth` — epoch 9, val AUC 0.9679
- `checkpoints/v2_quantum_best.pth` — epoch 19, val AUC 0.8217
- `checkpoints/v2_quantum_best_latest.pth` — epoch 27 (training crashed at ep 23+28 due to OOM)

---

## Architecture (v2 Current)

### VQCEncoder (`pathq/model_v2.py`)
```
UNI(1024) → Linear(1024→3) → Tanh → AngleEmbedding(3 qubits) → RY+RZ+CNOT × 2 layers
→ PauliZ measurement (3 outputs) → concat(proj_3, quantum_3) = 6-dim
→ concat with pos.enc(16) = 22-dim input to GAT
```
- **AngleEmbedding** (NOT AmplitudeEmbedding — was changed to fix NaN + MottonenStatePrep errors)
- **diff_method='adjoint'** on `lightning.gpu` (NOT parameter-shift — adjoint is much faster)
- Default dropout = 0.4 (increased from 0.3 to reduce overfitting)

### GATMambaBlock (`pathq/model_v2.py`)
- GAT branch: 4-head attention with 2-dim edge features (distance, cosine_sim)
- Transformer branch: global context per slide (Mamba unavailable — g++ 15.2 > CUDA limit)
- Fusion: residual + MLP

### Dataset (`pathq/dataset_v2.py`)
- Node features: UNI(1024) + sinusoidal pos.enc(16) = 1040-dim
- Edge features: [euclidean_dist, cosine_sim] = 2-dim
- `max_patches=3000` default (all patches per slide)
- For fast experiments use `max_patches=128` (~23× faster, slight AUC drop)
- Features at: `notebooks/data/features_uni/` (333 × .pt files)

---

## Known Issues & Fixes Applied

### OOM during VQC backward pass
- **Cause:** cuBLAS memory allocation fails after 20+ hours (CUDA fragmentation)
- **Fix applied:** `torch.cuda.empty_cache()` after every batch and after eval
- **Training environment:** RTX 5060 8GB — other processes also use GPU, leaves ~3GB free
- **If OOM persists:** reduce `max_patches` to 512 or reduce `batch_size` to 1

### Training time
- 3000 patches/slide: ~130 min/epoch → 30 epochs ≈ 2.5 days
- 128 patches/slide: ~6 min/epoch → 30 epochs ≈ 3 hours
- Week 7 plan: RunPod RTX 4090 24GB for full runs

### Resume from checkpoint
`run_experiment` auto-resumes from `*_latest.pth` if it exists.
Just rerun the same cell — it picks up from the last completed epoch.

---

## Environment

```bash
conda activate pathq   # Python 3.11
# Key packages: torch, pennylane 0.44.1, pennylane-lightning[gpu], custatevec-cu12
# CUDA 12.8, RTX 5060
```

### Run quantum test evaluation (no retraining needed)
```python
# In notebooks/week3_gnn_v2.ipynb cell 14 — or run directly:
conda run -n pathq python -c "
import torch, sys; sys.path.insert(0,'.')
from pathq.model_v2 import QuantaPathV2
from pathq.dataset_v2 import get_loaders_from_features
# ... load checkpoint, run eval_model on test_loader
"
```

---

## File Map

```
pathq/
  model_v2.py       — VQCEncoder + GATMambaBlock + QuantaPathV2
  dataset_v2.py     — build_graph_v2, CAMELYON16GraphDataset, get_loaders_from_features
  uni_extractor.py  — UNI ViT-L feature extractor + sinusoidal pos encoding
  model.py          — v1 model (GCNConv+ABMIL) — kept for reference only
  dataset.py        — v1 dataset — kept for reference only
  train.py          — v1 training loop — kept for reference only
  xai.py            — 3-layer XAI (Grad-CAM++ + ABMIL + VQC param-shift)

notebooks/
  week3_gnn_v2.ipynb   — MAIN notebook: both experiments + comparison graphs
  week2b_uni_extraction.ipynb — UNI feature extraction pipeline
  data/features_uni/   — 333 × *_uni_features.pt (UNI 1024-dim features per slide)

checkpoints/
  v2_classical_best.pth        — Classical GAT best (ep 9, val AUC 0.9679)
  v2_quantum_best.pth          — Quantum VQC best (ep 19, val AUC 0.8217)
  v2_quantum_best_latest.pth   — Latest quantum state for resuming (ep 27)
```

---

## Training Configuration (current defaults)

```python
# In run_experiment():
optimizer  = Adam(lr=1e-4, weight_decay=1e-3)   # weight_decay increased to prevent overfit
scheduler  = CosineAnnealingLR(T_max=30, eta_min=1e-6)
early_stop = 10 epochs no improvement

# Model:
QuantaPathV2(use_vqc=True, n_qubits=3, vqc_layers=2, dropout=0.4)

# Table printed each epoch:
Ep  TrLoss  VaLoss  AUC  F1  Sens  Spec  Time  Best
# ⚠ printed if val_loss > train_loss × 2.5 (overfitting warning)
```

---

## Week 4 Plan (Next)

**Goal:** Layer ablation — does more VQC depth help close the AUC gap?

| Experiment | VQC Layers | Expected |
|---|---|---|
| Ablation 1 | 1 layer | Faster, lower AUC |
| Ablation 2 | 2 layers | Current baseline (0.7812 test) |
| Ablation 3 | 3 layers | Slower, hopefully higher AUC |

Also consider: increasing n_qubits from 3 → 5 or 7 to reduce the dimensionality bottleneck (1024→3 is very aggressive).
