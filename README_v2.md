# QuantaPath v2 Implementation Guide

## Overview

QuantaPath v2 is a complete architectural upgrade from the old system:

| Component | v1 (Old) | v2 (New) | Improvement |
|---|---|---|---|
| **Backbone** | ResNet-50 (ImageNet, 512-dim) | UNI (H&E pathology, 1024-dim) | Domain-specific, 2x feature dim |
| **Position Encoding** | None | Sinusoidal (16-dim) | Spatial awareness |
| **Graph Conv** | GCNConv | GAT (Graph Attention) | Learned attention on edges |
| **Global Aggregation** | ABMIL | Mamba (with GRU fallback) | Efficient sequence modeling |
| **VQC Input** | 2048-dim ResNet | 1024-dim UNI | Smaller, more tractable |
| **VQC Architecture** | 2-qubit, 1-layer | 3-qubit, 2-layer | More expressive circuits |

## Installation (Task 2)

```bash
# 1. Navigate to project root
cd /path/to/pathq_project

# 2. Activate conda environment
conda activate pathq

# 3. Run installation script
bash install_v2.sh

# 4. Login to HuggingFace (when prompted)
huggingface-cli login
# Paste token from https://huggingface.co/settings/tokens
```

**Key dependencies:**
- `timm>=0.9.16` — UNI model architecture
- `mamba-ssm` (optional) — fast sequence modeling, falls back to GRU if unavailable
- `pennylane-lightning` — 10x faster VQC simulation
- `torch-geometric` — graph neural networks
- `huggingface_hub` — download UNI model weights

## Architecture (Tasks 3-5)

### New Modules Created

1. **`pathq/uni_extractor.py`** (Task 3)
   - `build_uni_extractor(device)` → loads frozen UNI ViT-L
   - `sinusoidal_pos_encoding(coords)` → spatial positional encoding
   - `extract_uni_features(patches, coords, uni_model, transform)` → batch extraction

2. **`pathq/model_v2.py`** (Task 4)
   - `VQCEncoder` — 3-qubit VQC with amplitude embedding
   - `GATMambaBlock` — fused GAT + Mamba layer (with GRU fallback)
   - `QuantaPathV2` — full model with optional VQC

3. **`pathq/dataset_v2.py`** (Task 5)
   - `build_graph_v2()` — build PyG graphs with:
     - Node features: UNI(1024) + pos.enc(16) = 1040-dim
     - Edge features: [distance, cosine_similarity] = 2-dim
   - `make_graphs_patchcamelyon_fixed()` — **FIXED bag generation**
     - Separates positive/negative patches
     - Guarantees balanced bags (no all-negative test sets)
     - Asserts label balance for safety

### Data Flow

```
Input patches (PIL Images)
    ↓
UNI ViT-L (frozen)
    ↓
Features: (N, 1024)
    ↓
+ Sinusoidal pos.enc: (N, 16)
    ↓
Full node features: (N, 1040)
    ↓
VQC (optional): 1024 → 8-dim proj + 3-dim quantum = 11-dim hybrid
    ↓
Concat with pos.enc: (N, 27) or (N, 1040)
    ↓
Input projection to hidden=256
    ↓
GATMambaBlock: local attention (GAT) + global context (Mamba)
    ↓
Global mean pooling over patch nodes
    ↓
Classification head
    ↓
Binary logits (2 classes)
```

## Workflow (Tasks 6-7)

### Step 1: Extract UNI Features (Task 6)

Run notebook: `notebooks/week2b_uni_extraction.ipynb`

**What it does:**
- Loads patch .pkl files from `data/patches/`
- Extracts 1024-dim features using frozen UNI ViT-L
- Saves to `data/features_uni/{slide_id}_uni_features.pt`
- Each file contains: `{'features': (N, 1024), 'coords': (N, 2)}`

**Time estimate:**
- ~1 hour per 100 slides on RTX 5060 (8GB VRAM)
- Uses batch_size=32 (smaller than ResNet-50 due to ViT-L size)

**VRAM management:**
```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

# After extraction, free UNI from GPU:
del uni_model
torch.cuda.empty_cache()
import gc; gc.collect()
```

### Step 2: Train Models (Task 7)

Run notebook: `notebooks/week3_gnn_v2.ipynb`

**What it does:**

1. **Build MIL bags** using `make_graphs_patchcamelyon_fixed()`:
   - ✅ **FIXED:** Separates pos/neg patches — guarantees balanced test set
   - Creates equal numbers of all-normal and all-tumor bags
   - Each bag: 16 patches in 4×4 grid, K-NN edges (k=5)

2. **Train classical baseline** (no VQC):
   - Input: 1040-dim (UNI + pos.enc)
   - Architecture: ProjectionLayer → GATMambaBlock → ClassHead
   - Baseline for comparison

3. **Train quantum hybrid** (with VQC):
   - Input to VQC: 1024-dim UNI features
   - VQC: 3-qubit, 2-layer, amplitude encoding
   - Output: hybrid 27-dim (proj_8 + quantum_3 + pos.enc_16)
   - Architecture: same as above with VQC preprocessing

4. **Compare results:**
   - Saves to `outputs/v2_results.json`
   - Reports: AUC, F1, sensitivity, specificity
   - Asserts test AUC > 0.5 (catches broken models)

**Time estimate:**
- ~15 min per model on RTX 5060
- Early stopping at 10 epochs without improvement

**Key fix in this notebook:**

```python
# WRONG — creates all-positive test bags
for slide_key in all_slides:  # Mixed normal + tumor
    bags.append(...)

# RIGHT — separate pools, guaranteed balance
rng.shuffle(neg_idx)
for i in range(0, len(neg_pool), BAG_SIZE):
    graphs.append(_build_one_graph(..., label=0))  # all negatives

rng.shuffle(pos_idx)
for i in range(0, len(pos_pool), BAG_SIZE):
    graphs.append(_build_one_graph(..., label=1))  # all positives

# Verify
assert pos > 0 and neg > 0, "STILL BROKEN"
```

## Verification & Sanity Checks (Task 10)

After completing each task, verify with these checks:

### After Task 3 (UNI Extractor)
```python
from pathq.uni_extractor import build_uni_extractor
uni, tfm = build_uni_extractor(torch.device('cuda'))
x = torch.randn(2, 3, 224, 224).cuda()
with torch.no_grad(): out = uni(x)
assert out.shape == (2, 1024), f"Expected (2,1024), got {out.shape}"
print("✅ UNI OK")
```

### After Task 4 (Model)
```python
from pathq.model_v2 import QuantaPathV2
m = QuantaPathV2(use_vqc=False)
print("✅ Model instantiated")

# Can also run if __name__ == '__main__' block:
python -c "from pathq.model_v2 import *"
```

### After Task 5 (Graph Builder)
```python
from pathq.dataset_v2 import build_graph_v2
g = build_graph_v2(
    features=torch.randn(20, 1024),
    coords=torch.randint(0,10,(20,2)).float(),
    label=1, k=8
)
assert g.x.shape == (20, 1040), f"Expected (20,1040), got {g.x.shape}"
assert g.edge_attr.shape[1] == 2, "edge_attr should have 2 features"
print("✅ Graph builder OK")
```

### After Task 6 (UNI Extraction)
```python
from pathlib import Path
fp = list(Path('./data/features_uni').glob('*.pt'))[0]
d = torch.load(fp, weights_only=False)
assert d['features'].shape[1] == 1024
print("✅ UNI features OK")
```

### After Task 7 (Training)
```python
# Notebook checks:
assert res_c['auc'] > 0.5, f"TEST AUC {res_c['auc']:.4f} <= 0.5 — BROKEN"
assert res_q['auc'] > 0.5, f"TEST AUC {res_q['auc']:.4f} <= 0.5 — BROKEN"
assert res_c['f1'] > 0.0, "All predictions one class"
assert res_q['f1'] > 0.0, "All predictions one class"
print("✅ All checks passed")
```

## VRAM Management (Task 8)

**RTX 5060 8GB safety guidelines:**

| Operation | Batch Size | Notes |
|---|---|---|
| UNI extraction | 32 | ViT-L is large |
| GNN training | 4 | 4 slides × 16 patches = 64 nodes/batch |
| VQC training | 4 | VQC runs on CPU but takes time |

**VRAM monitoring:**
```python
def vram():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved() / 1e9
    print(f'VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved')
```

**If OOM errors:**
1. Reduce batch_size: 4 → 2
2. Clear cache: `torch.cuda.empty_cache()` after loading models
3. Use gradient checkpointing (not yet implemented)

## Dataloader Setup (Task 9)

Once UNI features are extracted to `data/features_uni/`:

```python
from pathq.dataset_v2 import get_loaders

train_loader, val_loader, test_loader = get_loaders(
    features_dir = Path('./data/features_uni'),
    normal_dir   = Path('./data/camelyon16/normal'),
    tumor_dir    = Path('./data/camelyon16/tumor'),
    batch_size   = 4,
    k            = 8,  # K-NN edges
)
```

**Feature file requirements:**
- Name: `{slide_id}_uni_features.pt`
- Contents:
  - `features`: (N, 1024) UNI features
  - `coords`: (N, 2) patch grid coordinates
  - Optional: `slide_id`, `n_patches`

## Expected Project Structure After All Tasks

```
pathq_project/
├── pathq/
│   ├── __init__.py
│   ├── model.py           (old v1 model)
│   ├── model_v2.py        ✨ NEW (Task 4)
│   ├── dataset.py         (old v1 dataset)
│   ├── dataset_v2.py      ✨ NEW (Task 5)
│   ├── uni_extractor.py   ✨ NEW (Task 3)
│   ├── train.py
│   ├── xai.py
│
├── notebooks/
│   ├── week2_feature_extraction.ipynb    (ResNet-50)
│   ├── week2b_uni_extraction.ipynb       ✨ NEW (Task 6)
│   ├── week3_gnn_baseline.ipynb          (old v1)
│   ├── week3_gnn_v2.ipynb                ✨ NEW (Task 7)
│   ├── week4_vqc_prototype.ipynb         (to be updated)
│   ├── week5_hybrid_training.ipynb       (to be updated)
│   ├── week6_xai_complete.ipynb          (to be updated)
│
├── data/
│   ├── patches/               (256×256 PIL images)
│   ├── features/              (ResNet-50 2048-dim, old)
│   ├── features_uni/          ✨ NEW (UNI 1024-dim)
│   ├── camelyon16/
│   │   ├── normal/
│   │   ├── tumor/
│
├── checkpoints/
│   ├── v2_classical_best.pth  ✨ NEW
│   ├── v2_quantum_best.pth    ✨ NEW
│
├── outputs/
│   ├── v2_results.json        ✨ NEW
│
├── install_v2.sh              ✨ NEW (Task 2)
└── README_v2.md               ✨ NEW (this file)
```

## Next Steps After v2 Complete

1. **Week 4 v2**: Expand VQC experiments (layer ablation, qubit studies)
2. **Week 5 v2**: Low-data regime, sample efficiency, quantum advantage
3. **Week 6 v2**: 3-layer XAI (Grad-CAM++ + ABMIL attention + VQC param-shift)

## Troubleshooting

### "UNI model not found / HuggingFace auth error"
```bash
huggingface-cli login  # Paste token
# Accept terms: https://huggingface.co/MahmoodLab/uni
```

### "CUDA out of memory" during extraction
```python
# Reduce batch size in week2b_uni_extraction.ipynb
BATCH_SIZE = 16  # or even 8
```

### "mamba-ssm: No module named mamba_ssm"
This is OK — model has GRU fallback. For speed, later:
```bash
pip install mamba-ssm --no-build-isolation
```

### "Test AUC still 0.5000, F1=0.0"
**This means the bag balance fix did not work.** Check:
```python
# Verify in week3_gnn_v2.ipynb Cell 3:
test_labels = [g.y.item() for g in TEST_G]
pos = sum(test_labels)
neg = len(test_labels) - pos
print(f'TEST_G: {pos} positive, {neg} negative')
assert pos > 0 and neg > 0, "BAGS STILL BROKEN"
```

## References

- **UNI paper**: Pretrained Foundation Model for Pathology (MahmoodLab)
  - https://huggingface.co/MahmoodLab/uni
- **GAT-Mamba**: Graph Attention Networks with Mamba (Ding et al., Sci Rep 2025)
- **PennyLane**: Quantum machine learning in Python
  - https://pennylane.ai/
- **PyTorch Geometric**: Graph neural networks
  - https://pytorch-geometric.readthedocs.io/

## Citation

If you use QuantaPath v2 in your research:

```bibtex
@software{quantapath_v2,
  title={QuantaPath v2: Quantum-Hybrid AI for WSI Cancer Detection},
  author={...},
  year={2026},
  url={https://github.com/...}
}
```

---

**Status:** ✅ Tasks 1-7 Complete | Tasks 8-10 Documented
**Estimated Time:** ~2 hours setup + 2 hours execution (feature extraction + training)
**Next Update:** Week 4 experiments (layer ablation, quantum advantage in low-data regime)
