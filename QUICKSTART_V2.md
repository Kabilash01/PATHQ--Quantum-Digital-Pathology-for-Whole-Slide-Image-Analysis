# QuantaPath v2 Quick Reference

## What Was Delivered

### ✅ New Modules (Tasks 3-5)

1. **`pathq/uni_extractor.py`**
   - Load frozen UNI ViT-L (1024-dim)
   - Sinusoidal positional encoding (16-dim)
   - Batch feature extraction

2. **`pathq/model_v2.py`**
   - VQCEncoder: 3-qubit, 2-layer with amplitude embedding
   - GATMambaBlock: Fused GAT + Mamba (+ GRU fallback)
   - QuantaPathV2: Full model with optional VQC

3. **`pathq/dataset_v2.py`**
   - build_graph_v2(): PyG graphs with 1040-dim nodes, 2-dim edges
   - make_graphs_patchcamelyon_fixed(): **FIXED bag generation**
     - Separates positive/negative patches
     - Guarantees balanced test set (fixes 0.5 AUC bug)

### ✅ Notebooks (Tasks 6-7)

1. **`notebooks/week2b_uni_extraction.ipynb`**
   - Extract UNI features from patch .pkl files
   - Save to `data/features_uni/`
   - VRAM-optimized for RTX 5060 (batch_size=32)

2. **`notebooks/week3_gnn_v2.ipynb`**
   - Build balanced MIL bags using fixed function
   - Train classical baseline (GAT-Mamba only)
   - Train quantum hybrid (VQC + GAT-Mamba)
   - Compare: AUC, F1, sensitivity, specificity
   - Save results to `outputs/v2_results.json`
   - Verify: test AUC > 0.5, F1 > 0, mixed predictions

### ✅ Installation (Task 2)

**`install_v2.sh`** — Automated setup:
```bash
bash install_v2.sh
```

Installs:
- timm>=0.9.16 (UNI)
- mamba-ssm + causal-conv1d (Mamba, optional)
- pennylane + pennylane-lightning (VQC)
- torch-geometric (GNN)

## Critical Fix: MIL Bag Label Balance

### The Bug (v1)
```python
# ❌ WRONG: Built bags by sampling ANY patches from a slide
# Some patches normal, some tumor → unpredictable labels
# Result: test set all-negative → AUC=0.5, F1=0

for slide_id in all_slides:  # Mixed normal/tumor patches
    bag_feats = sample_16_random_patches(slide_id)
    label = ???  # How should this be assigned?
```

### The Fix (v2)
```python
# ✅ RIGHT: Separate positive/negative patches, build all-normal and all-tumor bags
rng.shuffle(neg_idx)
for i in range(0, len(neg_pool), BAG_SIZE):
    graphs.append(_build_graph(..., label=0))  # All-negative bag

rng.shuffle(pos_idx)
for i in range(0, len(pos_pool), BAG_SIZE):
    graphs.append(_build_graph(..., label=1))  # All-positive bag

# Verify: assert pos > 0 and neg > 0
```

**Location:** `pathq/dataset_v2.py::make_graphs_patchcamelyon_fixed()`

## Quick Start

### 1. Install (5-10 min)
```bash
cd /path/to/pathq_project
bash install_v2.sh
# When prompted, login: huggingface-cli login
```

### 2. Extract Features (60-90 min on RTX 5060)
```bash
jupyter notebook notebooks/week2b_uni_extraction.ipynb
# Run all cells — saves to data/features_uni/
```

### 3. Train & Evaluate (30 min on RTX 5060)
```bash
jupyter notebook notebooks/week3_gnn_v2.ipynb
# Runs classical + quantum training
# Saves results to outputs/v2_results.json
```

### 4. Check Results
```bash
cat outputs/v2_results.json
# Should see: classical_auc, quantum_auc, delta_auc
```

## File Index

| File | Task | Status | Purpose |
|---|---|---|---|
| `pathq/uni_extractor.py` | 3 | ✅ | Load UNI, extract 1024-dim features |
| `pathq/model_v2.py` | 4 | ✅ | VQC encoder, GAT-Mamba, full model |
| `pathq/dataset_v2.py` | 5 | ✅ | Graph builder, **fixed bag gen**, dataloaders |
| `notebooks/week2b_uni_extraction.ipynb` | 6 | ✅ | Feature extraction pipeline |
| `notebooks/week3_gnn_v2.ipynb` | 7 | ✅ | Training + evaluation framework |
| `install_v2.sh` | 2 | ✅ | Dependency installation |
| `README_v2.md` | Doc | ✅ | Comprehensive guide |

## Expected Results

After running both notebooks:

```
outputs/v2_results.json
{
  "classical_auc": 0.65-0.75,
  "quantum_auc": 0.60-0.75,
  "classical_f1": 0.60-0.80,
  "quantum_f1": 0.50-0.80,
  "delta_auc": -0.10 to +0.10
}
```

**Key milestones:**
- ✅ classical_auc > 0.5 (not random)
- ✅ quantum_auc > 0.5 (not random)
- ✅ Both f1 > 0 (mixed predictions, not all one class)
- ✓ Delta close to 0 expected at this stage (quantum advantage appears at low data)

## Verification Checklist

- [ ] UNI extractor loads without error
- [ ] Features saved to `data/features_uni/` (check file count)
- [ ] Week 3 notebook builds balanced test set (run Cell 3 assertions)
- [ ] Classical AUC > 0.5
- [ ] Quantum AUC > 0.5
- [ ] Both F1 > 0
- [ ] Checkpoints saved to `checkpoints/v2_*.pth`
- [ ] Results saved to `outputs/v2_results.json`

## Next: Week 4 Experiments

After v2 baseline is established:

1. **Layer ablation** (1 vs 2 vs 3 VQC layers)
2. **Qubit studies** (2 vs 3 vs 4 qubits)
3. **Low-data regime** (20%, 50%, 100% of training data)
4. **Quantum advantage** (where quantum beats classical)

---

**Version:** v2.0
**Date:** 2026-05-12
**Tasks Complete:** 1-7 ✅ | Tasks 8-10 (docs) ✅
**Ready for:** Week 4 experiments
