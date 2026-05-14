# UNI vs ResNet-50 — Performance Comparison Guide

**Goal:** Quantify the improvement of UNI (pathology-specific) over ResNet-50 (ImageNet)

---

## Timeline

### Phase 1: Extract UNI Features (⏱️ 60-90 min)
```bash
cd /home/kabi/PATHQ--Quantum-Digital-Pathology-for-Whole-Slide-Image-Analysis/notebooks
jupyter notebook week2b_uni_extraction.ipynb
```

**What it does:**
- Runs through all 666 patch .pkl files
- Extracts UNI features (1024-dim per patch)
- Saves to `data/features_uni/{slide_id}_uni_features.pt`
- Verifies extraction

**Expected output:**
```
Extracting UNI features (batch_size=32)...
UNI output: 1024-dim features per patch

Extracting UNI features: 100%|████████| 666/666 [85:32<00:00, 7.72s/patch]
✅ Done. Feature files saved to: ../data/features_uni
Total files: 666

Slide: normal_001
Features shape: torch.Size([N, 1024])   # N varies per slide
Coords shape: torch.Size([N, 2])
Feature range: [-2.154, 3.821]
Feature mean: -0.012
Feature std: 0.987

✅ UNI extraction verified
```

**Storage:** ~20GB in `data/features_uni/` (666 files × ~30MB avg)

---

### Phase 2: Train Models on UNI (⏱️ 2-3 hours)
```bash
cd /home/kabi/PATHQ--Quantum-Digital-Pathology-for-Whole-Slide-Image-Analysis/notebooks
jupyter notebook week3_gnn_v2.ipynb
```

**Models trained:**
1. **Classical (GAT-Mamba, no VQC)** — UNI baseline
2. **Quantum (VQC + GAT-Mamba)** — UNI hybrid

**Output files:**
- `checkpoints/v2_classical_best.pth`
- `checkpoints/v2_quantum_best.pth`
- `outputs/v2_results.json`

---

## Comparison: UNI vs ResNet-50

### Previous Results (v1 — ResNet-50, Week 3)

From `notebooks/week3_gnn_baseline.ipynb`:
```
Classical GNN Baseline (ResNet-50 + GCNConv)
Test AUC    : 0.7000
Test F1     : 0.7500
Sensitivity : 0.6850
Specificity : 0.7150
```

### New Results (v2 — UNI, After Phase 2)

Will be in `outputs/v2_results.json`:
```json
{
  "classical_auc": 0.XXXX,    ← Compare to 0.7000
  "quantum_auc": 0.XXXX,
  "classical_f1": 0.XXXX,      ← Compare to 0.7500
  "quantum_f1": 0.XXXX,
  "delta_auc": 0.XXXX
}
```

---

## Key Differences: UNI vs ResNet-50

| Aspect | ResNet-50 (v1) | UNI (v2) |
|--------|---|---|
| **Training data** | ImageNet (generic images) | 100M+ H&E pathology slides |
| **Feature dim** | 2048 | 1024 |
| **Model size** | ResNet50 | ViT-L |
| **GNN backbone** | GCNConv + ABMIL | GAT-Mamba |
| **Positional encoding** | None | Sinusoidal (16-dim) |
| **Edge features** | K-NN only | K-NN + distance + cosine_sim |
| **Dataset** | PatchCamelyon bags | PatchCamelyon bags |

**Expected improvement:** UNI should outperform ResNet-50 because:
1. ✅ Trained on pathology data (H&E tissue) — domain-specific
2. ✅ Better architecture (ViT-L vs ResNet-50)
3. ✅ Better GNN (GAT-Mamba vs GCNConv)
4. ✅ Better graph features (pos.enc + edge features)

---

## After Training: Analysis Tasks

### 1. Check if UNI > ResNet-50

```python
# In jupyter or terminal after week3_gnn_v2.ipynb completes:
import json

with open('outputs/v2_results.json') as f:
    v2 = json.load(f)

# ResNet-50 v1 baseline
v1_classical_auc = 0.70

print(f"v1 (ResNet-50) Classical AUC: {v1_classical_auc:.4f}")
print(f"v2 (UNI)       Classical AUC: {v2['classical_auc']:.4f}")
print(f"Improvement:                   {v2['classical_auc'] - v1_classical_auc:+.4f}")
print()

if v2['classical_auc'] > v1_classical_auc:
    pct = (v2['classical_auc'] - v1_classical_auc) / v1_classical_auc * 100
    print(f"✅ UNI beats ResNet-50 by {pct:+.1f}%")
else:
    print(f"❌ ResNet-50 still better (unexpected)")
```

### 2. Quantum advantage

```python
print(f"\nQuantum advantage:")
print(f"  Classical (UNI): {v2['classical_auc']:.4f}")
print(f"  Quantum (UNI):   {v2['quantum_auc']:.4f}")
print(f"  Delta:           {v2['delta_auc']:+.4f}")
```

### 3. Create comparison table for paper

```python
results_table = f"""
RESULTS: UNI Foundation Model vs ResNet-50 ImageNet

Model                          AUC     F1      Sens    Spec
───────────────────────────────────────────────────────────
ResNet-50 + GCNConv (v1)       0.7000  0.7500  0.6850  0.7150
UNI + GAT-Mamba (v2)           {v2['classical_auc']:.4f}  {v2['classical_f1']:.4f}  ????    ????
UNI + VQC + GAT-Mamba (v2)     {v2['quantum_auc']:.4f}  {v2['quantum_f1']:.4f}  ????    ????

Improvement (Classical):       {v2['classical_auc']-0.7000:+.4f}
Quantum Advantage:             {v2['delta_auc']:+.4f}
"""
print(results_table)
```

---

## Troubleshooting

### ❌ week2b fails with memory error
- Reduce `BATCH_SIZE` from 32 → 16 in Cell 3
- UNI is ViT-L (bigger than ResNet-50)

### ❌ UNI model won't download
- Verify HuggingFace login: `hf auth status`
- Check UNI terms accepted: https://huggingface.co/MahmoodLab/uni

### ❌ week3 training fails on Cell 3 (bag building)
- The fixed `make_graphs_patchcamelyon_fixed()` should handle it
- If it fails, check that TEST_G bags are balanced (pos > 0 AND neg > 0)

---

## Expected Outcomes

**Best case (UNI is significantly better):**
- Classical AUC: 0.72-0.75 (vs 0.70 ResNet-50)
- Quantum AUC: 0.73-0.76 (gains 1-2% from VQC)
- **Publishing angle:** "Pathology-specific foundation model + quantum gates improve WSI analysis"

**Conservative case (UNI comparable):**
- Classical AUC: 0.70-0.71 (similar to ResNet-50)
- Quantum AUC: 0.71-0.72 (quantum gates add value)
- **Publishing angle:** "VQC hybrid architecture shows quantum advantage even with modern backbones"

**Worst case (unexpected):**
- Classical AUC: < 0.70 (unusual)
- Check: Is make_graphs_patchcamelyon_fixed truly balanced? Is test AUC > 0.5?

---

## Next Steps

1. **Run week2b_uni_extraction.ipynb** now (grab coffee ☕, ~90 min)
2. **Run week3_gnn_v2.ipynb** after extraction completes (~2-3 hours)
3. **Post results** with comparison table

Let me know when you hit any errors or milestone completions!

**Current status:** Ready to start Phase 1 ✅
