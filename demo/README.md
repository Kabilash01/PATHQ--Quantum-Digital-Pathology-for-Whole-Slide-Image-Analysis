# QuantaPath v2 — Lab Exam Demo

Interactive Gradio web app for showcasing the quantum-hybrid cancer detection system.

---

## Quick Start

```bash
# From project root
bash demo/run_demo.sh
```

Then open **http://localhost:7860** in your browser.

---

## Tabs

| Tab | Description | Time |
|-----|-------------|------|
| 🔬 Live Prediction | Pick any slide → Quantum model → probability bars + attention heatmap | ~5s |
| 📊 Full Evaluation | Both models on test set → ROC curves + confusion matrices + metrics table | ~60s |
| ⚛️ Quantum XAI | VQC gradient sensitivity (12 circuit weights) + quantum output distribution | ~30s |
| 📋 Results Summary | Static table — all results including RunPod A100 numbers | instant |

---

## Checkpoints Used

The app automatically selects the best available checkpoint:

| Priority | Quantum | Classical |
|----------|---------|-----------|
| 1st (best) | `E2_quantum_3000_best.pth` | `E3_classical_3000_best.pth` |
| 2nd | `E2_quantum_best.pth` | `E3_classical_best.pth` |
| 3rd (fallback) | `v2_quantum_best.pth` | `v2_classical_best.pth` |

---

## Key Results (RunPod A100, 3000 patches/slide)

| Model | Test AUC | Val→Test Gap |
|-------|----------|--------------|
| Classical GAT-Transformer | **0.870** | −0.084 |
| Quantum VQC+GAT (3q, 2L) | **0.820** | −0.002 ✅ |
| Ensemble (α=0.6) | **0.880** 🏆 | — |

---

## Architecture

```
UNI ViT-L (1024-dim pathology features)
    ↓
VQC Encoder: 1024 → 128 → 3 qubits (AngleEmbedding, 2 layers, data re-uploading)
    ↓
PauliZ measurement → concat(proj, quantum) → 64-dim
    ↓
+ positional encoding (16-dim) → 80-dim
    ↓
Linear projection → 256-dim
    ↓
GATMambaBlock: GAT (local) + Transformer (global context per slide)
    ↓
Global mean pooling → Classifier head → Binary (Normal/Tumor)
```

---

## Troubleshooting

**Port in use:**
```bash
# Change port in app.py: server_port=7861
```

**OOM on prediction:**
```python
# In app.py, reduce max_patches in _build_graph_from_pt calls (default 512)
```

**Models load slowly:**
The first run loads both models into memory (~60s). Subsequent inferences are fast (<5s).
