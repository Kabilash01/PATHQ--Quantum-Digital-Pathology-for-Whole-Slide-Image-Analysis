#!/bin/bash
# Install QuantaPath v2 dependencies (skip Mamba due to g++ incompatibility)

set -e

echo "=================================================="
echo "QuantaPath v2 — Dependency Installation"
echo "=================================================="
echo ""

# Step 1: HuggingFace CLI + Login
echo "[1/5] Installing HuggingFace hub..."
pip install huggingface_hub -q

echo ""
echo "🔐 HuggingFace Login Required"
echo "   1. Go to: https://huggingface.co/settings/tokens"
echo "   2. Create new token (read access)"
echo "   3. Accept UNI model terms: https://huggingface.co/MahmoodLab/uni"
echo ""
read -p "Press Enter when ready, then paste token when prompted..."
huggingface-cli login

# Step 2: Upgrade timm
echo ""
echo "[2/5] Upgrading timm for UNI support..."
pip install 'timm>=0.9.16' -q

# Step 3: Skip Mamba (g++ incompatibility) — use GRU fallback
echo ""
echo "[3/5] Skipping mamba-ssm (g++ 15.2.0 > CUDA 12.8 limit)"
echo "      ✓ Model will use GRU fallback automatically"

# Step 4: Install remaining dependencies
echo ""
echo "[4/5] Installing PyTorch Geometric & Quantum stack..."
pip install torch-geometric -q
pip install 'pennylane>=0.33.0' pennylane-lightning qiskit 'qiskit-aer>=0.13' -q

# Step 5: Verify
echo ""
echo "[5/5] Verification..."
echo ""

python3 << 'EOF'
import sys
errors = []

try:
    import timm
    print(f"✅ timm {timm.__version__}")
except Exception as e:
    errors.append(f"❌ timm: {e}")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    errors.append(f"❌ PyTorch: {e}")

try:
    import pennylane as qml
    print(f"✅ PennyLane {qml.__version__}")
except Exception as e:
    errors.append(f"❌ PennyLane: {e}")

try:
    from torch_geometric.nn import GATConv
    print(f"✅ PyTorch Geometric (GATConv)")
except Exception as e:
    errors.append(f"❌ PyTorch Geometric: {e}")

try:
    from mamba_ssm import Mamba
    print(f"✅ Mamba-SSM (fast path)")
    mamba_ok = True
except Exception as e:
    print(f"⚠️  Mamba-SSM not available (expected)")
    print(f"    → Using GRU fallback (model handles automatically)")
    mamba_ok = False

try:
    from datasets import load_dataset
    print(f"✅ Datasets (HuggingFace)")
except Exception as e:
    errors.append(f"❌ Datasets: {e}")

print()
if errors:
    print("ERRORS:")
    for err in errors:
        print(f"  {err}")
    sys.exit(1)
else:
    print("✅ All imports verified!")
    if not mamba_ok:
        print("\n⚠️  NOTE: GRU fallback will be used instead of Mamba")
        print("          This is expected and fine for development")
        print("          Full Mamba can be installed on RunPod later")

EOF

echo ""
echo "=================================================="
echo "✅ Installation complete!"
echo "=================================================="
echo ""
echo "Next: Run week3_gnn_v2.ipynb"
echo "  cd notebooks && jupyter notebook week3_gnn_v2.ipynb"
echo ""
