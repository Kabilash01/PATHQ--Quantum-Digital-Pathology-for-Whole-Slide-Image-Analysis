#!/bin/bash
#
# QuantaPath v2 — Installation script for pathq conda environment
# Requires: CUDA 12.1, conda/mamba, RTX 5060 8GB VRAM
#
# Usage: bash install_v2.sh

set -e

echo "════════════════════════════════════════════════════════════════"
echo "QuantaPath v2 Installation"
echo "════════════════════════════════════════════════════════════════"

# Check Python environment
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found. Activate conda env first:"
    echo "  conda activate pathq"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Step 1: Upgrade pip
echo ""
echo "Step 1: Upgrading pip..."
pip install --upgrade pip

# Step 2: HuggingFace login (needed for UNI)
echo ""
echo "Step 2: Setting up HuggingFace access for UNI..."
pip install huggingface_hub

echo ""
echo "⚠️  IMPORTANT: You must accept UNI model terms at:"
echo "    https://huggingface.co/MahmoodLab/uni"
echo ""
echo "Then login with:"
echo "    huggingface-cli login"
echo ""
read -p "Press ENTER after login, or CTRL+C to cancel: "

# Step 3: Upgrade timm for UNI support
echo ""
echo "Step 3: Upgrading timm for UNI support (need ≥0.9.16)..."
pip install timm>=0.9.16

# Step 4: Install Mamba (CUDA required)
echo ""
echo "Step 4: Installing Mamba (requires CUDA 12.1)..."
echo "        This may take a few minutes..."

if pip install mamba-ssm causal-conv1d 2>&1 | grep -q "error\|failed"; then
    echo "⚠️  Mamba installation failed. Trying --no-build-isolation..."
    pip install mamba-ssm --no-build-isolation causal-conv1d
    if [ $? -ne 0 ]; then
        echo "⚠️  Mamba-SSM not installed. Will use GRU fallback (slower but functional)."
        echo "    To enable Mamba later: pip install mamba-ssm --no-build-isolation"
    fi
else
    echo "✅ Mamba installed"
fi

# Step 5: Install PyTorch Geometric
echo ""
echo "Step 5: Installing torch-geometric..."
pip install torch-geometric

# Step 6: Install PennyLane and quantum simulators
echo ""
echo "Step 6: Installing PennyLane for VQC support..."
pip install pennylane pennylane-lightning qiskit qiskit-aer

# Step 7: Verify installation
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Step 7: Verifying installation..."
echo "════════════════════════════════════════════════════════════════"

python << 'EOF'
import sys
errors = []

# Check timm
try:
    import timm
    print(f'✅ timm: {timm.__version__}')
except ImportError as e:
    errors.append(f'❌ timm: {e}')

# Check torch
try:
    import torch
    print(f'✅ torch: {torch.__version__}')
    if torch.cuda.is_available():
        print(f'   CUDA device: {torch.cuda.get_device_name(0)}')
        print(f'   CUDA version: {torch.version.cuda}')
    else:
        print('   ⚠️  CUDA not available — CPU mode only')
except ImportError as e:
    errors.append(f'❌ torch: {e}')

# Check torch_geometric
try:
    import torch_geometric
    print(f'✅ torch_geometric: {torch_geometric.__version__}')
except ImportError as e:
    errors.append(f'❌ torch_geometric: {e}')

# Check PennyLane
try:
    import pennylane as qml
    print(f'✅ pennylane: {qml.__version__}')
except ImportError as e:
    errors.append(f'❌ pennylane: {e}')

# Check GATConv (key for model)
try:
    from torch_geometric.nn import GATConv
    print(f'✅ GATConv: available')
except ImportError as e:
    errors.append(f'❌ GATConv: {e}')

# Check Mamba (optional, has fallback)
try:
    from mamba_ssm import Mamba
    print(f'✅ mamba-ssm: available (performant quantum)')
except ImportError:
    print(f'⚠️  mamba-ssm: NOT installed (will use GRU fallback)')

# HuggingFace
try:
    import huggingface_hub
    print(f'✅ huggingface_hub: {huggingface_hub.__version__}')
except ImportError as e:
    errors.append(f'❌ huggingface_hub: {e}')

# Datasets
try:
    from datasets import load_dataset
    print(f'✅ datasets: available')
except ImportError as e:
    errors.append(f'❌ datasets: {e}')

if errors:
    print('\n❌ Some imports failed:')
    for err in errors:
        print(f'  {err}')
    sys.exit(1)
else:
    print('\n✅ All required packages installed and working')

EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "✅ Installation complete!"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "Next steps:"
    echo "  1. Run Week 2b notebook to extract UNI features:"
    echo "     jupyter notebook notebooks/week2b_uni_extraction.ipynb"
    echo ""
    echo "  2. Run Week 3 v2 training notebook:"
    echo "     jupyter notebook notebooks/week3_gnn_v2.ipynb"
    echo ""
    echo "  3. Check results:"
    echo "     cat outputs/v2_results.json"
    echo ""
else
    echo ""
    echo "❌ Installation verification failed!"
    exit 1
fi
