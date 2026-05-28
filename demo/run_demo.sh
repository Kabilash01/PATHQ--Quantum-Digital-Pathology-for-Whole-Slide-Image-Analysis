#!/usr/bin/env bash
# demo/run_demo.sh — Launch QuantaPath v2 Gradio Demo
# Usage: bash demo/run_demo.sh
# Then open: http://localhost:7860

set -e

cd "$(dirname "$0")/.."
echo "======================================================"
echo "  QuantaPath v2 — Lab Exam Demo Launcher"
echo "======================================================"
echo ""

# Check conda env
if ! conda run -n pathq python -c "import gradio" 2>/dev/null; then
  echo "[!] gradio not found in pathq env. Installing..."
  conda install -n pathq -c conda-forge gradio -y
fi

echo "[✓] gradio found"
echo "[✓] Starting demo at http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop."
echo ""

conda run -n pathq python demo/app.py
