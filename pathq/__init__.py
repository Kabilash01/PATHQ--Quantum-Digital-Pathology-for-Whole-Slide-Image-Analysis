"""
PATHQ — Quantum Digital Pathology
===================================
Quantum-hybrid AI for Whole Slide Image analysis.
"""
from .model import PATHQModel, VQCEncoder, GNNEncoder, ABMILAggregator

__version__ = "1.0.0"
__all__ = ["PATHQModel", "VQCEncoder", "GNNEncoder", "ABMILAggregator"]
