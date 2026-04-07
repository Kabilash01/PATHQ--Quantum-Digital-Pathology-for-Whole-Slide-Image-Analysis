"""
pathq/model.py
==============
Core PATHQ model architecture.

Three classes:
    VQCEncoder         - Variational Quantum Circuit patch encoder
    ABMILAggregator    - Attention-based slide-level aggregation
    PATHQModel         - Full hybrid model (ResNet + VQC + GNN + ABMIL)

Usage:
    from pathq.model import PATHQModel
    model = PATHQModel(use_vqc=True)
    out = model(graph_batch)  # returns (logits, attention_weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pennylane as qml
from torch_geometric.nn import GCNConv, global_mean_pool


# ─────────────────────────────────────────────────────────────────────────────
# 1. Variational Quantum Circuit Encoder
# ─────────────────────────────────────────────────────────────────────────────

class VQCEncoder(nn.Module):
    """
    Quantum patch feature encoder using a Variational Quantum Circuit.
    
    Pipeline:
        512-dim feature → Linear projection → 8-dim →
        Amplitude encoding into 3-qubit state →
        2-layer VQC with RY/RZ/CNOT →
        3 Pauli-Z measurements → 3-dim quantum features
    
    The 3-dim quantum output is concatenated with the projected 8-dim
    classical features → 11-dim hybrid feature per patch.
    
    Args:
        feature_dim:  Input dimension (512 from ResNet)
        vqc_input:    Projected dimension = 2^n_qubits (8 for 3 qubits)
        n_qubits:     Number of qubits (3 recommended for speed)
        n_layers:     VQC circuit depth (2 recommended)
        device_name:  PennyLane device ('default.qubit' or 'qiskit.aer')
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        vqc_input: int = 8,       # must equal 2^n_qubits
        n_qubits: int = 3,
        n_layers: int = 2,
        device_name: str = 'default.qubit',
    ):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.vqc_input = vqc_input
        
        assert vqc_input == 2 ** n_qubits, \
            f"vqc_input ({vqc_input}) must equal 2^n_qubits ({2**n_qubits})"
        
        # Classical projection: 512 → 8
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, vqc_input),
            nn.Tanh(),  # Tanh bounds output — important for amplitude encoding
        )
        
        # PennyLane quantum device (CPU simulator)
        self.dev = qml.device(device_name, wires=n_qubits)
        
        # Weight shapes for the VQC: (n_layers, n_qubits) parameters per layer type
        # Each layer has RY + RZ rotations → 2 * n_qubits params per layer
        n_params = n_layers * 2 * n_qubits
        weight_shapes = {"weights": (n_layers, 2, n_qubits)}
        
        # Define the quantum circuit
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def quantum_circuit(inputs, weights):
            """
            VQC circuit:
            1. Amplitude encode the 8-dim classical input
            2. Apply n_layers of: RY rotations → RZ rotations → CNOT entanglement
            3. Measure Pauli-Z on all qubits → n_qubits expectation values
            """
            # Amplitude encoding: map 8 classical values to quantum state amplitudes
            # Requires inputs to be normalised to unit norm (done in forward())
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
            
            # Variational layers
            for l in range(n_layers):
                # RY rotations (parameterised)
                for q in range(n_qubits):
                    qml.RY(weights[l, 0, q], wires=q)
                
                # RZ rotations (parameterised)
                for q in range(n_qubits):
                    qml.RZ(weights[l, 1, q], wires=q)
                
                # CNOT entanglement (brick-wall pattern)
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                if n_qubits > 2:
                    qml.CNOT(wires=[n_qubits - 1, 0])  # wrap-around
            
            # Measure Pauli-Z expectation on all qubits
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
        
        # Wrap as PyTorch layer (enables joint gradient-based training)
        self.vqc = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # Output: projected (8) + quantum measurements (n_qubits)
        self.output_dim = vqc_input + n_qubits
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 512) patch feature vectors
        Returns:
            (B, 8 + n_qubits) hybrid classical-quantum features
        """
        # Project to VQC input dimension
        projected = self.projection(x)  # (B, 8)
        
        # Normalise to unit norm for amplitude encoding
        # AmplitudeEmbedding requires ||x||₂ = 1
        projected_norm = F.normalize(projected, p=2, dim=1)  # (B, 8)
        
        # Pass through VQC — produces quantum expectation values
        quantum_out = self.vqc(projected_norm)  # (B, n_qubits)
        
        # Concatenate classical projection + quantum output
        hybrid = torch.cat([projected, quantum_out], dim=1)  # (B, 8+n_qubits)
        
        return hybrid


# ─────────────────────────────────────────────────────────────────────────────
# 2. Graph Neural Network + ABMIL Aggregator
# ─────────────────────────────────────────────────────────────────────────────

class ABMILAggregator(nn.Module):
    """
    Attention-Based Multiple Instance Learning aggregator.
    
    Produces a slide-level representation as the attention-weighted
    mean of all patch features. The attention weights are the XAI
    Layer 2 output — high weight = this patch was important.
    
    Args:
        input_dim:   Dimension of each patch feature
        hidden_dim:  Hidden dimension of attention network
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Attention network: maps each patch feature to a scalar weight
        # Uses tanh + sigmoid activation (Ilse et al. 2018 formulation)
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        """
        Args:
            x:     (N_total, D) all patch features from the batch
            batch: (N_total,)   graph index for each patch (from PyG batch)
        Returns:
            slide_feat:   (B, D) slide-level features
            attn_weights: (N_total, 1) attention weight per patch
        """
        # Compute attention scores
        A_V = self.attention_V(x)       # (N, hidden)
        A_U = self.attention_U(x)       # (N, hidden)
        A   = self.attention_w(A_V * A_U)  # (N, 1) — gated attention
        
        # Softmax within each slide (using batch index for grouping)
        # We manually do this per slide
        B = batch.max().item() + 1  # number of slides in batch
        slide_features = []
        attn_weights   = torch.zeros(x.shape[0], 1, device=x.device)
        
        for b_idx in range(B):
            mask = (batch == b_idx)  # patches belonging to slide b_idx
            A_b  = A[mask]           # attention logits for this slide
            x_b  = x[mask]          # features for this slide
            
            # Softmax to get normalised attention weights
            w_b  = torch.softmax(A_b, dim=0)  # (N_b, 1)
            attn_weights[mask] = w_b
            
            # Weighted sum → slide representation
            slide_feat = (w_b * x_b).sum(dim=0, keepdim=True)  # (1, D)
            slide_features.append(slide_feat)
        
        slide_features = torch.cat(slide_features, dim=0)  # (B, D)
        
        return slide_features, attn_weights


class GNNEncoder(nn.Module):
    """
    Two-layer Graph Convolutional Network encoder.
    
    Takes patch graph as input, outputs enriched patch representations
    that incorporate spatial neighbourhood context.
    
    Args:
        input_dim:  Dimension of input patch features
        hidden_dim: Hidden dimension
        output_dim: Output dimension
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.bn2   = nn.BatchNorm1d(output_dim)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          (N, input_dim) node features
            edge_index: (2, E) edge connectivity
        Returns:
            (N, output_dim) enriched node features
        """
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. Full PATHQ Model
# ─────────────────────────────────────────────────────────────────────────────

class PATHQModel(nn.Module):
    """
    PATHQ: Quantum Digital Pathology Model
    
    Full pipeline:
        Patch features (512-dim)
            → [Optional] VQC Encoder → hybrid features (511-dim)
            → GNN Encoder (2-layer GCNConv) → enriched features (256-dim)
            → ABMIL Aggregator → slide-level feature (256-dim) + attention weights
            → Classifier head → binary prediction
    
    Args:
        feature_dim:   Input patch feature dimension (512 from ResNet-50)
        gnn_hidden:    GCN hidden dimension
        n_qubits:      Number of qubits in VQC (3 default)
        vqc_layers:    VQC circuit depth (2 default)
        n_classes:     Output classes (2 for binary)
        use_vqc:       If False, skips VQC (classical baseline mode)
        dropout:       Dropout rate in classifier
    """
    
    def __init__(
        self,
        feature_dim:  int  = 512,
        gnn_hidden:   int  = 256,
        n_qubits:     int  = 3,
        vqc_layers:   int  = 2,
        n_classes:    int  = 2,
        use_vqc:      bool = True,
        dropout:      float = 0.3,
    ):
        super().__init__()
        
        self.use_vqc = use_vqc
        
        if use_vqc:
            # Quantum encoder: 512 → 8 + 3 = 11
            self.vqc_encoder = VQCEncoder(
                feature_dim=feature_dim,
                vqc_input=2 ** n_qubits,
                n_qubits=n_qubits,
                n_layers=vqc_layers,
            )
            gnn_input_dim = self.vqc_encoder.output_dim  # 11
        else:
            self.vqc_encoder = None
            gnn_input_dim = feature_dim  # 512
        
        # GNN encoder
        self.gnn = GNNEncoder(
            input_dim=gnn_input_dim,
            hidden_dim=gnn_hidden,
            output_dim=gnn_hidden,
        )
        
        # ABMIL slide aggregator
        self.aggregator = ABMILAggregator(
            input_dim=gnn_hidden,
            hidden_dim=128,
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(gnn_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )
        
        print(f'PATHQModel initialised:')
        print(f'  VQC:          {"ON" if use_vqc else "OFF (classical baseline)"}')
        print(f'  GNN input:    {gnn_input_dim}-dim')
        print(f'  GNN hidden:   {gnn_hidden}-dim')
        print(f'  Output:       {n_classes} classes')
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'  Trainable params: {n_params:,}')
    
    def forward(self, batch):
        """
        Args:
            batch: PyTorch Geometric batch object containing:
                   batch.x          - (N_total, 512) all patch features
                   batch.edge_index  - (2, E_total) all edges
                   batch.batch       - (N_total,) graph membership index
        
        Returns:
            logits:       (B, n_classes) classification logits
            attn_weights: (N_total, 1) ABMIL attention weights per patch
        """
        x          = batch.x           # (N_total, 512)
        edge_index = batch.edge_index   # (2, E_total)
        graph_idx  = batch.batch        # (N_total,)
        
        # Step 1: Optional VQC encoding
        if self.use_vqc:
            x = self.vqc_encoder(x)    # (N_total, 11)
        
        # Step 2: GNN message passing
        x = self.gnn(x, edge_index)    # (N_total, 256)
        
        # Step 3: ABMIL slide aggregation
        slide_feat, attn_weights = self.aggregator(x, graph_idx)  # (B, 256), (N_total, 1)
        
        # Step 4: Classify
        logits = self.classifier(slide_feat)  # (B, 2)
        
        return logits, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# Quick test when run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from torch_geometric.data import Data, Batch
    
    print('Testing PATHQModel...')
    
    # Create two fake slides
    def make_fake_slide(n_patches, label, feature_dim=512, k=8):
        from scipy.spatial import KDTree
        features = torch.randn(n_patches, feature_dim)
        coords   = torch.randint(0, 20, (n_patches, 2)).float()
        
        tree = KDTree(coords.numpy())
        _, indices = tree.query(coords.numpy(), k=min(k+1, n_patches))
        src, tgt = [], []
        for i in range(n_patches):
            for j in indices[i, 1:]:
                src.extend([i, j])
                tgt.extend([j, i])
        edge_index = torch.tensor([src, tgt], dtype=torch.long)
        
        return Data(x=features, edge_index=edge_index,
                    y=torch.tensor([label], dtype=torch.long),
                    num_nodes=n_patches)
    
    slide1 = make_fake_slide(50, label=1)
    slide2 = make_fake_slide(30, label=0)
    batch  = Batch.from_data_list([slide1, slide2])
    
    # Test classical baseline
    print('\n--- Classical Baseline (no VQC) ---')
    model_classical = PATHQModel(use_vqc=False)
    model_classical.eval()
    with torch.no_grad():
        logits, attn = model_classical(batch)
    print(f'Logits:  {logits.shape}  →  {logits}')
    print(f'Attention: {attn.shape}  min={attn.min():.4f} max={attn.max():.4f}')
    
    # Test quantum hybrid
    print('\n--- Quantum Hybrid (with VQC) ---')
    model_quantum = PATHQModel(use_vqc=True, n_qubits=3, vqc_layers=2)
    model_quantum.eval()
    with torch.no_grad():
        logits_q, attn_q = model_quantum(batch)
    print(f'Logits:  {logits_q.shape}  →  {logits_q}')
    print(f'Attention: {attn_q.shape}')
    
    print('\nAll tests passed.')
