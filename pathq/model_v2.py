"""
pathq/model_v2.py
QuantaPath v2 — UNI + positional encoding + VQC + GAT-Transformer

Architecture:
    (N, 1040) node features [UNI 1024 + pos.enc 16]
        -> [Optional VQC on UNI part only] -> (N, 22) or (N, 1040)
        -> Linear projection -> (N, hidden=256)
        -> GATMambaBlock (GAT local + Transformer global, fused)
        -> Global mean pooling -> (B, hidden)
        -> Classifier head -> (B, 2) binary logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from torch_geometric.nn import GATConv, global_mean_pool

# Global context: Transformer encoder
print('[model_v2] Using Transformer for global branch')


class VQCEncoder(nn.Module):
    """
    VQC encoder using AngleEmbedding: UNI(1024) -> proj(3) -> AngleEmbedding -> VQC -> measure(3)
    Output: concat(proj_3, quantum_3) = 6-dim hybrid features

    AngleEmbedding encodes n values as rotation angles on n qubits — no norm-1 constraint,
    no MottonenStatePreparation decomposition, no NaN from normalization.
    parameter-shift gradient works directly on rotation gates.
    """
    def __init__(self, in_dim=1024, n_qubits=3, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits

        # Project to n_qubits (AngleEmbedding takes one value per qubit)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, n_qubits),
            nn.Tanh(),   # bounds to [-1, 1] → safe rotation angles
        )

        try:
            dev = qml.device('lightning.gpu', wires=n_qubits)
            diff_method = 'adjoint'
            print(f'[VQC] lightning.gpu ({n_qubits}q, {n_layers}L) — adjoint gradients')
        except Exception:
            try:
                dev = qml.device('lightning.qubit', wires=n_qubits)
                diff_method = 'adjoint'
                print(f'[VQC] lightning.qubit ({n_qubits}q, {n_layers}L) — adjoint gradients')
            except Exception:
                dev = qml.device('default.qubit', wires=n_qubits)
                diff_method = 'parameter-shift'
                print(f'[VQC] default.qubit ({n_qubits}q, {n_layers}L) — parameter-shift')

        @qml.qnode(dev, interface='torch', diff_method=diff_method)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            for l in range(n_layers):
                for q in range(n_qubits): qml.RY(weights[l, 0, q], wires=q)
                for q in range(n_qubits): qml.RZ(weights[l, 1, q], wires=q)
                for q in range(n_qubits - 1): qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

        self.vqc = qml.qnn.TorchLayer(circuit, {'weights': (n_layers, 2, n_qubits)})
        self.out_dim = n_qubits + n_qubits   # proj(3) + quantum(3) = 6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.proj(x)   # (N, n_qubits) — bounded by Tanh

        vqc_outputs = []
        for i in range(p.shape[0]):
            out = self.vqc(p[i])
            vqc_outputs.append(out.unsqueeze(0))
            if i % 50 == 49 and x.device.type == 'cuda':
                torch.cuda.empty_cache()
        q_out = torch.cat(vqc_outputs, dim=0)

        return torch.cat([p, q_out], dim=1)


class GATMambaBlock(nn.Module):
    """
    GAT (local attention) + Transformer (global context) fused block.
    Transformer replaces Mamba/GRU for global sequence modeling.
    """
    def __init__(self, dim=256, n_heads=4, dropout=0.3, edge_dim=2):
        super().__init__()
        # GAT branch (local attention with edge features)
        self.gat    = GATConv(dim, dim, heads=n_heads, concat=False,
                              dropout=dropout, edge_dim=edge_dim)
        self.bn_gat = nn.BatchNorm1d(dim)
        self.drop   = nn.Dropout(dropout)

        # Transformer branch (global context per slide)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN — more stable
        )
        self.global_enc = nn.TransformerEncoder(
            transformer_layer,
            num_layers=1,
            enable_nested_tensor=False  # ← Suppress warning
        )
        self.bn_transformer = nn.BatchNorm1d(dim)

        # Fusion MLP
        self.mlp    = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim)
        )
        self.bn_out = nn.BatchNorm1d(dim)

    def _run_global(self, x, batch_idx):
        """Run Transformer per slide to avoid mixing patches across slides."""
        n_slides = batch_idx.max().item() + 1
        parts = []
        for b in range(n_slides):
            mask = (batch_idx == b)
            x_b  = x[mask].unsqueeze(0)            # (1, N_b, dim)
            out  = self.global_enc(x_b).squeeze(0) # (N_b, dim)
            parts.append(out)
        return torch.cat(parts, dim=0)              # (N_total, dim)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Local branch (GAT)
        x_gat = self.drop(F.gelu(self.bn_gat(
            self.gat(x, edge_index, edge_attr=edge_attr)
        )))
        # Global branch (Transformer)
        x_transformer = self._run_global(x, batch) if batch is not None \
                        else self.global_enc(x.unsqueeze(0)).squeeze(0)
        x_transformer = self.bn_transformer(x_transformer)
        # Fuse and MLP
        fused = x_gat + x_transformer
        return self.bn_out(self.mlp(fused)) + fused    # residual


class QuantaPathV2(nn.Module):
    """
    Full QuantaPath v2 model.

    use_vqc=True  -> quantum mode  (VQC + GAT-Transformer) — paper row 2
    use_vqc=False -> classical mode (GAT-Transformer only) — paper row 1 baseline
    """
    def __init__(
        self,
        in_dim    = 1040,    # 1024 UNI + 16 pos.enc
        hidden    = 256,
        n_heads   = 4,
        n_qubits  = 3,
        vqc_layers= 2,
        n_classes = 2,
        use_vqc   = True,
        dropout   = 0.4,
    ):
        super().__init__()
        self.use_vqc = use_vqc

        if use_vqc:
            self.vqc = VQCEncoder(in_dim=1024, n_qubits=n_qubits, n_layers=vqc_layers)
            proj_in  = self.vqc.out_dim + 16   # 6 + 16 = 22
        else:
            self.vqc = None
            proj_in  = in_dim                  # 1040

        self.input_proj = nn.Sequential(
            nn.Linear(proj_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gat_mamba = GATMambaBlock(dim=hidden, n_heads=n_heads,
                                        dropout=dropout, edge_dim=2)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(),
            nn.Dropout(dropout),   nn.Linear(128, n_classes),
        )

        n_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'QuantaPathV2: use_vqc={use_vqc}, trainable={n_p:,}')

    def forward(self, batch):
        x  = batch.x              # (N, 1040)
        ei = batch.edge_index
        ea = getattr(batch, 'edge_attr', None)   # (E, 2)
        bi = batch.batch

        uni  = x[:, :1024]        # UNI features
        pe   = x[:, 1024:]        # positional encoding

        if self.use_vqc:
            x_in = torch.cat([self.vqc(uni), pe], dim=1)   # (N, 22)
        else:
            x_in = x                                        # (N, 1040)

        x_h = self.input_proj(x_in)                         # (N, hidden)
        x_h = self.gat_mamba(x_h, ei, ea, bi)               # (N, hidden)
        sf  = global_mean_pool(x_h, bi)                     # (B, hidden)
        return self.head(sf), None


if __name__ == '__main__':
    # Test model
    from torch_geometric.data import Data, Batch
    print('Testing QuantaPathV2 models...')

    # Create fake batch
    g1 = Data(x=torch.randn(20, 1040), edge_index=torch.randint(0,20,(2,60)),
              edge_attr=torch.randn(60,2), y=torch.tensor([1]))
    g2 = Data(x=torch.randn(25, 1040), edge_index=torch.randint(0,25,(2,75)),
              edge_attr=torch.randn(75,2), y=torch.tensor([0]))
    batch = Batch.from_data_list([g1, g2])

    # Test classical
    mc = QuantaPathV2(use_vqc=False)
    with torch.no_grad():
        lc, _ = mc(batch)
    print(f'Classical logits: {lc.shape}')

    # Test quantum
    mq = QuantaPathV2(use_vqc=True)
    with torch.no_grad():
        lq, _ = mq(batch)
    print(f'Quantum logits: {lq.shape}')

    print('✅ QuantaPathV2 models OK')
