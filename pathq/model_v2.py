"""
pathq/model_v2.py
QuantaPath v2 — UNI + positional encoding + VQC + GAT-Transformer

Architecture:
    (N, 1040) node features [UNI 1024 + pos.enc 16]
        -> [Optional VQC on UNI part only] -> (N, 80) or (N, 1040)
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
    QuantaPath v2 VQCEncoder — optimised for RTX 5060 laptop

    Improvements:
      - 1024 → 128 → n_qubits compression (not direct 1024→3)
      - Data re-uploading every layer (Perez-Salinas 2020)
      - Batched forward pass (no Python for-loop — 10× faster)
      - Post-VQC expansion to 64-dim for fair comparison with classical
    """
    def __init__(self, in_dim=1024, n_qubits=3, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Compression: 1024 → 128 → n_qubits
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, n_qubits),
            nn.Tanh(),
        )

        # Post-VQC expansion for fair comparison with classical
        self.post_vqc = nn.Sequential(
            nn.Linear(n_qubits * 2, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.out_dim = 64

        try:
            dev = qml.device('lightning.gpu', wires=n_qubits)
            diff_method = 'adjoint'
            print(f'[VQC] lightning.gpu  {n_qubits}q {n_layers}L — batched + re-uploading')
        except Exception:
            dev = qml.device('lightning.qubit', wires=n_qubits)
            diff_method = 'parameter-shift'
            print(f'[VQC] lightning.qubit {n_qubits}q {n_layers}L — batched + re-uploading')

        # Data re-uploading circuit (AngleEmbedding inside the layer loop)
        @qml.qnode(dev, interface='torch', diff_method=diff_method)
        def circuit(inputs, weights):
            for l in range(n_layers):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
                for q in range(n_qubits):
                    qml.RY(weights[l, 0, q], wires=q)
                    qml.RZ(weights[l, 1, q], wires=q)
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

        # Batched TorchLayer — handles all patches in a single broadcasted call
        self.vqc = qml.qnn.TorchLayer(circuit, {'weights': (n_layers, 2, n_qubits)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p     = self.proj(x)                     # (N, n_qubits) — bounded by Tanh
        q_out = self.vqc(p)                      # (N, n_qubits) — batched, no Python loop
        cat   = torch.cat([p, q_out], dim=1)     # (N, n_qubits * 2)
        return self.post_vqc(cat)                # (N, 64)


class GATMambaBlock(nn.Module):
    """
    GAT (local attention) + Transformer (global context) fused block.
    Transformer replaces Mamba/GRU for global sequence modeling.
    """
    def __init__(self, dim=256, n_heads=4, dropout=0.5, edge_dim=2):
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
        dropout   = 0.5,
    ):
        super().__init__()
        self.use_vqc  = use_vqc
        self.feat_dim = in_dim - 16   # feature dim without pos_enc (1024 UNI, 512 ResNet)

        if use_vqc:
            self.vqc = VQCEncoder(in_dim=self.feat_dim, n_qubits=n_qubits, n_layers=vqc_layers)
            proj_in  = self.vqc.out_dim + 16   # 64 + 16 = 80
        else:
            self.vqc = None
            proj_in  = in_dim                  # 1040 or 528

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

        uni  = x[:, :self.feat_dim]   # feature part (1024 UNI or 512 ResNet)
        pe   = x[:, self.feat_dim:]   # positional encoding (last 16 dims)

        if self.use_vqc:
            x_in = torch.cat([self.vqc(uni), pe], dim=1)   # (N, 80)
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
