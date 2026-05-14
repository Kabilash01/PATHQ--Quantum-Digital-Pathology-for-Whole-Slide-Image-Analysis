"""
pathq/uni_extractor.py
UNI foundation model feature extractor.
UNI: trained on 100M+ H&E pathology slides from MahmoodLab.
Replaces ResNet-50 (512-dim ImageNet) with UNI (1024-dim pathology-specific).
"""
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from pathlib import Path
import math


def build_uni_extractor(device):
    """
    Load frozen UNI ViT-L model.
    Output: 1024-dimensional feature vectors per patch.
    """
    print('Loading UNI feature extractor from HuggingFace...')
    print('(First run downloads ~1.8GB — subsequent runs use cache)')

    backbone = timm.create_model(
        'hf-hub:MahmoodLab/uni',
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    ).to(device).eval()

    for p in backbone.parameters():
        p.requires_grad = False

    # UNI preprocessing (from MahmoodLab official repo)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    n_params = sum(p.numel() for p in backbone.parameters())
    print(f'UNI loaded: {n_params:,} params (all frozen)')
    print(f'Output dimension: 1024')

    return backbone, transform


def sinusoidal_pos_encoding(coords: torch.Tensor, d_model: int = 16) -> torch.Tensor:
    """
    Sinusoidal positional encoding from 2D patch coordinates.
    Adopted from GAT-Mamba paper (Ding et al., Scientific Reports 2025).

    Args:
        coords: (N, 2) float tensor of (col, row) grid positions
        d_model: output dimension (must be even, default 16)
    Returns:
        (N, d_model) positional encoding
    """
    N = coords.shape[0]
    pe = torch.zeros(N, d_model, dtype=torch.float32)
    for i in range(d_model // 2):
        div = math.exp(i * -math.log(10000.0) / (d_model // 2))
        pe[:, 2 * i]     = torch.sin(coords[:, 0] * div)
        pe[:, 2 * i + 1] = torch.cos(coords[:, 1] * div)
    return pe


def extract_uni_features(
    patches,          # list of PIL Images
    coords,           # list of (col, row) tuples
    uni_model,        # from build_uni_extractor()
    transform,        # from build_uni_extractor()
    device,
    batch_size=64,    # UNI is ViT-L — smaller batches than ResNet
    pos_enc_dim=16,
):
    """
    Extract UNI features + positional encoding for all patches in a slide.

    Returns:
        features: (N, 1040) tensor = UNI(1024) + pos_enc(16)
        coords_t: (N, 2) tensor
    """
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch_pils = patches[i:i+batch_size]
            batch_t    = torch.stack([transform(p) for p in batch_pils]).to(device)
            feats      = uni_model(batch_t)   # (B, 1024)
            all_feats.append(feats.cpu())

    uni_feats = torch.cat(all_feats, dim=0)           # (N, 1024)
    coords_t  = torch.tensor(coords, dtype=torch.float32)  # (N, 2)
    pos_enc   = sinusoidal_pos_encoding(coords_t, d_model=pos_enc_dim)  # (N, 16)

    features  = torch.cat([uni_feats, pos_enc], dim=1)  # (N, 1040)
    return features, coords_t
