"""
pathq/dataset_v2.py
Updated dataset and graph builder for QuantaPath v2.
Key additions vs v1:
  - Node features: UNI(1024) + sinusoidal pos.enc(16) = 1040-dim
  - Edge features: Euclidean distance + cosine similarity = 2-dim
  - Balanced splits from separate normal/tumor folders
  - FIXED: Bag labels guaranteed correct by separating pos/neg indices
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.loader import DataLoader as PyGLoader
from pathq.uni_extractor import sinusoidal_pos_encoding


def get_label(path: Path) -> int:
    parts = [p.lower() for p in path.parts]
    return 1 if 'tumor' in parts else 0


def get_splits(slides_dir, train_r=0.70, val_r=0.15, seed=42):
    """70/15/15 stratified split from flat directory with normal_xxx/tumor_xxx files."""
    slides_dir = Path(slides_dir)
    all_slides = sorted(slides_dir.glob('*.tif'))

    # Parse label from filename (normal_xxx -> 0, tumor_xxx -> 1)
    paths = []
    labels = []
    for p in all_slides:
        label = 1 if p.stem.startswith('tumor') else 0
        paths.append(p)
        labels.append(label)

    # Stratified split
    tr_p, tmp_p, tr_l, tmp_l = train_test_split(
        paths, labels, test_size=1-train_r, stratify=labels, random_state=seed)
    va_p, te_p, va_l, te_l = train_test_split(
        tmp_p, tmp_l, test_size=0.5, stratify=tmp_l, random_state=seed)

    pos_train = sum(tr_l)
    pos_val = sum(va_l)
    pos_test = sum(te_l)
    print(f'Split: train={len(tr_p)} (pos={pos_train}) val={len(va_p)} (pos={pos_val}) test={len(te_p)} (pos={pos_test})')
    return {'train': list(zip(tr_p, tr_l)),
            'val':   list(zip(va_p, va_l)),
            'test':  list(zip(te_p, te_l))}


def build_graph_v2(features: torch.Tensor, coords: torch.Tensor,
                   label: int, k: int = 8, pos_enc_dim: int = 16) -> Data:
    """
    Build PyG graph.
    - Node features: UNI(1024) + pos.enc(16) = (N, 1040)
    - Edge features: [dist, cosine_sim] = (E, 2)
    """
    N = features.shape[0]
    pos_enc       = sinusoidal_pos_encoding(coords, d_model=pos_enc_dim)
    node_features = torch.cat([features.float(), pos_enc], dim=1)  # (N, 1040)

    tree = KDTree(coords.numpy())
    _, indices = tree.query(coords.numpy(), k=min(k+1, N))

    src, tgt, dists, sims = [], [], [], []
    for i in range(N):
        for j in indices[i, 1:]:
            j = int(j)
            src += [i, j]; tgt += [j, i]
            d = torch.norm(coords[i] - coords[j]).item()
            dists += [d, d]
            fi = F.normalize(features[i:i+1], p=2, dim=1)
            fj = F.normalize(features[j:j+1], p=2, dim=1)
            s  = (fi * fj).sum().item()
            sims += [s, s]

    ei  = torch.tensor([src, tgt], dtype=torch.long)
    dst = torch.tensor(dists, dtype=torch.float32)
    if dst.max() > 0: dst = dst / dst.max()
    ea  = torch.stack([dst, torch.tensor(sims, dtype=torch.float32)], dim=1)

    return Data(x=node_features, edge_index=ei, edge_attr=ea,
                y=torch.tensor([label], dtype=torch.long),
                coords=coords.float(), num_nodes=N)


class CAMELYON16GraphDataset(PyGDataset):
    """
    Loads pre-extracted UNI feature .pt files and builds graphs.
    Each .pt file must contain: {'features': (N,1024), 'coords': (N,2)}
    """
    def __init__(self, slide_items, features_dir: Path, k=8):
        self.items        = slide_items
        self.features_dir = features_dir
        self.k            = k
        valid = []
        for path, label in slide_items:
            sid = Path(path).stem
            fp  = features_dir / f'{sid}_uni_features.pt'
            if fp.exists():
                valid.append((path, label, fp))
            else:
                print(f'  MISSING: {sid}_uni_features.pt — skipping')
        self.valid = valid
        print(f'  Dataset: {len(valid)}/{len(slide_items)} slides have features')
        super().__init__(root=None)

    def len(self): return len(self.valid)

    def get(self, idx):
        path, label, fp = self.valid[idx]
        d = torch.load(fp, weights_only=False)
        return build_graph_v2(d['features'], d['coords'], label, self.k)


def get_loaders_from_features(features_dir, batch_size=4, k=8, seed=42, num_workers=0):
    """
    Load all pre-extracted UNI features and split into train/val/test.
    Used when features are pre-computed from patch extraction.
    """
    features_dir = Path(features_dir)
    all_features = sorted(features_dir.glob('*_uni_features.pt'))

    if len(all_features) == 0:
        raise ValueError(f"No features found in {features_dir}")

    # Parse labels from filenames (normal_xxx -> 0, tumor_xxx -> 1)
    paths, labels = [], []
    for fp in all_features:
        label = 1 if 'tumor' in fp.stem else 0
        paths.append(fp)
        labels.append(label)

    # Stratified split
    tr_p, tmp_p, tr_l, tmp_l = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=seed)
    va_p, te_p, va_l, te_l = train_test_split(
        tmp_p, tmp_l, test_size=0.5, stratify=tmp_l, random_state=seed)

    pos_train = sum(tr_l)
    pos_val = sum(va_l)
    pos_test = sum(te_l)
    print(f'Split: train={len(tr_p)} (pos={pos_train}) val={len(va_p)} (pos={pos_val}) test={len(te_p)} (pos={pos_test})')

    # Build dataset from feature files
    class SimpleGraphDataset(PyGDataset):
        def __init__(self, feature_paths, labels_list, k=8):
            self.pairs = list(zip(feature_paths, labels_list))
            self.k = k
            super().__init__(root=None)

        def len(self):
            return len(self.pairs)

        def get(self, idx):
            fp, label = self.pairs[idx]
            d = torch.load(fp, weights_only=False)
            return build_graph_v2(d['features'], d['coords'], label, self.k)

    train_ds = SimpleGraphDataset(tr_p, tr_l, k)
    val_ds   = SimpleGraphDataset(va_p, va_l, k)
    test_ds  = SimpleGraphDataset(te_p, te_l, k)

    return (
        PyGLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        PyGLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        PyGLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


# ════════════════════════════════════════════════════════════════════════════
# PATCHCAMELYON MIL BAGS WITH FIXED LABEL ASSIGNMENT
# ════════════════════════════════════════════════════════════════════════════
# NOTE: Use this for building bags from PatchCamelyon when UNI features not yet extracted

BAG_SIZE = 16  # patches per bag


def _build_one_graph_from_patches(
    dataset, patch_indices, label, uni_extractor, transform, device
):
    """Build one PyG graph from a list of patch indices using UNI extraction."""
    imgs = torch.stack([transform(dataset[int(j)]['image']) for j in patch_indices]).to(device)
    with torch.no_grad():
        feats = uni_extractor(imgs).cpu()  # (B, 1024)

    side   = int(BAG_SIZE**0.5)
    coords = torch.tensor(
        [(r, c) for r in range(side) for c in range(side)],
        dtype=torch.float32)[:BAG_SIZE]

    tree = KDTree(coords.numpy())
    _, nn_idx = tree.query(coords.numpy(), k=min(5, BAG_SIZE))

    src, tgt = [], []
    for ni, nbrs in enumerate(nn_idx):
        for nj in nbrs[1:]:
            src += [ni, int(nj)]
            tgt += [int(nj), ni]

    return Data(
        x=feats, coords=coords,
        edge_index=torch.tensor([src, tgt], dtype=torch.long),
        y=torch.tensor([label], dtype=torch.long),
    )


def make_graphs_patchcamelyon_fixed(split, dataset, uni_extractor, transform,
                                     device, n_bags=400, seed=42):
    """
    FIXED VERSION: Builds balanced bags by keeping positive and negative
    indices SEPARATE. Never mixes them so bag labels are guaranteed correct.

    Args:
        split: 'train', 'valid', or 'test'
        dataset: PatchCamelyon dataset (from load_dataset)
        uni_extractor: UNI model from build_uni_extractor()
        transform: UNI transform from build_uni_extractor()
        device: torch device
        n_bags: number of bags to create (will be split 50/50 normal/tumor)
        seed: random seed

    Returns:
        graphs: list of PyG Data objects with guaranteed balanced labels
    """
    rng = np.random.default_rng(seed)
    ds = dataset[split]

    # Separate positive and negative indices
    print(f'  Indexing {split} (this takes ~1 min)...')
    pos_idx = [i for i in range(len(ds)) if ds[int(i)]['label'] == 1]
    neg_idx = [i for i in range(len(ds)) if ds[int(i)]['label'] == 0]
    print(f'  pos={len(pos_idx):,}  neg={len(neg_idx):,}')

    half = n_bags // 2
    graphs = []

    # Build negative bags (all-normal patches) -> label = 0
    print(f'  Building {half} negative bags...')
    rng.shuffle(neg_idx)
    neg_pool = neg_idx[:half * BAG_SIZE]
    for i in range(0, len(neg_pool) - BAG_SIZE + 1, BAG_SIZE):
        bi = neg_pool[i:i+BAG_SIZE]
        graphs.append(_build_one_graph_from_patches(
            ds, bi, label=0,
            uni_extractor=uni_extractor, transform=transform, device=device
        ))

    # Build positive bags (all-tumour patches) -> label = 1
    print(f'  Building {half} positive bags...')
    rng.shuffle(pos_idx)
    pos_pool = pos_idx[:half * BAG_SIZE]
    for i in range(0, len(pos_pool) - BAG_SIZE + 1, BAG_SIZE):
        bi = pos_pool[i:i+BAG_SIZE]
        graphs.append(_build_one_graph_from_patches(
            ds, bi, label=1,
            uni_extractor=uni_extractor, transform=transform, device=device
        ))

    rng.shuffle(graphs)
    pos = sum(g.y.item() for g in graphs)
    neg = len(graphs) - pos
    print(f'  {split}: {len(graphs)} bags  pos={pos}  neg={neg}')

    # Assert balance — this catches any issues immediately
    assert pos > 0, f'ERROR: {split} has no positive bags'
    assert neg > 0, f'ERROR: {split} has no negative bags'

    return graphs
