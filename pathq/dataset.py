"""
pathq/dataset.py
================
CAMELYON16 Dataset — handles separate tumor/ and normal/ folders.

Your folder structure:
    data/camelyon16/
        normal/     ← normal_001.tif ... normal_111.tif  (label = 0)
        tumor/      ← tumor_001.tif  ... tumor_111.tif   (label = 1)

This file gives you:
    - CAMELYON16Raw       : loads raw .tif slides (for patch extraction)
    - CAMELYON16GraphDataset : loads pre-extracted .pt features as PyG graphs
    - get_splits()        : reproducible 70/15/15 train/val/test split
    - get_loaders()       : returns all three DataLoaders ready to use
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from scipy.spatial import KDTree
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# Helper: assign label from folder name
# ─────────────────────────────────────────────────────────────────────────────

def label_from_path(path: Path) -> int:
    """
    Returns 1 if slide is in tumor/ folder, 0 if in normal/ folder.
    Works for any subfolder depth.
    """
    parts = [p.lower() for p in path.parts]
    if 'tumor' in parts:
        return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Helper: balanced split
# ─────────────────────────────────────────────────────────────────────────────

def get_splits(normal_dir: Path, tumor_dir: Path,
               train_ratio: float = 0.70,
               val_ratio:   float = 0.15,
               seed:        int   = 42) -> dict:
    """
    Creates reproducible 70/15/15 stratified train/val/test splits.

    Balances classes: caps normal slides to match tumor count.

    Returns:
        {
          'train': [(path, label), ...],
          'val':   [(path, label), ...],
          'test':  [(path, label), ...],
        }
    """
    # Collect all slides
    normal_slides = sorted(normal_dir.glob('*.tif'))
    tumor_slides  = sorted(tumor_dir.glob('*.tif'))

    # Balance: cap normal to match tumor count
    n = min(len(normal_slides), len(tumor_slides))
    rng = np.random.default_rng(seed)
    normal_slides = rng.choice(normal_slides, size=n, replace=False).tolist()
    tumor_slides  = rng.choice(tumor_slides,  size=n, replace=False).tolist()

    all_paths  = normal_slides + tumor_slides
    all_labels = [0] * n + [1] * n

    print(f'Dataset: {n} normal + {n} tumor = {2*n} total slides')

    # Stratified split: train vs (val + test)
    test_val_ratio = 1.0 - train_ratio
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels,
        test_size=test_val_ratio,
        stratify=all_labels,
        random_state=seed
    )

    # Split temp into val and test equally
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=seed
    )

    splits = {
        'train': list(zip(train_paths, train_labels)),
        'val':   list(zip(val_paths,   val_labels)),
        'test':  list(zip(test_paths,  test_labels)),
    }

    for split_name, items in splits.items():
        pos = sum(2 for _, l in items if l == 1)
        neg = len(items) - pos
        print(f'  {split_name:5s}: {len(items):3d} slides  '
              f'({pos} tumor, {neg} normal)')

    return splits


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 1: Raw slides (for patch extraction — Week 1-2)
# ─────────────────────────────────────────────────────────────────────────────

class CAMELYON16Raw(Dataset):
    """
    Simple dataset that returns (slide_path, label) pairs.
    Used during patch extraction (Week 2) — not for training.

    Usage:
        dataset = CAMELYON16Raw(
            normal_dir=Path('./data/camelyon16/normal'),
            tumor_dir=Path('./data/camelyon16/tumor'),
            split='train'
        )
        for slide_path, label in dataset:
            patches, coords, slide_id = extract_patches_from_slide(slide_path)
    """

    def __init__(self, normal_dir: Path, tumor_dir: Path,
                 split: str = 'train', seed: int = 42):
        self.splits = get_splits(normal_dir, tumor_dir, seed=seed)
        self.items  = self.splits[split]
        self.split  = split

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        return Path(path), label

    def get_slide_id(self, idx):
        path, _ = self.items[idx]
        return Path(path).stem


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build K-NN graph from features + coords
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(features: torch.Tensor, coords: torch.Tensor,
                label: int, k: int = 8) -> Data:
    """
    Builds a PyG graph from patch features and 2D coordinates.

    Nodes  = patches (feature vectors)
    Edges  = K nearest spatial neighbours
    Label  = slide-level binary label
    """
    N = features.shape[0]
    k_actual = min(k + 1, N)

    tree = KDTree(coords.numpy())
    _, indices = tree.query(coords.numpy(), k=k_actual)

    src, tgt = [], []
    for i in range(N):
        for j in indices[i, 1:]:   # skip self (index 0)
            src += [i, int(j)]
            tgt += [int(j), i]     # undirected: both directions

    edge_index = torch.tensor([src, tgt], dtype=torch.long)

    return Data(
        x          = features.float(),                       # (N, 512)
        edge_index = edge_index,                              # (2, E)
        y          = torch.tensor([label], dtype=torch.long),# slide label
        coords     = coords.float(),                          # (N, 2) for XAI
        num_nodes  = N,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 2: Graph dataset from pre-extracted features (for GNN training)
# ─────────────────────────────────────────────────────────────────────────────

class CAMELYON16GraphDataset(PyGDataset):
    """
    Loads pre-extracted ResNet-50 feature .pt files and builds
    patch graphs for GNN training.

    Pre-requisite: run extract_all_features() first (Week 2 notebook).

    Each .pt file contains:
        {
            'slide_id': str,
            'features': Tensor (N, 512),
            'coords':   Tensor (N, 2),
            'n_patches': int,
        }

    Usage:
        train_ds, val_ds, test_ds = get_graph_datasets(
            features_dir=Path('./data/features'),
            normal_dir=Path('./data/camelyon16/normal'),
            tumor_dir=Path('./data/camelyon16/tumor'),
        )
    """

    def __init__(self, slide_items: list, features_dir: Path, k: int = 8):
        """
        Args:
            slide_items:  List of (slide_path, label) tuples
            features_dir: Directory containing {slide_id}_features.pt files
            k:            K-nearest neighbours for graph construction
        """
        self.items        = slide_items
        self.features_dir = features_dir
        self.k            = k

        # Filter to only slides that have extracted features
        self.valid_items = []
        for path, label in slide_items:
            slide_id  = Path(path).stem
            feat_path = features_dir / f'{slide_id}_features.pt'
            if feat_path.exists():
                self.valid_items.append((path, label, feat_path))
            else:
                print(f'  WARNING: No features for {slide_id} — skipping')

        print(f'  Loaded {len(self.valid_items)}/{len(slide_items)} slides '
              f'(features found)')
        super().__init__(root=None)

    def len(self):
        return len(self.valid_items)

    def get(self, idx):
        path, label, feat_path = self.valid_items[idx]
        slide_id = Path(path).stem

        # Load pre-extracted features
        data = torch.load(feat_path, weights_only=False)
        features = data['features']  # (N, 512)
        coords   = data['coords']    # (N, 2)

        # Build K-NN graph
        graph          = build_graph(features, coords, label, self.k)
        graph.slide_id = slide_id
        graph.label_from_folder = label  # sanity check attribute

        return graph


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point: get all three splits as graph datasets
# ─────────────────────────────────────────────────────────────────────────────

def get_graph_datasets(features_dir: Path,
                       normal_dir:   Path,
                       tumor_dir:    Path,
                       k:    int = 8,
                       seed: int = 42) -> tuple:
    """
    Returns (train_dataset, val_dataset, test_dataset) ready for training.

    Usage:
        train_ds, val_ds, test_ds = get_graph_datasets(
            features_dir=Path('./data/features'),
            normal_dir=Path('./data/camelyon16/normal'),
            tumor_dir=Path('./data/camelyon16/tumor'),
        )
    """
    print('Building CAMELYON16 splits...')
    splits = get_splits(normal_dir, tumor_dir, seed=seed)

    train_ds = CAMELYON16GraphDataset(splits['train'], features_dir, k)
    val_ds   = CAMELYON16GraphDataset(splits['val'],   features_dir, k)
    test_ds  = CAMELYON16GraphDataset(splits['test'],  features_dir, k)

    return train_ds, val_ds, test_ds


def get_loaders(features_dir: Path,
                normal_dir:   Path,
                tumor_dir:    Path,
                batch_size:   int = 4,
                k:            int = 8,
                seed:         int = 42,
                num_workers:  int = 0) -> tuple:
    """
    Returns (train_loader, val_loader, test_loader) ready for training.

    batch_size=4 is safe for RTX 5060 8GB with pre-extracted features.

    Usage:
        train_loader, val_loader, test_loader = get_loaders(
            features_dir=Path('./data/features'),
            normal_dir=Path('./data/camelyon16/normal'),
            tumor_dir=Path('./data/camelyon16/tumor'),
            batch_size=4,
        )
        for batch in train_loader:
            logits, attn = model(batch.to(device))
    """
    train_ds, val_ds, test_ds = get_graph_datasets(
        features_dir, normal_dir, tumor_dir, k, seed
    )

    train_loader = PyGDataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True,  num_workers=num_workers
    )
    val_loader   = PyGDataLoader(
        val_ds,   batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )
    test_loader  = PyGDataLoader(
        test_ds,  batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    print(f'\nDataLoaders ready:')
    print(f'  Train: {len(train_ds)} slides → {len(train_loader)} batches')
    print(f'  Val:   {len(val_ds)} slides → {len(val_loader)} batches')
    print(f'  Test:  {len(test_ds)} slides → {len(test_loader)} batches')

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helper (run once in Week 2)
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(normal_dir:   Path,
                         tumor_dir:    Path,
                         features_dir: Path,
                         patches_dir:  Path,
                         device,
                         batch_size:   int = 128):
    """
    Runs ResNet-50 feature extraction on all slides.
    Saves {slide_id}_features.pt to features_dir.

    This is the VRAM-saving trick:
    - You run this ONCE
    - During training you load 512-dim vectors, NOT raw images
    - VRAM drops from ~10 GB to ~5 GB

    Usage (in Week 2 notebook):
        from pathq.dataset import extract_all_features
        extract_all_features(
            normal_dir=Path('./data/camelyon16/normal'),
            tumor_dir=Path('./data/camelyon16/tumor'),
            features_dir=Path('./data/features'),
            patches_dir=Path('./data/patches'),
            device=torch.device('cuda'),
        )
    """
    import timm
    import pickle
    import torch.nn as nn
    from torchvision import transforms
    from tqdm import tqdm

    features_dir.mkdir(exist_ok=True)

    # Build ResNet-50 extractor (frozen)
    backbone = timm.create_model('resnet50', pretrained=True)
    extractor = nn.Sequential(
        backbone.conv1, backbone.bn1, backbone.act1, backbone.maxpool,
        backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        backbone.global_pool,
    ).to(device).eval()

    for p in extractor.parameters():
        p.requires_grad = False

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Find all processed patch .pkl files
    pkl_files = list(patches_dir.glob('*.pkl'))
    print(f'Found {len(pkl_files)} processed slides in {patches_dir}')

    for pkl_path in tqdm(pkl_files, desc='Extracting features'):
        slide_id  = pkl_path.stem
        save_path = features_dir / f'{slide_id}_features.pt'

        if save_path.exists():
            continue  # skip already extracted

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        patches = data['patches']   # list of PIL Images
        coords  = data['coords']    # list of (col, row)

        all_features = []
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch = torch.stack(
                    [transform(p) for p in patches[i:i+batch_size]]
                ).to(device)
                feats = extractor(batch)          # (B, 512)
                all_features.append(feats.cpu())

        features_tensor = torch.cat(all_features, dim=0)   # (N, 512)
        coords_tensor   = torch.tensor(coords, dtype=torch.float32)

        torch.save({
            'slide_id':  slide_id,
            'features':  features_tensor,
            'coords':    coords_tensor,
            'n_patches': len(patches),
        }, save_path)

    print(f'Feature extraction complete. Saved to {features_dir}')


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    # Try both locations (for notebook and root directory)
    if Path('./data/camelyon16/normal').exists():
        normal_dir   = Path('./data/camelyon16/normal')
        tumor_dir    = Path('./data/camelyon16/tumor')
        features_dir = Path('./data/features')
    else:
        normal_dir   = Path('./notebooks/data/camelyon16/normal')
        tumor_dir    = Path('./notebooks/data/camelyon16/tumor')
        features_dir = Path('./notebooks/data/features')

    # Check folders exist
    if not normal_dir.exists() or not tumor_dir.exists():
        print(f'Slide folders not found.')
        print(f'  Expected: {normal_dir.resolve()}')
        print(f'  Expected: {tumor_dir.resolve()}')
        sys.exit(1)

    # Show what we have
    normal_count = len(list(normal_dir.glob('*.tif')))
    tumor_count  = len(list(tumor_dir.glob('*.tif')))
    print(f'Slides on disk: {normal_count} normal, {tumor_count} tumor')
    print()

    # Test the split
    splits = get_splits(normal_dir, tumor_dir)
    print()

    # Test raw dataset
    raw = CAMELYON16Raw(normal_dir, tumor_dir, split='train')
    path, label = raw[0]
    print(f'First train slide: {path.name}  label={label}')

    # Test graph dataset (only if features exist)
    feat_files = list(features_dir.glob('*_features.pt'))
    if feat_files:
        train_ds, val_ds, test_ds = get_graph_datasets(
            features_dir, normal_dir, tumor_dir
        )
        g = train_ds[0]
        print(f'Graph: {g.num_nodes} nodes, {g.num_edges} edges, '
              f'label={g.y.item()}, slide={g.slide_id}')
    else:
        print(f'No .pt feature files in {features_dir} yet.')
        print('Run extract_all_features() after Week 2 patch extraction.')