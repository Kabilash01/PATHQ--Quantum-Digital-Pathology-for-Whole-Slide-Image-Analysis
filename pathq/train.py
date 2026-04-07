"""
pathq/train.py
==============
Training loop for PATHQ.

Usage:
    python train.py --mode classical    # train GNN baseline (no VQC)
    python train.py --mode quantum      # train full VQC+GNN model
    python train.py --mode compare      # train both and compare
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import wandb
from tqdm import tqdm

# Import PATHQ modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from pathq.model import PATHQModel


def train_one_epoch(model, loader, optimizer, device, epoch):
    """One training epoch. Returns average loss."""
    model.train()
    total_loss = 0
    n_batches  = 0
    
    for batch in tqdm(loader, desc=f'Epoch {epoch} [train]', leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        logits, _ = model(batch)
        labels    = batch.y.squeeze()
        
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        
        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, split_name='val'):
    """
    Evaluate model on a dataloader.
    Returns dict with loss, AUC, F1, sensitivity, specificity.
    """
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0
    
    for batch in tqdm(loader, desc=f'[{split_name}]', leave=False):
        batch  = batch.to(device)
        logits, _ = model(batch)
        labels = batch.y.squeeze()
        
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item()
        
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)  # (N, 2)
    all_labels = torch.cat(all_labels, dim=0)  # (N,)
    
    probs = torch.softmax(all_logits, dim=1)[:, 1].numpy()  # P(tumour)
    preds = all_logits.argmax(dim=1).numpy()
    labels_np = all_labels.numpy()
    
    # Compute metrics
    try:
        auc = roc_auc_score(labels_np, probs)
    except Exception:
        auc = 0.5
    
    f1  = f1_score(labels_np, preds, zero_division=0)
    
    # Sensitivity and specificity
    tp = ((preds == 1) & (labels_np == 1)).sum()
    tn = ((preds == 0) & (labels_np == 0)).sum()
    fp = ((preds == 1) & (labels_np == 0)).sum()
    fn = ((preds == 0) & (labels_np == 1)).sum()
    
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    
    return {
        'loss':        total_loss / max(len(loader), 1),
        'auc':         auc,
        'f1':          f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }


def train_pathq(
    features_dir: Path,
    mode: str         = 'quantum',   # 'classical' or 'quantum'
    n_epochs: int     = 50,
    batch_size: int   = 4,
    lr: float         = 1e-4,
    n_qubits: int     = 3,
    vqc_layers: int   = 2,
    save_dir: Path    = Path('./checkpoints'),
    use_wandb: bool   = True,
    project_name: str = 'PATHQ',
    max_slides: int   = None,
):
    """
    Main training function.
    
    Args:
        features_dir: Directory containing *_features.pt files
        mode:         'classical' (no VQC) or 'quantum' (with VQC)
        n_epochs:     Number of training epochs
        batch_size:   Slides per batch (4 is safe for 8GB VRAM)
        lr:           Learning rate
        n_qubits:     Qubits in VQC (3 recommended)
        vqc_layers:   VQC circuit depth (2 recommended)
        save_dir:     Where to save checkpoints
        use_wandb:    Log to Weights & Biases
        max_slides:   Cap dataset size (None = use all)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir.mkdir(exist_ok=True)
    
    print(f'PATHQ Training')
    print(f'  Mode:    {mode}')
    print(f'  Device:  {device}')
    print(f'  Epochs:  {n_epochs}')
    print('='*50)
    
    # ── Dataset ──────────────────────────────────────────────────────────────
    # Import here to avoid circular deps
    from week2_feature_extraction import CAMELYON16Dataset
    
    train_dataset = CAMELYON16Dataset(features_dir, split='train', k=8,
                                       max_slides=max_slides)
    val_dataset   = CAMELYON16Dataset(features_dir, split='val',   k=8,
                                       max_slides=max_slides)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=0)
    
    print(f'Train: {len(train_dataset)} slides')
    print(f'Val:   {len(val_dataset)} slides')
    
    # ── Model ─────────────────────────────────────────────────────────────────
    use_vqc = (mode == 'quantum')
    model = PATHQModel(
        feature_dim=512,
        gnn_hidden=256,
        n_qubits=n_qubits,
        vqc_layers=vqc_layers,
        use_vqc=use_vqc,
    ).to(device)
    
    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )
    
    # ── WandB ─────────────────────────────────────────────────────────────────
    if use_wandb:
        run = wandb.init(
            project=project_name,
            name=f'pathq_{mode}_q{n_qubits}l{vqc_layers}',
            config={
                'mode': mode, 'epochs': n_epochs, 'batch_size': batch_size,
                'lr': lr, 'n_qubits': n_qubits, 'vqc_layers': vqc_layers,
            }
        )
    
    # ── Training loop ─────────────────────────────────────────────────────────
    best_auc  = 0.0
    best_path = save_dir / f'pathq_{mode}_best.pth'
    
    for epoch in range(1, n_epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, 'val')
        
        # Scheduler step
        scheduler.step()
        
        # Log
        log_dict = {
            'epoch':            epoch,
            'train/loss':       train_loss,
            'val/loss':         val_metrics['loss'],
            'val/auc':          val_metrics['auc'],
            'val/f1':           val_metrics['f1'],
            'val/sensitivity':  val_metrics['sensitivity'],
            'val/specificity':  val_metrics['specificity'],
            'lr':               scheduler.get_last_lr()[0],
        }
        
        print(f"Epoch {epoch:3d}/{n_epochs}  "
              f"loss={train_loss:.4f}  "
              f"val_auc={val_metrics['auc']:.4f}  "
              f"val_f1={val_metrics['f1']:.4f}")
        
        if use_wandb:
            wandb.log(log_dict)
        
        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'val_auc':     best_auc,
                'config': {
                    'mode': mode, 'n_qubits': n_qubits,
                    'vqc_layers': vqc_layers,
                },
            }, best_path)
            print(f'  >>> New best AUC: {best_auc:.4f} — saved to {best_path.name}')
    
    if use_wandb:
        wandb.finish()
    
    print(f'\nTraining complete. Best val AUC: {best_auc:.4f}')
    return best_auc, best_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PATHQ model')
    parser.add_argument('--mode',        default='quantum',
                        choices=['classical', 'quantum'])
    parser.add_argument('--features_dir', default='./data/features')
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--batch_size',  type=int,   default=4)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--n_qubits',    type=int,   default=3)
    parser.add_argument('--vqc_layers',  type=int,   default=2)
    parser.add_argument('--no_wandb',    action='store_true')
    parser.add_argument('--max_slides',  type=int,   default=None)
    
    args = parser.parse_args()
    
    train_pathq(
        features_dir = Path(args.features_dir),
        mode         = args.mode,
        n_epochs     = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        n_qubits     = args.n_qubits,
        vqc_layers   = args.vqc_layers,
        use_wandb    = not args.no_wandb,
        max_slides   = args.max_slides,
    )
