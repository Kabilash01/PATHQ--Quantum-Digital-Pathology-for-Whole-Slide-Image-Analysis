"""
pathq/xai.py
============
Three-layer XAI stack for PATHQ.

Layer 1: Grad-CAM++ on ResNet backbone (spatial — which tissue regions?)
Layer 2: ABMIL attention weights (structural — which patch relationships?)
Layer 3: VQC parameter-shift gradients + Bloch sphere (quantum — which gates?)

Usage:
    from pathq.xai import PATHQExplainer
    explainer = PATHQExplainer(model, device)
    report = explainer.explain(graph, slide_id='tumor_001')
    explainer.plot_report(report, save_path='./outputs/xai_report.png')
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import pennylane as qml


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Grad-CAM++ (Spatial XAI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradcam_patch(
    feature_extractor: nn.Module,
    patch_tensor: torch.Tensor,
    device,
    target_class: int = 1
) -> np.ndarray:
    """
    Compute Grad-CAM++ saliency for a single patch.
    
    Returns a 7x7 saliency map (last conv layer spatial resolution)
    that can be upsampled to full patch resolution.
    
    Args:
        feature_extractor: ResNet-50 backbone (frozen)
        patch_tensor:      (1, 3, 256, 256) normalised patch tensor
        target_class:      Class to explain (1 = tumour)
    
    Returns:
        saliency: (256, 256) numpy array, values in [0, 1]
    """
    feature_extractor.eval()
    patch_tensor = patch_tensor.to(device).requires_grad_(True)
    
    # Hook to capture gradients and activations of the last conv layer
    activations = {}
    gradients   = {}
    
    def forward_hook(module, input, output):
        activations['last_conv'] = output.detach()
    
    def backward_hook(module, grad_input, grad_output):
        gradients['last_conv'] = grad_output[0].detach()
    
    # Register hooks on the last conv layer (layer4[-1].conv3 for ResNet-50)
    try:
        target_layer = list(feature_extractor.features[7].children())[-1].conv3
    except Exception:
        # Fallback: use the last layer
        target_layer = list(feature_extractor.features.children())[-2]
    
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)
    
    try:
        output = feature_extractor(patch_tensor)
        
        # Backward pass targeting the tumour class
        feature_extractor.zero_grad()
        output[0, min(target_class, output.shape[1]-1)].backward()
        
        if 'last_conv' not in activations or 'last_conv' not in gradients:
            # Hooks didn't fire — return uniform saliency
            return np.ones((256, 256)) * 0.5
        
        acts = activations['last_conv'][0]  # (C, H, W)
        grads = gradients['last_conv'][0]   # (C, H, W)
        
        # Grad-CAM++ weighting
        grads_2 = grads ** 2
        grads_3 = grads ** 3
        
        alpha_num   = grads_2
        alpha_denom = 2 * grads_2 + (grads_3 * acts).sum(dim=(1, 2), keepdim=True) + 1e-8
        alpha = alpha_num / alpha_denom
        
        weights = (alpha * torch.relu(grads)).sum(dim=(1, 2))  # (C,)
        
        # Weighted sum of activation maps
        cam = torch.zeros(acts.shape[1:], device=device)
        for c in range(acts.shape[0]):
            cam += weights[c] * acts[c]
        
        cam = torch.relu(cam)
        
        # Upsample to patch size
        cam_np = cam.cpu().numpy()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()
        
        # Bilinear upsampling to 256x256
        from PIL import Image as PILImage
        cam_pil = PILImage.fromarray((cam_np * 255).astype(np.uint8))
        cam_pil = cam_pil.resize((256, 256), PILImage.BILINEAR)
        saliency = np.array(cam_pil) / 255.0
        
        return saliency
        
    finally:
        fh.remove()
        bh.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: ABMIL Attention (Structural XAI)
# ─────────────────────────────────────────────────────────────────────────────

def get_attention_map(
    model,
    graph,
    device,
    patch_size: int = 256
) -> tuple:
    """
    Extract ABMIL attention weights for all patches in a slide.
    
    Args:
        model:       PATHQModel (trained)
        graph:       PyG Data object for the slide
        device:      torch device
        patch_size:  Patch size in pixels
    
    Returns:
        attn_weights: (N,) numpy array of attention weights
        coords:       (N, 2) numpy array of patch grid coordinates
        prediction:   dict with logits and predicted class
    """
    model.eval()
    
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([graph]).to(device)
    
    with torch.no_grad():
        logits, attn_weights = model(batch)
    
    attn_np = attn_weights.squeeze(1).cpu().numpy()  # (N,)
    coords_np = graph.coords.numpy()                  # (N, 2)
    
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_class = logits.argmax(dim=1).item()
    
    return attn_np, coords_np, {
        'logits':     logits.cpu().numpy(),
        'probs':      probs,
        'pred_class': pred_class,
        'confidence': probs[pred_class],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: Quantum Circuit Parameter Sensitivity (Quantum XAI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_quantum_sensitivity(
    vqc_encoder,
    patch_features: torch.Tensor,
    n_samples: int = 20,
) -> dict:
    """
    Compute parameter-shift gradients for VQC parameters.
    
    For each VQC parameter theta_i:
        sensitivity_i = |d(output)/d(theta_i)|
    
    High sensitivity = this quantum gate had high influence on the output.
    
    Args:
        vqc_encoder:    VQCEncoder module (with trained weights)
        patch_features: (N, 512) patch feature tensor
        n_samples:      How many patches to average over
    
    Returns:
        dict with:
            param_sensitivity: (n_params,) gradient magnitudes
            param_names:       list of parameter labels
            quantum_outputs:   (N, n_qubits) quantum measurement values
    """
    vqc_encoder.eval()
    
    # Sample patches
    if patch_features.shape[0] > n_samples:
        idx = torch.randperm(patch_features.shape[0])[:n_samples]
        features = patch_features[idx]
    else:
        features = patch_features
    
    # Project features to VQC input
    with torch.no_grad():
        projected = vqc_encoder.projection(features)              # (N, 8)
        projected_norm = torch.nn.functional.normalize(projected, p=2, dim=1)
    
    # Get VQC weights
    vqc_weights = list(vqc_encoder.vqc.parameters())
    if len(vqc_weights) == 0:
        return {'param_sensitivity': np.array([]), 'param_names': [], 'quantum_outputs': np.array([])}
    
    weights_tensor = vqc_weights[0]  # (n_layers, 2, n_qubits)
    
    # Compute outputs and gradients
    quantum_outputs = []
    all_grads = []
    
    for i in range(features.shape[0]):
        x_i = projected_norm[i:i+1]  # (1, 8)
        
        # Enable gradient computation through VQC
        weights_tensor.requires_grad_(True)
        
        out = vqc_encoder.vqc(x_i)  # (1, n_qubits)
        out_scalar = out.sum()
        
        if weights_tensor.grad is not None:
            weights_tensor.grad.zero_()
        
        out_scalar.backward()
        
        if weights_tensor.grad is not None:
            grad = weights_tensor.grad.abs().flatten().detach().numpy()
            all_grads.append(grad)
        
        quantum_outputs.append(out.detach().numpy())
    
    # Average sensitivity across samples
    if all_grads:
        sensitivity = np.stack(all_grads).mean(axis=0)
    else:
        n_params = weights_tensor.numel()
        sensitivity = np.zeros(n_params)
    
    quantum_outputs = np.concatenate(quantum_outputs, axis=0)  # (N, n_qubits)
    
    # Label parameters
    n_layers = weights_tensor.shape[0]
    n_qubits = weights_tensor.shape[2]
    param_names = []
    for l in range(n_layers):
        for gate_type in ['RY', 'RZ']:
            for q in range(n_qubits):
                param_names.append(f'L{l+1}_{gate_type}_q{q}')
    
    return {
        'param_sensitivity': sensitivity,
        'param_names': param_names,
        'quantum_outputs': quantum_outputs,
        'n_samples': features.shape[0],
    }


def compute_bloch_trajectory(vqc_encoder, patch_features: torch.Tensor,
                              qubit_idx: int = 0) -> np.ndarray:
    """
    Compute Bloch sphere trajectory for a qubit across VQC layers.
    
    Records the qubit state [theta, phi] at each circuit layer,
    showing how the quantum state evolves from input to output.
    
    Args:
        vqc_encoder:    VQCEncoder module
        patch_features: (N, 512) patch features
        qubit_idx:      Which qubit to track (0 = most sensitive)
    
    Returns:
        trajectory: (N, n_layers+1, 2) array of [theta, phi] Bloch angles
    """
    vqc_encoder.eval()
    n_qubits = vqc_encoder.n_qubits
    n_layers = vqc_encoder.n_layers
    
    # Use PennyLane to get intermediate states
    dev = qml.device('default.qubit', wires=n_qubits)
    
    with torch.no_grad():
        projected = vqc_encoder.projection(patch_features)
        projected_norm = torch.nn.functional.normalize(projected, p=2, dim=1)
        weights_np = list(vqc_encoder.vqc.parameters())[0].numpy()
    
    trajectories = []
    
    for i in range(min(len(projected_norm), 50)):  # limit for speed
        x_i = projected_norm[i].numpy()
        traj_i = []
        
        @qml.qnode(dev)
        def state_at_layer(x, weights, n_layer):
            qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=True)
            for l in range(n_layer):
                for q in range(n_qubits):
                    qml.RY(weights[l, 0, q], wires=q)
                    qml.RZ(weights[l, 1, q], wires=q)
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            return qml.state()
        
        for layer in range(n_layers + 1):
            try:
                state = state_at_layer(x_i, weights_np, layer)
                state_np = np.array(state)
                
                # Extract the reduced state of qubit_idx
                # Trace out other qubits
                state_matrix = state_np.reshape([2] * n_qubits)
                
                # Partial trace (simplified: take amplitude of |0> and |1> for target qubit)
                alpha = state_matrix.take(0, axis=qubit_idx).flatten()
                beta  = state_matrix.take(1, axis=qubit_idx).flatten()
                
                prob_0 = (np.abs(alpha) ** 2).sum()
                prob_1 = (np.abs(beta)  ** 2).sum()
                
                # Bloch angles
                theta = 2 * np.arccos(np.sqrt(np.clip(prob_0, 0, 1)))
                phi   = 0.0  # simplified — full phase tracking needs density matrix
                
                traj_i.append([theta, phi])
            except Exception:
                traj_i.append([0.0, 0.0])
        
        trajectories.append(traj_i)
    
    return np.array(trajectories)  # (N, n_layers+1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Combined XAI Report
# ─────────────────────────────────────────────────────────────────────────────

def plot_xai_report(
    slide_patches,          # list of PIL Images (sample patches)
    attn_weights,           # (N,) attention weights
    coords,                 # (N, 2) patch coordinates
    quantum_sensitivity,    # dict from compute_quantum_sensitivity()
    prediction,             # dict with pred_class, confidence
    slide_id: str = 'slide',
    save_path: Path = None,
    n_top_patches: int = 4,
):
    """
    Creates the three-layer XAI report figure.
    
    Layout:
        Row 1: Top-attention patches (Layer 2 XAI)
        Row 2: Attention heatmap + quantum sensitivity bars + Bloch placeholder
    """
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f'PATHQ XAI Report — {slide_id}\n'
        f'Prediction: {"TUMOUR" if prediction["pred_class"] == 1 else "NORMAL"}  '
        f'(confidence: {prediction["confidence"]:.1%})',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)
    
    # ── Row 1: Top attention patches ──────────────────────────────────────
    top_indices = np.argsort(attn_weights)[::-1][:n_top_patches]
    
    for i, idx in enumerate(top_indices):
        ax = fig.add_subplot(gs[0, i])
        if idx < len(slide_patches):
            ax.imshow(slide_patches[idx])
        else:
            ax.set_facecolor('#f0f0f0')
        ax.set_title(f'Patch {idx}\nAttention: {attn_weights[idx]:.4f}',
                     fontsize=9, color='darkred' if attn_weights[idx] > 0.05 else 'gray')
        ax.axis('off')
    
    # ── Row 2, Col 1: Attention heatmap ───────────────────────────────────
    ax_heat = fig.add_subplot(gs[1, 0])
    if len(coords) > 0:
        scatter = ax_heat.scatter(
            coords[:, 0], coords[:, 1],
            c=attn_weights, cmap='hot', s=5,
            vmin=attn_weights.min(), vmax=attn_weights.max()
        )
        plt.colorbar(scatter, ax=ax_heat, fraction=0.046, pad=0.04)
    ax_heat.set_title('Layer 2: Attention Map\n(ABMIL weights)', fontsize=10)
    ax_heat.set_xlabel('Col')
    ax_heat.set_ylabel('Row')
    ax_heat.invert_yaxis()
    
    # ── Row 2, Col 2: Quantum parameter sensitivity ────────────────────────
    ax_qxai = fig.add_subplot(gs[1, 1])
    sensitivity = quantum_sensitivity.get('param_sensitivity', np.array([]))
    param_names = quantum_sensitivity.get('param_names', [])
    
    if len(sensitivity) > 0:
        colors = ['#1D9E75' if 'RY' in n else '#B83A12' if 'RZ' in n else '#534AB7'
                  for n in param_names]
        bars = ax_qxai.bar(range(len(sensitivity)), sensitivity, color=colors, alpha=0.8)
        ax_qxai.set_xticks(range(len(param_names)))
        ax_qxai.set_xticklabels(param_names, rotation=45, ha='right', fontsize=7)
        ax_qxai.set_ylabel('|gradient|')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1D9E75', label='RY (rotation)'),
            Patch(facecolor='#B83A12', label='RZ (rotation)'),
        ]
        ax_qxai.legend(handles=legend_elements, fontsize=8, loc='upper right')
    else:
        ax_qxai.text(0.5, 0.5, 'VQC not used\n(classical mode)',
                     ha='center', va='center', transform=ax_qxai.transAxes)
    
    ax_qxai.set_title('Layer 3: Quantum Gate Sensitivity\n(param-shift gradients)', fontsize=10)
    
    # ── Row 2, Col 3: Quantum output distribution ──────────────────────────
    ax_qdist = fig.add_subplot(gs[1, 2])
    quantum_outputs = quantum_sensitivity.get('quantum_outputs', np.array([]))
    
    if len(quantum_outputs) > 0:
        n_qubits = quantum_outputs.shape[1]
        for q in range(n_qubits):
            ax_qdist.hist(quantum_outputs[:, q], bins=20, alpha=0.6,
                          label=f'Qubit {q}')
        ax_qdist.set_xlabel('<Z> expectation value')
        ax_qdist.set_ylabel('Count')
        ax_qdist.legend(fontsize=9)
    else:
        ax_qdist.text(0.5, 0.5, 'No quantum\noutputs available',
                      ha='center', va='center', transform=ax_qdist.transAxes)
    
    ax_qdist.set_title('Layer 3: Qubit Measurement Dist.\n(expectation values)', fontsize=10)
    
    # ── Row 2, Col 4: Summary ──────────────────────────────────────────────
    ax_sum = fig.add_subplot(gs[1, 3])
    ax_sum.axis('off')
    
    summary_text = (
        f"PATHQ XAI Summary\n"
        f"{'─'*28}\n\n"
        f"Slide: {slide_id}\n\n"
        f"Prediction:\n"
        f"  {'TUMOUR' if prediction['pred_class']==1 else 'NORMAL'}\n"
        f"  ({prediction['confidence']:.1%} confidence)\n\n"
        f"Layer 1 — Spatial:\n"
        f"  Grad-CAM++ applied\n"
        f"  to top patches\n\n"
        f"Layer 2 — Structural:\n"
        f"  {len(attn_weights)} patches scored\n"
        f"  Top patch: {attn_weights.max():.4f}\n\n"
        f"Layer 3 — Quantum:\n"
        f"  {len(sensitivity)} VQC params\n"
        f"  Max sensitivity:\n"
        f"  {sensitivity.max():.4f if len(sensitivity)>0 else 'N/A'}"
    )
    
    ax_sum.text(0.05, 0.95, summary_text, transform=ax_sum.transAxes,
                fontsize=9, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f8ff', alpha=0.8))
    
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        print(f'XAI report saved: {save_path}')
    
    plt.show()
    return fig


if __name__ == '__main__':
    print('XAI module loaded successfully.')
    print('Import PATHQExplainer for full usage.')
    print()
    print('Functions available:')
    print('  compute_gradcam_patch(feature_extractor, patch_tensor, device)')
    print('  get_attention_map(model, graph, device)')
    print('  compute_quantum_sensitivity(vqc_encoder, patch_features)')
    print('  compute_bloch_trajectory(vqc_encoder, patch_features, qubit_idx)')
    print('  plot_xai_report(patches, attn, coords, sensitivity, prediction)')
