"""Spatial map of per-pixel MC-DDIM predictions for VS-Diff, the
diffusion-model counterpart to pix2pix's spatial_map_sample_1.png (see the
commented-out cell in pix2pix_minimal's pix2pix.ipynb). Unlike the
whole-val-set UMAP (full_umap.py), every pixel here stays at its real (x, y)
position, so the panels keep the same silhouette as the actual tissue patch,
just recoloured by prediction, tissue class, uncertainty, or confidence.

Single patch only, not the full 7373-image val set. That's what makes
MC-DDIM tractable here, since each sample is a full 50-step DDIM pass rather
than pix2pix's near-instant dropout forward pass. See Section
methods-uncertainty / res-uq for why this wasn't extended to the full val
set for the diffusion model.

TARGET_IDX=5914 is deliberately the same physical patch as pix2pix's "Sample
1" (datasets/polyps_v7/val/5914.tif; VirtualStainingDataset sorts by
int(filename), so index == filename here), for a direct visual comparison
between the two models' uncertainty structure on the same tissue.

Usable two ways, mirroring full_umap.py / full_uncertainty.py:
  - from a notebook cell that already has model/scheduler/device/dataset built:
        import spatial_map_ddim as smd
        smd.run_spatial_map(model, inference_noise_scheduler, device, test_dataset)
  - standalone: python spatial_map_ddim.py [N_RUNS]
"""
import os
import sys
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(__file__))
import virt_stain_utils2 as vsu

BASE_DIR = "/cs/student/project_msc/2025/aibh/razarine"
CKPT_PATH = os.path.join(BASE_DIR, "runs/exp35/checkpoints/best.pth")
VAL_DIR = os.path.join(BASE_DIR, "datasets/polyps_v7/val")
OUT_DIR = os.path.join(BASE_DIR, "sweep_output/uncertainty_spatial_map")

STEPS = 50
ETA = 1.0  # stochastic, same convention as full_uncertainty.py / full_umap.py
TARGET_IDX = 5914  # same physical patch as pix2pix's "Sample 1"
N_RUNS_DEFAULT = 20  # matches pix2pix's spatial-map cell

TISSUE_NAMES = ['Nucleus', 'Cytoplasm & stroma', 'Background/lumen']
TISSUE_COLORS = ['#7c1fd6', '#ff6fae', '#9a9a9a']
CONF_NAMES = ['High-confidence', 'Low-confidence']
CONF_COLORS = ['#9ecae1', '#08306b']
COLOUR_NOTE = ('Purple = nucleus (hematoxylin-stained chromatin)   |   '
               'Pink = cytoplasm & stroma (eosin-stained protein/collagen)   |   '
               'Grey/white = background or gland lumen (no tissue)')


def denorm01(x):
    return x.clamp(-1, 1) * 0.5 + 0.5


def classify_tissue(gt01):
    """Rough 3-way H&E tissue labelling from ground-truth colour (brightness
    percentile): darkest 25% is nucleus, lightest 25% is background/lumen,
    rest is cytoplasm/stroma. gt01 is a (H,W,3) numpy array in [0,1]."""
    v = mcolors.rgb_to_hsv(gt01)[..., 2]
    p25, p75 = np.percentile(v, [25, 75])
    labels = np.full(v.shape, 1, dtype=np.int32)
    labels[v <= p25] = 0
    labels[v >= p75] = 2
    return labels


def run_spatial_map(model, scheduler, device, dataset, n_runs=N_RUNS_DEFAULT,
                     target_idx=TARGET_IDX, out_dir=OUT_DIR):
    """Runs n_runs stochastic MC-DDIM samples on dataset[target_idx], then
    plots the 4-panel spatial map: mean brightness, tissue class,
    uncertainty, confidence. Call from a notebook cell that already has
    model/scheduler/device/dataset built (see module docstring)."""
    os.makedirs(out_dir, exist_ok=True)

    phase, gt = dataset[target_idx]
    phase_b = phase.unsqueeze(0).to(device)
    gt01 = denorm01(gt).permute(1, 2, 0).numpy()

    t0 = time.time()
    preds = []
    for i in range(n_runs):
        with torch.no_grad():
            pred = vsu.ddim_sample_full(model, phase_b, scheduler, device,
                                         num_inference_steps=STEPS, eta=ETA)
        preds.append(denorm01(pred[0]).permute(1, 2, 0).cpu().numpy())
        elapsed = time.time() - t0
        print(f'run {i + 1}/{n_runs}  elapsed={elapsed:.1f}s  '
              f'avg/run={elapsed / (i + 1):.1f}s  '
              f'est.remaining={(n_runs - i - 1) * elapsed / (i + 1):.1f}s', flush=True)

    stack01 = np.stack(preds, axis=0)  # (N, H, W, 3)

    labels = classify_tissue(gt01)
    uncertainty = stack01.var(axis=0).mean(axis=-1)          # (H, W) MC-DDIM variance
    mean_v = mcolors.rgb_to_hsv(stack01.mean(axis=0))[..., 2]  # (H, W) mean-prediction brightness

    uncert_median = np.median(uncertainty)
    confidence = (uncertainty >= uncert_median).astype(int)

    print('uncertainty mean:', uncertainty.mean(), 'min:', uncertainty.min(),
          'max:', uncertainty.max(), 'p97:', np.percentile(uncertainty, 97), flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    for ax, lbl in zip(axes.flat, ['a', 'b', 'c', 'd']):
        ax.text(-0.02, 1.03, lbl, transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom')
        ax.axis('off')

    im_a = axes[0, 0].imshow(mean_v, cmap='RdPu_r')
    plt.colorbar(im_a, ax=axes[0, 0], fraction=0.046, pad=0.04, label='Mean-prediction brightness (V)')
    axes[0, 0].set_title('Prediction (mean brightness)', fontsize=12)

    axes[0, 1].imshow(labels, cmap=ListedColormap(TISSUE_COLORS), vmin=0, vmax=2)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=9)
               for c in TISSUE_COLORS]
    axes[0, 1].legend(handles, TISSUE_NAMES, fontsize=9, loc='upper right')
    axes[0, 1].set_title('Tissue class (from ground truth)', fontsize=12)

    vmax = np.percentile(uncertainty, 97)
    im_c = axes[1, 0].imshow(uncertainty, cmap='inferno', vmin=0, vmax=vmax)
    plt.colorbar(im_c, ax=axes[1, 0], fraction=0.046, pad=0.04, label='MC-DDIM variance (clipped at p97)')
    axes[1, 0].set_title('Uncertainty', fontsize=12)

    axes[1, 1].imshow(confidence, cmap=ListedColormap(CONF_COLORS), vmin=0, vmax=1)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=9)
               for c in CONF_COLORS]
    axes[1, 1].legend(handles, CONF_NAMES, fontsize=9, loc='upper right')
    axes[1, 1].set_title('Confidence (median split of uncertainty)', fontsize=12)

    plt.suptitle('Spatial map of per-pixel MC-DDIM predictions - Sample 1', fontsize=13, y=1.0)
    fig.text(0.5, -0.01, COLOUR_NOTE, ha='center', va='top', fontsize=9.5, color='dimgray', wrap=True)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'spatial_map_ddim_sample_1.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print('saved ->', out_path, flush=True)


def main():
    from vsdiff_model import build_model_and_scheduler, load_checkpoint, VirtualStainingDataset
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else N_RUNS_DEFAULT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}', flush=True)
    model, scheduler = build_model_and_scheduler(device)
    load_checkpoint(model, CKPT_PATH, device)
    model.eval()
    dataset = VirtualStainingDataset(VAL_DIR)
    run_spatial_map(model, scheduler, device, dataset, n_runs=n_runs)


if __name__ == "__main__":
    main()
