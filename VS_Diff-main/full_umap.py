"""UMAP visualization of VS-Diff's predictive uncertainty structure, mirroring
umap_uncertainty.py on the Pix2Pix side, but sourced from MC-DDIM stochastic
sampling (eta=1.0, see full_uncertainty.py) instead of MC-Dropout. Three parts:
  1. Per-sample 4-panel UMAP for a few fixed val images.
  2. Whole-val-set UMAP (pooled across all val images, subsampled per image).
  3. Point-count vs. embedding-spread summary per tissue class, reusing part 2's cache.

Caches at every stage: per-sample embeddings, whole-val-set per-image
extracted features (resumable, one file per image, so an interrupted part-2
run picks up where it left off), and the whole-val-set fitted embedding
itself. Re-running any part to tweak only the plot (colors, titles, layout)
loads straight from the relevant cache instead of re-sampling or re-fitting.

Needs `umap-learn` (pip install umap-learn) in vsdiff_env, not installed by
default. Not launched automatically anywhere; call the run_* functions from a
notebook cell, or run this file directly:
    vsdiff_env/bin/python full_umap.py [1|2|3|all]
"""
import os
import sys
import time
import pathlib

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(__file__))
import virt_stain_utils2 as vsu
from vsdiff_model import build_model_and_scheduler, load_checkpoint, VirtualStainingDataset

BASE_DIR = "/cs/student/project_msc/2025/aibh/razarine"
CKPT_PATH = os.path.join(BASE_DIR, "runs/exp35/checkpoints/best.pth")
VAL_DIR = os.path.join(BASE_DIR, "datasets/polyps_v7/val")
OUT_DIR = os.path.join(BASE_DIR, "sweep_output/umap")
STEPS = 50
ETA = 1.0  # stochastic, same convention as full_uncertainty.py's mc_uncertainty

TISSUE_NAMES = ['Nucleus', 'Cytoplasm & stroma', 'Background/lumen']
TISSUE_COLORS = ['#7c1fd6', '#ff6fae', '#9a9a9a']
CONF_NAMES = ['High-confidence', 'Low-confidence']
CONF_COLORS = ['#9ecae1', '#08306b']
COLOUR_NOTE = ('H&E colour key  ->  purple = nucleus (hematoxylin-stained chromatin)   |   '
               'pink = cytoplasm & stroma (eosin-stained protein/collagen)   |   '
               'grey/white = background or gland lumen (no tissue)')


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


def mc_ddim_stack(model, phase, scheduler, device, n_runs, steps=STEPS, eta=ETA):
    """N independent stochastic DDIM samples for one image, the VS-Diff
    analogue of mc_dropout_stack on the Pix2Pix side. Returns an
    (N, H, W, 3) numpy array in [0, 1]."""
    preds = []
    for _ in range(n_runs):
        with torch.no_grad():
            pred = vsu.ddim_sample_full(model, phase, scheduler, device,
                                         num_inference_steps=steps, eta=eta)
        preds.append(denorm01(pred[0]).permute(1, 2, 0).cpu().numpy())
    return np.stack(preds, axis=0)  # (N, H, W, 3)


def _plot_4panel(embedding, y_meanv, y_tissue, y_uncert, title, out_path):
    uncert_median = np.median(y_uncert)
    y_confidence = (y_uncert >= uncert_median).astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(15, 14))
    for ax, lbl in zip(axes.flat, ['a', 'b', 'c', 'd']):
        ax.text(-0.02, 1.03, lbl, transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom')
        ax.set_xticks([]); ax.set_yticks([])

    sc_a = axes[0, 0].scatter(embedding[:, 0], embedding[:, 1], s=6, c=y_meanv,
                               cmap='RdPu_r', alpha=0.7, edgecolors='none')
    plt.colorbar(sc_a, ax=axes[0, 0], fraction=0.046, pad=0.04, label='Mean-prediction brightness (V)')
    axes[0, 0].set_title('Prediction (mean brightness)', fontsize=12)

    for cls in range(3):
        m = y_tissue == cls
        axes[0, 1].scatter(embedding[m, 0], embedding[m, 1], s=6, alpha=0.6,
                            c=TISSUE_COLORS[cls], label=TISSUE_NAMES[cls], edgecolors='none')
    axes[0, 1].legend(markerscale=4, fontsize=9, loc='upper right')
    axes[0, 1].set_title('Tissue class (from ground truth)', fontsize=12)

    vmax = np.percentile(y_uncert, 97)
    order = np.argsort(y_uncert)
    sc_c = axes[1, 0].scatter(embedding[order, 0], embedding[order, 1], s=6, alpha=0.7,
                               c=y_uncert[order], cmap='inferno', vmin=0, vmax=vmax, edgecolors='none')
    plt.colorbar(sc_c, ax=axes[1, 0], fraction=0.046, pad=0.04, label='MC-DDIM variance (clipped at p97)')
    axes[1, 0].set_title('Uncertainty', fontsize=12)

    for cls in range(2):
        m = y_confidence == cls
        axes[1, 1].scatter(embedding[m, 0], embedding[m, 1], s=6, alpha=0.6,
                            c=CONF_COLORS[cls], label=CONF_NAMES[cls], edgecolors='none')
    axes[1, 1].legend(markerscale=4, fontsize=9, loc='upper right')
    axes[1, 1].set_title('Confidence (median split of uncertainty)', fontsize=12)

    plt.suptitle(title, fontsize=14, y=1.0)
    fig.text(0.5, -0.01, COLOUR_NOTE, ha='center', va='top', fontsize=9.5, color='dimgray', wrap=True)
    plt.tight_layout()
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()


def run_umap_per_sample(model, scheduler, device, dataset, sample_indices=(0, 1, 2),
                         stride=2, n_runs=8):
    """Part 1: per-sample 4-panel UMAP. Each sample's fitted embedding is
    cached, so re-running to tweak only the plot skips MC-DDIM + UMAP fit."""
    import umap
    os.makedirs(OUT_DIR, exist_ok=True)

    for idx in sample_indices:
        cache_path = os.path.join(OUT_DIR, f"umap_sample_{idx}_embedding.npz")

        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            embedding, y_tissue, y_uncert, y_meanv = (
                cached["embedding"], cached["y_tissue"], cached["y_uncert"], cached["y_meanv"])
            print(f"sample {idx}: loaded cached embedding ({embedding.shape[0]} points), "
                  f"skipping MC-DDIM + UMAP fit", flush=True)
        else:
            phase, gt = dataset[idx]
            phase_b = phase.unsqueeze(0).to(device)
            gt01 = denorm01(gt).permute(1, 2, 0).numpy()

            stack01 = mc_ddim_stack(model, phase_b, scheduler, device, n_runs)  # (N, H, W, 3)

            labels = classify_tissue(gt01)
            uncertainty = stack01.var(axis=0).mean(axis=-1)
            mean_v = mcolors.rgb_to_hsv(stack01.mean(axis=0))[..., 2]

            sub = stack01[:, ::stride, ::stride, :]
            n, h, w, c = sub.shape
            X = sub.transpose(1, 2, 0, 3).reshape(h * w, n * c)
            y_tissue = labels[::stride, ::stride].reshape(-1)
            y_uncert = uncertainty[::stride, ::stride].reshape(-1)
            y_meanv = mean_v[::stride, ::stride].reshape(-1)

            print(f"sample {idx}: {X.shape[0]} points, feature dim {X.shape[1]}", flush=True)
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42, verbose=False)
            embedding = reducer.fit_transform(X)

            np.savez(cache_path, embedding=embedding, y_tissue=y_tissue, y_uncert=y_uncert, y_meanv=y_meanv)
            print(f"sample {idx}: cached embedding to {cache_path}", flush=True)

        _plot_4panel(embedding, y_meanv, y_tissue, y_uncert,
                     f"UMAP embedding of per-pixel MC-DDIM predictions - Sample {idx}",
                     os.path.join(OUT_DIR, f"umap_sample_{idx}.png"))


def run_umap_valset(model, scheduler, device, dataset, n_runs_all=4, points_per_image=150,
                     max_umap_points=400_000):
    """Part 2: whole-val-set UMAP. Resumable per-image feature extraction
    (valset_feats/<image_idx>.npz, so an interrupted run picks up where it
    left off), then a cached whole-set embedding (umap_valset_embedding.npz)
    once fit. Re-running to tweak only the plot loads the embedding straight
    from disk instead of re-extracting or re-fitting.

    n_runs_all=4 keeps the cost down: each MC-DDIM sample is a full 50-step
    DDIM pass (~0.7s measured), so 7373 images x 4 runs is already ~5.7 hours,
    versus Pix2Pix's near-instant dropout forward passes. Raise it only if
    you have the time budget."""
    import umap
    os.makedirs(OUT_DIR, exist_ok=True)
    feats_dir = os.path.join(OUT_DIR, "valset_feats")
    os.makedirs(feats_dir, exist_ok=True)
    embedding_cache_path = os.path.join(OUT_DIR, "umap_valset_embedding.npz")

    if os.path.exists(embedding_cache_path):
        cached = np.load(embedding_cache_path)
        embedding, y_tissue, y_uncert, y_meanv, sample_id = (
            cached["embedding"], cached["y_tissue"], cached["y_uncert"],
            cached["y_meanv"], cached["sample_id"])
        n_images = int(cached["n_images"])
        print(f"loaded cached embedding: {n_images} images, {embedding.shape[0]} points, "
              f"skipping feature extraction and UMAP fit entirely", flush=True)
    else:
        n_total = len(dataset)
        rng = np.random.default_rng(42)
        t0 = time.time()
        n_done_now = 0
        for i in range(n_total):
            img_cache = os.path.join(feats_dir, f"{i}.npz")
            if os.path.exists(img_cache):
                continue

            phase, gt = dataset[i]
            phase_b = phase.unsqueeze(0).to(device)
            gt01 = denorm01(gt).permute(1, 2, 0).numpy()

            stack01 = mc_ddim_stack(model, phase_b, scheduler, device, n_runs_all)
            labels = classify_tissue(gt01)
            uncertainty = stack01.var(axis=0).mean(axis=-1)
            mean_v = mcolors.rgb_to_hsv(stack01.mean(axis=0))[..., 2]

            h, w = uncertainty.shape
            n_pix = h * w
            n_take = min(points_per_image, n_pix)
            pix_idx = rng.choice(n_pix, size=n_take, replace=False)

            X_img = stack01.transpose(1, 2, 0, 3).reshape(n_pix, n_runs_all * 3)[pix_idx]
            np.savez(img_cache, X=X_img, y_tissue=labels.reshape(-1)[pix_idx],
                     y_uncert=uncertainty.reshape(-1)[pix_idx], y_meanv=mean_v.reshape(-1)[pix_idx])
            n_done_now += 1

            if (i + 1) % 25 == 0 or (i + 1) == n_total:
                elapsed = time.time() - t0
                rate = n_done_now / elapsed if elapsed > 0 else 0
                print(f"[umap valset] image {i + 1}/{n_total} | {n_done_now} new this run | "
                      f"{rate:.3f} img/s", flush=True)

        cached_files = sorted(pathlib.Path(feats_dir).glob("*.npz"), key=lambda p: int(p.stem))
        X_all, y_tissue_all, y_uncert_all, y_meanv_all, sample_id_all = [], [], [], [], []
        for f in cached_files:
            d = np.load(f)
            X_all.append(d["X"])
            y_tissue_all.append(d["y_tissue"])
            y_uncert_all.append(d["y_uncert"])
            y_meanv_all.append(d["y_meanv"])
            sample_id_all.append(np.full(d["X"].shape[0], int(f.stem)))

        n_images = len(cached_files)
        X = np.concatenate(X_all, axis=0)
        y_tissue = np.concatenate(y_tissue_all)
        y_uncert = np.concatenate(y_uncert_all)
        y_meanv = np.concatenate(y_meanv_all)
        sample_id = np.concatenate(sample_id_all)
        print(f"val set: {n_images} images, {X.shape[0]} points total, feature dim {X.shape[1]}", flush=True)

        if X.shape[0] > max_umap_points:
            sub_idx = np.random.default_rng(43).choice(X.shape[0], size=max_umap_points, replace=False)
            X, y_tissue, y_uncert, y_meanv, sample_id = (
                X[sub_idx], y_tissue[sub_idx], y_uncert[sub_idx], y_meanv[sub_idx], sample_id[sub_idx])
            print(f"subsampled to {X.shape[0]} points for UMAP fit", flush=True)

        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42, verbose=True)
        embedding = reducer.fit_transform(X)

        np.savez(embedding_cache_path, embedding=embedding, y_tissue=y_tissue, y_uncert=y_uncert,
                 y_meanv=y_meanv, sample_id=sample_id, n_images=n_images)
        print(f"cached fitted embedding to {embedding_cache_path}, future runs load it "
              f"directly and skip straight to plotting", flush=True)

    uncert_median = np.median(y_uncert)
    y_confidence = (y_uncert >= uncert_median).astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    for ax, lbl in zip(axes.flat, ['a', 'b', 'c', 'd', 'e', 'f']):
        ax.text(-0.02, 1.03, lbl, transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom')
        ax.set_xticks([]); ax.set_yticks([])

    sc_a = axes[0, 0].scatter(embedding[:, 0], embedding[:, 1], s=4, c=y_meanv,
                               cmap='RdPu_r', alpha=0.5, edgecolors='none')
    plt.colorbar(sc_a, ax=axes[0, 0], fraction=0.046, pad=0.04, label='Mean-prediction brightness (V)')
    axes[0, 0].set_title('Prediction (mean brightness)', fontsize=12)

    for cls in range(3):
        m = y_tissue == cls
        axes[0, 1].scatter(embedding[m, 0], embedding[m, 1], s=4, alpha=0.4,
                            c=TISSUE_COLORS[cls], label=TISSUE_NAMES[cls], edgecolors='none')
    axes[0, 1].legend(markerscale=4, fontsize=9, loc='upper right')
    axes[0, 1].set_title('Tissue class (from ground truth)', fontsize=12)

    sc_id = axes[0, 2].scatter(embedding[:, 0], embedding[:, 1], s=4, c=sample_id,
                                cmap='nipy_spectral', alpha=0.5, edgecolors='none')
    plt.colorbar(sc_id, ax=axes[0, 2], fraction=0.046, pad=0.04, label='Source image index')
    axes[0, 2].set_title(f'Source image (n={n_images} val images)', fontsize=12)

    vmax = np.percentile(y_uncert, 97)
    order = np.argsort(y_uncert)
    sc_c = axes[1, 0].scatter(embedding[order, 0], embedding[order, 1], s=4, alpha=0.5,
                               c=y_uncert[order], cmap='inferno', vmin=0, vmax=vmax, edgecolors='none')
    plt.colorbar(sc_c, ax=axes[1, 0], fraction=0.046, pad=0.04, label='MC-DDIM variance (clipped at p97)')
    axes[1, 0].set_title('Uncertainty', fontsize=12)

    for cls in range(2):
        m = y_confidence == cls
        axes[1, 1].scatter(embedding[m, 0], embedding[m, 1], s=4, alpha=0.4,
                            c=CONF_COLORS[cls], label=CONF_NAMES[cls], edgecolors='none')
    axes[1, 1].legend(markerscale=4, fontsize=9, loc='upper right')
    axes[1, 1].set_title('Confidence (median split of uncertainty)', fontsize=12)

    axes[1, 2].axis('off')

    plt.suptitle(f'UMAP embedding of per-pixel MC-DDIM predictions - whole val set ({n_images} images)',
                 fontsize=14, y=1.0)
    fig.text(0.5, -0.01, COLOUR_NOTE, ha='center', va='top', fontsize=9.5, color='dimgray', wrap=True)
    plt.tight_layout()
    out_path = pathlib.Path(OUT_DIR) / "umap_valset.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()


def run_umap_spread():
    """Part 3: point-count vs. embedding-spread summary per tissue class,
    reusing part 2's cached embedding without refitting."""
    embedding_cache_path = os.path.join(OUT_DIR, "umap_valset_embedding.npz")
    assert os.path.exists(embedding_cache_path), "run_umap_valset must run at least once first"

    cached = np.load(embedding_cache_path)
    embedding, y_tissue = cached["embedding"], cached["y_tissue"]
    print(f"loaded cached embedding from {embedding_cache_path}, reusing part 2's fit, "
          f"no refit needed", flush=True)

    summary = []
    print(f'{"Class":<20}{"Count":>10}{"% of points":>14}{"Mean dist to centroid":>24}{"Convex hull area":>18}')
    for cls in range(3):
        m = y_tissue == cls
        pts = embedding[m]
        count = int(m.sum())
        pct = 100 * count / len(y_tissue)
        centroid = pts.mean(axis=0)
        mean_dist = float(np.linalg.norm(pts - centroid, axis=1).mean())
        hull_area = float(ConvexHull(pts).volume)   # scipy: 'volume' is the area for a 2-D hull
        summary.append((count, pct, mean_dist, hull_area))
        print(f'{TISSUE_NAMES[cls]:<20}{count:>10}{pct:>13.1f}%{mean_dist:>24.3f}{hull_area:>18.1f}')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(TISSUE_NAMES, [s[1] for s in summary], color=TISSUE_COLORS)
    axes[0].set_ylabel('% of points')
    axes[0].set_title('Point count share per class')

    axes[1].bar(TISSUE_NAMES, [s[2] for s in summary], color=TISSUE_COLORS)
    axes[1].set_ylabel('Mean distance to class centroid (embedding units)')
    axes[1].set_title('Spread: mean distance to centroid')

    axes[2].bar(TISSUE_NAMES, [s[3] for s in summary], color=TISSUE_COLORS)
    axes[2].set_ylabel('Convex hull area (embedding units^2)')
    axes[2].set_title('Spread: convex hull area')

    for ax in axes:
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle('Point count vs. spread per tissue class in the whole-val-set UMAP embedding', fontsize=13)
    plt.tight_layout()
    out_path = pathlib.Path(OUT_DIR) / "umap_class_spread_vs_count.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scheduler = build_model_and_scheduler(device)
    load_checkpoint(model, CKPT_PATH, device)
    model.eval()
    dataset = VirtualStainingDataset(VAL_DIR)

    if which in ("1", "all"):
        run_umap_per_sample(model, scheduler, device, dataset)
    if which in ("2", "all"):
        run_umap_valset(model, scheduler, device, dataset)
    if which in ("3", "all"):
        run_umap_spread()


if __name__ == "__main__":
    main()
