"""Full-val-set (7373 images) uncertainty-vs-error retrieval validation for
the diffusion model, mirroring pix2pix_minimal/uncertainty_sweep.py's
protocol so both models are validated with the same metric: AUC of
predictive uncertainty for identifying the worst-error-quartile pixels, plus
a Youden's J-optimal threshold.

full_uncertainty.py only keeps per-image sufficient statistics (sums), which
is enough to reconstruct the exact whole-val-set Pearson r without holding
per-pixel arrays in memory, but not enough to compute an AUC, which needs
the actual pooled distribution of (uncertainty, error) pixel pairs. So this
is a separate pass rather than a hook into full_uncertainty.py, and can run
independently of an already-running or already-finished full_uncertainty.py
job. It reuses full_uncertainty.py's model setup and MC-sampling (same 8x
stochastic DDIM samples per image, eta=1.0), so the same ~11-hour cost.

Per image, the uncertainty (predictive std) and error (|mean pred - GT|)
maps are subsampled on an 8-pixel stride (same STRIDE=8 as
pix2pix_minimal/uncertainty_sweep.py, ~1024 points/image, to keep the pooled
arrays manageable over 7373 images) and cached to
sweep_output/uncertainty_auc/per_image_pixels/ as soon as each image is
done, so an interrupted run resumes instead of restarting. Re-running this
script (or calling compute_auc_summary directly) recomputes the AUC from
whatever's cached so far, matching full_uncertainty.py's partial-progress
correlation summary.

Run: vsdiff_env/bin/python full_uncertainty_auc.py
"""
import os
import sys
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(__file__))
from full_uncertainty import (
    BASE_DIR, CKPT_PATH, VAL_DIR, STEPS, ETA, N_SAMPLES, denorm01, mc_uncertainty,
)
from vsdiff_model import build_model_and_scheduler, load_checkpoint, VirtualStainingDataset

OUT_DIR = os.path.join(BASE_DIR, "sweep_output/uncertainty_auc")
STRIDE = 8  # matches pix2pix_minimal/uncertainty_sweep.py, keeps pooled arrays manageable


def load_done_indices(pix_dir):
    done = set()
    if os.path.isdir(pix_dir):
        for fn in os.listdir(pix_dir):
            if fn.endswith(".npz"):
                done.add(int(fn[:-4]))
    return done


def compute_auc_summary(pix_dir, out_dir):
    """Computes AUC, ROC, and Youden's J over all per-image pixel caches
    written so far. Called at the end of a full run, or standalone to check
    progress."""
    files = [f for f in os.listdir(pix_dir) if f.endswith(".npz")]
    if not files:
        return None

    all_err, all_unc = [], []
    for fn in files:
        d = np.load(os.path.join(pix_dir, fn))
        all_err.append(d["err"])
        all_unc.append(d["unc"])
    y_error = np.concatenate(all_err)
    y_uncert = np.concatenate(all_unc)

    error_q75 = np.percentile(y_error, 75)
    y_incorrect = (y_error >= error_q75).astype(int)

    auc = roc_auc_score(y_incorrect, y_uncert)
    fpr, tpr, thresholds = roc_curve(y_incorrect, y_uncert)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    theta = thresholds[best_idx]
    median_theta = np.median(y_uncert)

    summary_path = os.path.join(out_dir, "auc_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write(f"n_images={len(files)}\nn_pixels_pooled={y_uncert.shape[0]}\n"
                 f"error_q75={error_q75}\nauc={auc}\nyoudens_j_theta={theta}\n"
                 f"youdens_j_tpr={tpr[best_idx]}\nyoudens_j_fpr={fpr[best_idx]}\n"
                 f"median_theta={median_theta}\n")

    print(f"n_images={len(files)}  n_pixels_pooled={y_uncert.shape[0]}", flush=True)
    print(f"AUC (uncertainty predicting worst-error-quartile pixels): {auc:.4f}", flush=True)
    print(f"Youden's J theta: {theta:.6f}  (TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f})", flush=True)
    print(f"Naive median-split threshold: {median_theta:.6f}", flush=True)
    return dict(auc=auc, theta=theta, median_theta=median_theta, n_pixels=int(y_uncert.shape[0]))


def plot_auc_figure(out_dir=OUT_DIR, n_samples=N_SAMPLES):
    """3-panel figure mirroring pix2pix_minimal's
    uncertainty_vs_error_validation_FULLVAL.png: distribution histogram,
    joint hexbin, ROC curve, rebuilt from the same per-image pixel caches
    compute_auc_summary reads. Call after run_uncertainty_auc has finished,
    or to re-plot from whatever's cached so far."""
    pix_dir = os.path.join(out_dir, "per_image_pixels")
    files = [f for f in os.listdir(pix_dir) if f.endswith(".npz")]
    assert files, f"no cached pixels in {pix_dir}, run_uncertainty_auc must run at least once first"

    all_err, all_unc = [], []
    for fn in files:
        d = np.load(os.path.join(pix_dir, fn))
        all_err.append(d["err"])
        all_unc.append(d["unc"])
    y_error = np.concatenate(all_err)
    y_uncert = np.concatenate(all_unc)
    n_images = len(files)

    error_q75 = np.percentile(y_error, 75)
    y_incorrect = (y_error >= error_q75).astype(int)

    auc = roc_auc_score(y_incorrect, y_uncert)
    fpr, tpr, thresholds = roc_curve(y_incorrect, y_uncert)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    theta = thresholds[best_idx]
    median_theta = np.median(y_uncert)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    bins = np.linspace(0, np.percentile(y_uncert, 99), 60)
    axes[0].hist(y_uncert[y_incorrect == 0], bins=bins, density=True, alpha=0.5, color='seagreen', label='Correct (error < p75)')
    axes[0].hist(y_uncert[y_incorrect == 1], bins=bins, density=True, alpha=0.5, color='darkorange', label='Incorrect (error >= p75)')
    axes[0].axvline(theta, color='black', linestyle='--', linewidth=1.5, label=f"Youden's J theta={theta:.5f}")
    axes[0].axvline(median_theta, color='gray', linestyle=':', linewidth=1.5, label=f'median split={median_theta:.5f}')
    axes[0].set_xlabel('MC-DDIM uncertainty (std)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Uncertainty distribution: correct vs incorrect pixels', fontsize=12)
    axes[0].legend(fontsize=8)

    hb = axes[1].hexbin(y_uncert, y_error, gridsize=60, bins='log', cmap='viridis',
                         extent=(0, np.percentile(y_uncert, 99), 0, np.percentile(y_error, 99)))
    plt.colorbar(hb, ax=axes[1], fraction=0.046, pad=0.04, label='log10(pixel count)')
    axes[1].axvline(theta, color='white', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('MC-DDIM uncertainty (std)')
    axes[1].set_ylabel('Pixel error |mean pred - GT|')
    axes[1].set_title('Uncertainty vs. error (per pixel)', fontsize=12)

    axes[2].plot(fpr, tpr, color='steelblue', linewidth=2, label=f'ROC (AUC={auc:.3f})')
    axes[2].plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1)
    axes[2].scatter([fpr[best_idx]], [tpr[best_idx]], color='black', zorder=5, label="Youden's J optimum")
    axes[2].set_xlabel('False positive rate')
    axes[2].set_ylabel('True positive rate')
    axes[2].set_title('ROC: uncertainty predicting worst-error-quartile pixels', fontsize=12)
    axes[2].legend(fontsize=9, loc='lower right')

    plt.suptitle(f'Does MC-DDIM uncertainty track prediction error? '
                 f'(FULL val set, n={n_images} images, N_SAMPLES={n_samples})', fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'uncertainty_vs_error_validation_FULLVAL.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print('saved ->', out_path, flush=True)


def run_uncertainty_auc(model, scheduler, device, dataset):
    """Called from a notebook cell with a preloaded model, or from main()
    below. Same resumable per-image caching either way."""
    os.makedirs(OUT_DIR, exist_ok=True)
    pix_dir = os.path.join(OUT_DIR, "per_image_pixels")
    os.makedirs(pix_dir, exist_ok=True)

    n_total = len(dataset)
    print(f"{n_total} val images", flush=True)

    done = load_done_indices(pix_dir)
    t0 = time.time()
    n_done_now = 0
    for i in range(n_total):
        if i in done:
            continue

        phase, gt = dataset[i]
        phase_b = phase.unsqueeze(0).to(device)
        gt01 = denorm01(gt.to(device))

        mean_pred, std_pred = mc_uncertainty(model, phase_b, scheduler, device, STEPS, ETA, N_SAMPLES)
        err_map = (mean_pred - gt01).abs().mean(dim=0)  # [H, W]
        unc_map = std_pred.mean(dim=0)                  # [H, W]

        err_sub = err_map[::STRIDE, ::STRIDE].flatten().cpu().numpy().astype(np.float32)
        unc_sub = unc_map[::STRIDE, ::STRIDE].flatten().cpu().numpy().astype(np.float32)
        np.savez(os.path.join(pix_dir, f"{i}.npz"), err=err_sub, unc=unc_sub)
        n_done_now += 1

        if (i + 1) % 25 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t0
            rate = n_done_now / elapsed if elapsed > 0 else 0
            eta_remaining = (n_total - (i + 1)) / rate / 3600 if rate > 0 else float("nan")
            print(f"image {i + 1}/{n_total} | {n_done_now} new this run | "
                  f"{rate:.3f} img/s | ETA {eta_remaining:.1f} h", flush=True)

    print("All images done, pooling pixels and computing AUC...", flush=True)
    return compute_auc_summary(pix_dir, OUT_DIR)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scheduler = build_model_and_scheduler(device)
    load_checkpoint(model, CKPT_PATH, device)
    model.eval()
    dataset = VirtualStainingDataset(VAL_DIR)
    run_uncertainty_auc(model, scheduler, device, dataset)
    plot_auc_figure()


if __name__ == "__main__":
    main()
