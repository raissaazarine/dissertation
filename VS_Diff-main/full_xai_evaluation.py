"""Perturbation-based evaluation of XAI attribution methods for VS-Diff --
Deletion AUC, Insertion AUC, Sensitivity, Entropy -- mirroring
xai_evaluation.py (the Pix2Pix side, adapted from Mnyambo et al.). Runs the
same 3 methods -- Saliency (vanilla gradient), Grad x Input, SmoothGrad --
over the full val set. Resumable: re-running skips any (image_idx, method)
pair already present in the output CSV.

All 3 attribution methods share the SmoothGrad convention already used in
full_saliency.py: backprop the L2 norm of the predicted noise through the
phase input at a fixed mid-trajectory timestep (t_frac=0.5) with a fixed
random noisy-stained reference -- VS-Diff's UNet takes a single
(noisy_stained, phase, t) triple per call, unlike Pix2Pix's one-shot forward
pass, so there's no single whole-generation "loss" to differentiate; the
timestep is fixed instead so the 3 methods are directly comparable (only the
gradient post-processing differs between them).

Deletion/Insertion AUC needs a full DDIM sample per masking fraction (not a
single forward pass like Pix2Pix) -- by far the most expensive part of this
script. To keep the full-val-set version tractable it uses a reduced
DDIM_EVAL_STEPS=20 (vs. the 50 steps used for the main sweep/uncertainty
analyses) purely for these relative comparisons BETWEEN attribution methods,
not for reporting absolute image quality -- clean_pred here is computed at
the same reduced step count, so the comparison stays internally consistent.

Estimated ~37.6 hours over the full val set (measured: a 50-step DDIM sample
takes ~0.696s on this GPU, so a 20-step sample ~0.28s; 22 samples/method x 3
methods x 7373 images).
"""
import csv
import os
import sys
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure

sys.path.insert(0, os.path.dirname(__file__))
import virt_stain_utils2 as vsu
from vsdiff_model import build_model_and_scheduler, load_checkpoint, VirtualStainingDataset

BASE_DIR = "/cs/student/project_msc/2025/aibh/razarine"
CKPT_PATH = os.path.join(BASE_DIR, "runs/exp35/checkpoints/best.pth")
VAL_DIR = os.path.join(BASE_DIR, "datasets/polyps_v7/val")
OUT_DIR = os.path.join(BASE_DIR, "sweep_output/xai_evaluation")

T_FRAC = 0.5              # fixed mid-trajectory timestep, matches full_saliency.py
DDIM_EVAL_STEPS = 20      # reduced from 50 -- see module docstring
N_FRACTIONS = 10          # 11 points: 0.0, 0.1, ..., 1.0 -- matches xai_evaluation.py
N_NOISE = 5               # sensitivity: number of noisy re-evaluations
NOISE_STD = 0.03          # sensitivity: input noise std
FILL_VALUE = -1.0         # masked-out pixel value (normalized range)
SMOOTHGRAD_N_SAMPLES = 15  # matches full_saliency.py


def denorm01(x):
    return x.clamp(-1, 1) * 0.5 + 0.5


# ---- 3 attribution methods (share a fixed t/noisy_stained reference unless given one) ----

def _fixed_reference(scheduler, phase, device):
    t = torch.tensor([int(scheduler.config.num_train_timesteps * T_FRAC)], device=device)
    noisy_stained = torch.randn(1, 3, phase.shape[-2], phase.shape[-1], device=device)
    return t, noisy_stained


def saliency_map(model, phase, scheduler, device, t=None, noisy_stained=None):
    """Vanilla gradient: |d(score)/d(phase)|, single backward pass."""
    if t is None or noisy_stained is None:
        t, noisy_stained = _fixed_reference(scheduler, phase, device)
    phase_g = phase.clone().requires_grad_(True)
    model_input = torch.cat([noisy_stained, phase_g], dim=1)
    noise_pred = model(model_input, t).sample
    score = noise_pred.pow(2).sum()
    grad = torch.autograd.grad(score, phase_g)[0]
    sal = grad.abs().squeeze(0).squeeze(0).detach()
    return (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)


def grad_times_input(model, phase, scheduler, device, t=None, noisy_stained=None):
    """Gradient x Input: |d(score)/d(phase) * phase|, single backward pass."""
    if t is None or noisy_stained is None:
        t, noisy_stained = _fixed_reference(scheduler, phase, device)
    phase_g = phase.clone().requires_grad_(True)
    model_input = torch.cat([noisy_stained, phase_g], dim=1)
    noise_pred = model(model_input, t).sample
    score = noise_pred.pow(2).sum()
    grad = torch.autograd.grad(score, phase_g)[0]
    gxi = (grad * phase_g).abs().squeeze(0).squeeze(0).detach()
    return (gxi - gxi.min()) / (gxi.max() - gxi.min() + 1e-8)


def smoothgrad(model, phase, scheduler, device, t=None, noisy_stained=None,
               n_smooth=SMOOTHGRAD_N_SAMPLES, noise_level=0.15):
    """SmoothGrad (Smilkov et al., 2017) -- same as full_saliency.py's saliency_map."""
    if t is None or noisy_stained is None:
        t, noisy_stained = _fixed_reference(scheduler, phase, device)
    grad_sq_sum = torch.zeros_like(phase)
    for _ in range(n_smooth):
        noisy_phase = (phase + torch.randn_like(phase) * noise_level).clone().requires_grad_(True)
        model_input = torch.cat([noisy_stained, noisy_phase], dim=1)
        noise_pred = model(model_input, t).sample
        score = noise_pred.pow(2).sum()
        grad = torch.autograd.grad(score, noisy_phase)[0]
        grad_sq_sum = grad_sq_sum + grad.pow(2)
    sg = (grad_sq_sum / n_smooth).sqrt().squeeze(0).squeeze(0).detach()
    return (sg - sg.min()) / (sg.max() - sg.min() + 1e-8)


XAI_METHODS = {
    "Saliency": saliency_map,
    "Grad x Input": grad_times_input,
    "SmoothGrad": smoothgrad,
}


# ---- perturbation-based faithfulness metrics ----

def rank_positions(attribution_np):
    h, w = attribution_np.shape
    order = np.argsort(-attribution_np.reshape(-1))
    return np.unravel_index(order, (h, w))


def mask_delete(phase, ys, xs, n_masked, fill_value=FILL_VALUE):
    p = phase.clone()
    p[0, 0, ys[:n_masked], xs[:n_masked]] = fill_value
    return p


def mask_insert(phase, ys, xs, n_inserted, fill_value=FILL_VALUE):
    base = torch.full_like(phase, fill_value)
    base[0, 0, ys[:n_inserted], xs[:n_inserted]] = phase[0, 0, ys[:n_inserted], xs[:n_inserted]]
    return base


def deletion_insertion_auc(model, phase, scheduler, device, attribution, clean_01, ssim_metric,
                            steps=DDIM_EVAL_STEPS, n_fractions=N_FRACTIONS):
    """Progressively delete (mask out) / insert (keep only) the top-attributed
    pixels, re-sample, and measure SSIM against the (reduced-step) clean
    prediction. Deletion AUC low = attribution correctly finds pixels whose
    removal hurts the output; Insertion AUC high = attribution correctly
    finds pixels sufficient to reconstruct the output on their own."""
    attribution_np = attribution.cpu().numpy()
    h, w = attribution_np.shape
    n_pix = h * w
    ys, xs = rank_positions(attribution_np)
    fractions = np.linspace(0, 1, n_fractions + 1)
    del_ssim = np.empty(n_fractions + 1, dtype=np.float32)
    ins_ssim = np.empty(n_fractions + 1, dtype=np.float32)

    for i, frac in enumerate(fractions):
        n = int(round(frac * n_pix))

        del_phase = mask_delete(phase, ys, xs, n)
        with torch.no_grad():
            del_pred = vsu.ddim_sample_full(model, del_phase, scheduler, device,
                                             num_inference_steps=steps, eta=0.0)[0]
        del_01 = denorm01(del_pred)
        del_ssim[i] = ssim_metric(del_01.unsqueeze(0), clean_01.unsqueeze(0)).item()

        ins_phase = mask_insert(phase, ys, xs, n)
        with torch.no_grad():
            ins_pred = vsu.ddim_sample_full(model, ins_phase, scheduler, device,
                                             num_inference_steps=steps, eta=0.0)[0]
        ins_01 = denorm01(ins_pred)
        ins_ssim[i] = ssim_metric(ins_01.unsqueeze(0), clean_01.unsqueeze(0)).item()

    return float(np.trapz(del_ssim, fractions)), float(np.trapz(ins_ssim, fractions))


def explanation_sensitivity(attribution_fn, model, phase, scheduler, device, base_map,
                             n_noise=N_NOISE, noise_std=NOISE_STD):
    """How much the attribution map itself changes under small input
    perturbations -- correlation between the base map and maps recomputed on
    noisy copies of the input. Higher = more stable/trustworthy explanation."""
    base_flat = base_map.cpu().numpy().reshape(-1)
    corrs = []
    for _ in range(n_noise):
        noisy_phase = torch.clamp(phase + torch.randn_like(phase) * noise_std, -1.0, 1.0)
        noisy_map = attribution_fn(model, noisy_phase, scheduler, device)
        corrs.append(np.corrcoef(base_flat, noisy_map.cpu().numpy().reshape(-1))[0, 1])
    return float(np.mean(corrs))


def explanation_entropy(attribution):
    """Shannon entropy of the (normalized-to-sum-1) attribution map -- low =
    concentrated on a few pixels, high = spread out everywhere (less useful
    as an explanation)."""
    p = attribution.cpu().numpy().reshape(-1).astype(np.float64)
    p = p / (p.sum() + 1e-12)
    p = np.clip(p, 1e-12, None)
    return float(-(p * np.log2(p)).sum())


# ---- resumable full-val-set evaluation ----

def load_done_pairs(csv_path):
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                done.add((int(row["image_idx"]), row["method"]))
    return done


def run_xai_evaluation(model, scheduler, device, dataset, ssim_metric):
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "perturbation_metrics_per_image.csv")
    fieldnames = ["image_idx", "method", "deletion_auc", "insertion_auc", "sensitivity", "entropy"]

    done_pairs = load_done_pairs(csv_path)
    write_header = not os.path.exists(csv_path)
    fh = open(csv_path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        fh.flush()
    if done_pairs:
        print(f"resuming: {len(done_pairs)} (image, method) results already saved", flush=True)

    n_total = len(dataset)
    t0 = time.time()
    n_done_now = 0
    for img_idx in range(n_total):
        needed = [name for name in XAI_METHODS if (img_idx, name) not in done_pairs]
        if not needed:
            continue

        phase, _ = dataset[img_idx]
        phase = phase.unsqueeze(0).to(device)

        with torch.no_grad():
            clean_pred = vsu.ddim_sample_full(model, phase, scheduler, device,
                                               num_inference_steps=DDIM_EVAL_STEPS, eta=0.0)[0]
        clean_01 = denorm01(clean_pred)

        for name in needed:
            fn = XAI_METHODS[name]
            attribution = fn(model, phase, scheduler, device)

            deletion_auc, insertion_auc = deletion_insertion_auc(
                model, phase, scheduler, device, attribution, clean_01, ssim_metric)
            sensitivity = explanation_sensitivity(fn, model, phase, scheduler, device, attribution)
            entropy = explanation_entropy(attribution)

            writer.writerow({
                "image_idx": img_idx, "method": name,
                "deletion_auc": deletion_auc, "insertion_auc": insertion_auc,
                "sensitivity": sensitivity, "entropy": entropy,
            })
            fh.flush()
            n_done_now += 1

        if (img_idx + 1) % 25 == 0 or (img_idx + 1) == n_total:
            elapsed = time.time() - t0
            rate = n_done_now / elapsed if elapsed > 0 else 0
            print(f"[xai_eval] image {img_idx + 1}/{n_total} | {n_done_now} new (image,method) "
                  f"this run | {rate:.3f}/s | elapsed {elapsed/60:.1f} min", flush=True)

    fh.close()
    print(f"done -> results saved to {csv_path}", flush=True)
    return csv_path


def summarize_xai_evaluation():
    """Loads the per-image CSV, prints per-method means, and plots a 4-panel
    bar comparison (Deletion AUC / Insertion AUC / Sensitivity / Entropy)."""
    csv_path = os.path.join(OUT_DIR, "perturbation_metrics_per_image.csv")
    assert os.path.exists(csv_path), "run_xai_evaluation must run at least once first"

    rows_by_method = {name: [] for name in XAI_METHODS}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["method"] in rows_by_method:
                rows_by_method[row["method"]].append(row)

    metrics = ["deletion_auc", "insertion_auc", "sensitivity", "entropy"]
    means = {name: {m: float(np.mean([float(r[m]) for r in rows])) for m in metrics}
             for name, rows in rows_by_method.items() if rows}

    print(f'{"Method":<16}{"n":>6}{"Deletion AUC":>16}{"Insertion AUC":>16}'
          f'{"Sensitivity":>14}{"Entropy":>12}')
    for name, rows in rows_by_method.items():
        if not rows:
            continue
        m = means[name]
        print(f'{name:<16}{len(rows):>6}{m["deletion_auc"]:>16.4f}{m["insertion_auc"]:>16.4f}'
              f'{m["sensitivity"]:>14.4f}{m["entropy"]:>12.4f}')

    methods = [name for name in XAI_METHODS if means.get(name)]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    titles = ["Deletion AUC (lower = better)", "Insertion AUC (higher = better)",
              "Sensitivity (higher = better)", "Entropy (context-dependent)"]
    for ax, metric, title in zip(axes, metrics, titles):
        ax.bar(methods, [means[m][metric] for m in methods], color=["#4c72b0", "#dd8452", "#55a868"])
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("XAI attribution method comparison (perturbation-based faithfulness)", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "xai_method_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"plot saved -> {out_path}")


def pairwise_difference_heatmap(save=True):
    """For each perturbation-based faithfulness metric, show the 3x3 matrix of
    mean paired differences between XAI methods as a heatmap: cell (row, col) =
    mean over val images of (metric[row_method] - metric[col_method]). The
    matrix is antisymmetric with a zero diagonal; a diverging colormap is
    centred at 0 and each metric gets its own scale (entropy differences are
    ~10x the Deletion/Insertion AUC ones). If SciPy is available, cells are
    starred by paired t-test p-value (row vs. column, same images)."""
    import pandas as pd

    csv_path = os.path.join(OUT_DIR, "perturbation_metrics_per_image.csv")
    assert os.path.exists(csv_path), "run_xai_evaluation must run at least once first"

    df = pd.read_csv(csv_path)
    metrics = ["deletion_auc", "insertion_auc", "sensitivity", "entropy"]
    titles = {
        "deletion_auc": "Deletion AUC (lower = better)",
        "insertion_auc": "Insertion AUC (higher = better)",
        "sensitivity": "Sensitivity (higher = better)",
        "entropy": "Entropy (context-dependent)",
    }
    methods = [m for m in XAI_METHODS if m in set(df["method"])]
    n_m = len(methods)

    # one image-aligned wide frame (index=image_idx, columns=methods) per metric
    wide = {m: df.pivot(index="image_idx", columns="method", values=m)[methods].dropna()
            for m in metrics}

    try:
        from scipy import stats as _stats
        have_scipy = True
    except ImportError:
        have_scipy = False

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.8))
    for ax, metric in zip(axes, metrics):
        W = wide[metric]
        diff = np.zeros((n_m, n_m))
        pval = np.ones((n_m, n_m))
        for i in range(n_m):
            for j in range(n_m):
                a, b = W.iloc[:, i].to_numpy(), W.iloc[:, j].to_numpy()
                diff[i, j] = np.mean(a - b)
                if have_scipy and i != j:
                    pval[i, j] = _stats.ttest_rel(a, b).pvalue

        vmax = np.abs(diff).max() or 1.0
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(n_m)); ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_yticks(range(n_m)); ax.set_yticklabels(methods)
        ax.set_title(titles[metric], fontsize=10)
        for i in range(n_m):
            for j in range(n_m):
                star = ""
                if have_scipy and i != j:
                    p = pval[i, j]
                    star = "\n***" if p < 1e-3 else "\n**" if p < 1e-2 else "\n*" if p < 5e-2 else ""
                ax.text(j, i, f"{diff[i, j]:+.4f}{star}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n_img = len(next(iter(wide.values())))
    subtitle = f"paired over {n_img} val images  (row - column)"
    if have_scipy:
        subtitle += "   * p<.05   ** p<.01   *** p<.001 (paired t-test)"
    fig.suptitle("Pairwise mean difference between XAI attribution methods\n" + subtitle,
                 fontsize=12)
    plt.tight_layout()
    if save:
        out_path = os.path.join(OUT_DIR, "xai_pairwise_difference_heatmap.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"plot saved -> {out_path}")
    plt.show()


def pairwise_difference_heatmap_single(save=True):
    """One combined heatmap: 3 method pairs (rows) x 4 faithfulness metrics
    (columns). Each cell is the mean paired difference (methodA - methodB)
    over the val images. Because the metrics live on very different scales
    (entropy diffs ~10x the Deletion/Insertion AUC ones), the CELL COLOUR is
    normalized per column (divided by that column's max |difference|, diverging
    map centred at 0) while the printed number is the raw mean difference."""
    import pandas as pd

    csv_path = os.path.join(OUT_DIR, "perturbation_metrics_per_image.csv")
    assert os.path.exists(csv_path), "run_xai_evaluation must run at least once first"

    df = pd.read_csv(csv_path)
    metrics = ["deletion_auc", "insertion_auc", "sensitivity", "entropy"]
    col_labels = ["Deletion AUC", "Insertion AUC", "Sensitivity", "Entropy"]
    methods = [m for m in XAI_METHODS if m in set(df["method"])]
    pairs = [(a, b) for k, a in enumerate(methods) for b in methods[k + 1:]]
    _disp = lambda s: s.replace("Grad x Input", "Grad × Input")
    row_labels = [f"{_disp(a)} − {_disp(b)}" for a, b in pairs]

    wide = {m: df.pivot(index="image_idx", columns="method", values=m)[methods].dropna()
            for m in metrics}

    raw = np.zeros((len(pairs), len(metrics)))
    norm = np.zeros_like(raw)
    for cj, metric in enumerate(metrics):
        W = wide[metric]
        for ri, (a, b) in enumerate(pairs):
            raw[ri, cj] = np.mean(W[a].to_numpy() - W[b].to_numpy())
        scale = np.abs(raw[:, cj]).max() or 1.0
        norm[:, cj] = raw[:, cj] / scale

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    im = ax.imshow(norm, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(pairs))); ax.set_yticklabels(row_labels, fontsize=10)
    ax.tick_params(length=0)
    for ri in range(len(pairs)):
        for cj in range(len(metrics)):
            ax.text(cj, ri, f"{raw[ri, cj]:+.4f}", ha="center", va="center", fontsize=9)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("normalised Δ (per column)", fontsize=8)

    ax.set_title("Diffusion Model — Pairwise Differences Between XAI Attribution Methods",
                 fontsize=11.5, pad=10)
    plt.tight_layout()
    if save:
        out_path = os.path.join(OUT_DIR, "xai_pairwise_difference_heatmap_single.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"plot saved -> {out_path}")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scheduler = build_model_and_scheduler(device)
    load_checkpoint(model, CKPT_PATH, device)
    model.eval()
    dataset = VirtualStainingDataset(VAL_DIR)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    run_xai_evaluation(model, scheduler, device, dataset, ssim_metric)
    summarize_xai_evaluation()


if __name__ == "__main__":
    main()
