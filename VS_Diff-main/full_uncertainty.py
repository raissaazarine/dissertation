"""Full-val-set (7373 images) predictive-uncertainty analysis: 8 stochastic
DDIM samples per image (eta=1.0), giving per-pixel mean/std, checked against
actual error vs. ground truth. ~11.4 hours (a 50-step DDIM sample takes
~0.696s on this GPU).

Resumable: per-image sufficient statistics (n_pixels, sum_err, sum_std,
sum_err*std, sum_err^2, sum_std^2) are appended to
sweep_output/uncertainty/per_image_stats.csv as soon as each image is done.
The exact whole-val-set Pearson correlation is reconstructed from these sums
at the end, so there's no need to hold per-pixel arrays for 7373 images in
memory. Mean/std prediction arrays for the first N_VIS images are also saved
as .npy for the qualitative panel plotted separately in the notebook.

Run: vsdiff_env/bin/python full_uncertainty.py
"""
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import virt_stain_utils2 as vsu
from vsdiff_model import build_model_and_scheduler, load_checkpoint, VirtualStainingDataset

BASE_DIR = "/cs/student/project_msc/2025/aibh/razarine"
CKPT_PATH = os.path.join(BASE_DIR, "runs/exp35/checkpoints/best.pth")
VAL_DIR = os.path.join(BASE_DIR, "datasets/polyps_v7/val")
OUT_DIR = os.path.join(BASE_DIR, "sweep_output/uncertainty")
STEPS = 50
ETA = 1.0
N_SAMPLES = 8
N_VIS = 4  # first N_VIS images get their full mean/std arrays saved for plotting

FIELDNAMES = ["image_idx", "file", "n_pixels", "sum_err", "sum_std",
              "sum_err_std", "sum_err2", "sum_std2"]


def denorm01(x):
    return x.clamp(-1, 1) * 0.5 + 0.5


def mc_uncertainty(model, phase, scheduler, device, steps, eta, n_samples):
    preds = []
    for _ in range(n_samples):
        with torch.no_grad():
            pred = vsu.ddim_sample_full(model, phase, scheduler, device,
                                         num_inference_steps=steps, eta=eta)
        preds.append(denorm01(pred[0]))
    stack = torch.stack(preds, dim=0)
    return stack.mean(dim=0), stack.std(dim=0)


def load_done_indices(csv_path):
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                done.add(int(row["image_idx"]))
    return done


def rebuild_correlation_summary(out_dir, per_image_csv):
    """Computes the exact Pearson r over all rows written so far, from the
    per-image sufficient statistics. Called periodically during a run and
    once more at the end."""
    n = sum_e = sum_s = sum_es = sum_e2 = sum_s2 = 0.0
    with open(per_image_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            n += float(row["n_pixels"])
            sum_e += float(row["sum_err"])
            sum_s += float(row["sum_std"])
            sum_es += float(row["sum_err_std"])
            sum_e2 += float(row["sum_err2"])
            sum_s2 += float(row["sum_std2"])

    if n == 0:
        return None
    mean_e, mean_s = sum_e / n, sum_s / n
    cov = sum_es / n - mean_e * mean_s
    var_e = sum_e2 / n - mean_e ** 2
    var_s = sum_s2 / n - mean_s ** 2
    corr = cov / (var_e ** 0.5 * var_s ** 0.5)

    summary_path = os.path.join(out_dir, "correlation_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write(f"n_pixels_total={int(n)}\nmean_err={mean_e}\nmean_std={mean_s}\n"
                 f"pearson_r={corr}\n")
    return corr


def run_uncertainty(model, scheduler, device, dataset):
    """Called from a notebook cell with a preloaded model, or from main()
    below. Same resumable CSV logic either way."""
    os.makedirs(OUT_DIR, exist_ok=True)
    vis_dir = os.path.join(OUT_DIR, "vis_arrays")
    os.makedirs(vis_dir, exist_ok=True)
    per_image_csv = os.path.join(OUT_DIR, "per_image_stats.csv")

    n_total = len(dataset)
    print(f"{n_total} val images", flush=True)

    done = load_done_indices(per_image_csv)
    write_header = not os.path.exists(per_image_csv)
    fh = open(per_image_csv, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
        fh.flush()

    t0 = time.time()
    n_done_now = 0
    for i in range(n_total):
        if i in done:
            continue

        phase, gt = dataset[i]
        fname = os.path.basename(dataset.image_paths[i])
        phase_b = phase.unsqueeze(0).to(device)
        gt01 = denorm01(gt.to(device))

        mean_pred, std_pred = mc_uncertainty(model, phase_b, scheduler, device, STEPS, ETA, N_SAMPLES)
        err_map = (mean_pred - gt01).abs().mean(dim=0)  # [H, W]
        unc_map = std_pred.mean(dim=0)                  # [H, W]

        if i < N_VIS:
            np.savez(os.path.join(vis_dir, f"{i}.npz"),
                     phase=phase.cpu().numpy(), gt01=gt01.cpu().numpy(),
                     mean_pred=mean_pred.cpu().numpy(), unc_map=unc_map.cpu().numpy(),
                     fname=fname)

        e = err_map.flatten().double()
        s = unc_map.flatten().double()
        writer.writerow({
            "image_idx": i, "file": fname, "n_pixels": e.numel(),
            "sum_err": float(e.sum()), "sum_std": float(s.sum()),
            "sum_err_std": float((e * s).sum()),
            "sum_err2": float((e * e).sum()), "sum_std2": float((s * s).sum()),
        })
        fh.flush()
        n_done_now += 1

        if (i + 1) % 25 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t0
            rate = n_done_now / elapsed if elapsed > 0 else 0
            eta_remaining = (n_total - (i + 1)) / rate / 3600 if rate > 0 else float("nan")
            print(f"image {i + 1}/{n_total} | {n_done_now} new this run | "
                  f"{rate:.3f} img/s | ETA {eta_remaining:.1f} h", flush=True)
            rebuild_correlation_summary(OUT_DIR, per_image_csv)

    fh.close()
    corr = rebuild_correlation_summary(OUT_DIR, per_image_csv)
    print(f"DONE. Full-val-set pixel-wise Pearson r(error, uncertainty) = {corr:.4f}", flush=True)
    return corr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scheduler = build_model_and_scheduler(device)
    load_checkpoint(model, CKPT_PATH, device)
    model.eval()
    dataset = VirtualStainingDataset(VAL_DIR)
    run_uncertainty(model, scheduler, device, dataset)


if __name__ == "__main__":
    main()
