"""Full-val-set (7373 images) SmoothGrad input-saliency: much cheaper than the
sweep/uncertainty jobs (~0.44s/image measured -> ~54 min total, no iterative
DDIM sampling involved). Produces a dataset-level mean saliency map (which
phase-image regions the model relies on across the whole val set) plus a
per-image summary CSV.

Resumability: the running sum used for the mean saliency map is persisted to
running_sum.npy/count.txt after every image, and per-image scalar summaries
are appended to per_image_saliency.csv -- already-summed image indices are
skipped on restart. Raw per-image maps for the first N_VIS images are kept
for the notebook's qualitative panel.

Run with: vsdiff_env/bin/python full_saliency.py
"""
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from vsdiff_model import build_model_and_scheduler, load_checkpoint, VirtualStainingDataset

BASE_DIR = "/cs/student/project_msc/2025/aibh/razarine"
CKPT_PATH = os.path.join(BASE_DIR, "runs/exp35/checkpoints/best.pth")
VAL_DIR = os.path.join(BASE_DIR, "datasets/polyps_v7/val")
OUT_DIR = os.path.join(BASE_DIR, "sweep_output/saliency")
N_VIS = 4

FIELDNAMES = ["image_idx", "file", "mean_saliency", "max_saliency"]


def denorm01(x):
    return x.clamp(-1, 1) * 0.5 + 0.5


def saliency_map(model, phase, scheduler, device, t_frac=0.5, n_smooth=15, noise_level=0.15):
    model.eval()
    t = torch.tensor([int(scheduler.config.num_train_timesteps * t_frac)], device=device)
    noisy_stained = torch.randn(1, 3, phase.shape[-2], phase.shape[-1], device=device)

    grad_sq_sum = torch.zeros_like(phase)
    for _ in range(n_smooth):
        noisy_phase = (phase + torch.randn_like(phase) * noise_level).clone().requires_grad_(True)
        model_input = torch.cat([noisy_stained, noisy_phase], dim=1)
        noise_pred = model(model_input, t).sample
        score = noise_pred.pow(2).sum()
        grad = torch.autograd.grad(score, noisy_phase)[0]
        grad_sq_sum = grad_sq_sum + grad.pow(2)

    return (grad_sq_sum / n_smooth).sqrt().squeeze(0).squeeze(0).detach()


def load_done_indices(csv_path):
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                done.add(int(row["image_idx"]))
    return done


def run_saliency(model, scheduler, device, dataset):
    """Callable directly from a notebook cell (with an already-loaded model)
    or from main() below (standalone script) -- same resumable logic either
    way, so progress survives whichever way this is invoked."""
    os.makedirs(OUT_DIR, exist_ok=True)
    vis_dir = os.path.join(OUT_DIR, "vis_arrays")
    os.makedirs(vis_dir, exist_ok=True)
    per_image_csv = os.path.join(OUT_DIR, "per_image_saliency.csv")
    sum_path = os.path.join(OUT_DIR, "running_sum.npy")
    count_path = os.path.join(OUT_DIR, "count.txt")

    n_total = len(dataset)
    print(f"{n_total} val images", flush=True)

    done = load_done_indices(per_image_csv)
    running_sum = np.load(sum_path) if os.path.exists(sum_path) else None
    count = int(open(count_path).read()) if os.path.exists(count_path) else 0

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

        phase, _ = dataset[i]
        fname = os.path.basename(dataset.image_paths[i])
        phase_b = phase.unsqueeze(0).to(device)

        sal = saliency_map(model, phase_b, scheduler, device).cpu().numpy()

        if running_sum is None:
            running_sum = np.zeros_like(sal)
        running_sum += sal
        count += 1

        if i < N_VIS:
            np.savez(os.path.join(vis_dir, f"{i}.npz"),
                     phase=phase.cpu().numpy(), sal=sal, fname=fname)

        writer.writerow({
            "image_idx": i, "file": fname,
            "mean_saliency": float(sal.mean()), "max_saliency": float(sal.max()),
        })
        fh.flush()
        n_done_now += 1

        if (i + 1) % 200 == 0 or (i + 1) == n_total:
            np.save(sum_path, running_sum)
            with open(count_path, "w") as cf:
                cf.write(str(count))
            np.save(os.path.join(OUT_DIR, "mean_saliency_map.npy"), running_sum / count)
            elapsed = time.time() - t0
            rate = n_done_now / elapsed if elapsed > 0 else 0
            print(f"image {i + 1}/{n_total} | {n_done_now} new this run | {rate:.2f} img/s", flush=True)

    np.save(sum_path, running_sum)
    with open(count_path, "w") as cf:
        cf.write(str(count))
    fh.close()

    mean_saliency_map = running_sum / count
    np.save(os.path.join(OUT_DIR, "mean_saliency_map.npy"), mean_saliency_map)
    print(f"DONE. Mean saliency map over {count} images -> "
          f"{os.path.join(OUT_DIR, 'mean_saliency_map.npy')}", flush=True)
    return mean_saliency_map


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scheduler = build_model_and_scheduler(device)
    load_checkpoint(model, CKPT_PATH, device)
    model.eval()
    dataset = VirtualStainingDataset(VAL_DIR)
    run_saliency(model, scheduler, device, dataset)


if __name__ == "__main__":
    main()
