"""Full-val-set (7373 images) perturbation robustness sweep: Gaussian noise,
blur, salt & pepper, contrast. Estimated ~98 hours total (measured: one
50-step DDIM sample takes ~0.696s on this GPU) -- designed to run unattended
for days and survive interruption.

Seeded sampling: the clean baseline and every perturbation level for a given
image are sampled with the SAME DDIM starting noise (seed=image_idx, via
ddim_sample_seeded) instead of an independent random draw each time. This
isolates the perturbation's effect from the sampler's own initial-noise
randomness -- e.g. the no-op level (noise_std=0.00, kernel_size=1, ...) now
matches the clean prediction essentially exactly (SSIM~1.0), instead of only
reflecting how much two independent samples of the same input naturally
differ (that sampler-variance question is what the uncertainty analysis
answers instead, via genuinely independent stochastic samples, eta>0).

Saves both metrics and images: each (image, level) prediction is written to
sweep_output_seeded/<name>/<level>/<file>.png, alongside the SSIM/PSNR/LPIPS
row. The unperturbed baseline prediction is computed ONCE per image and
shared across all 4 sweep types via sweep_output_seeded/clean/ (not
recomputed per sweep -- same input, same STEPS, so there's nothing
sweep-specific about it).

Output lives under sweep_output_seeded/ (NOT sweep_output/) -- that old
directory holds results from before seeded sampling was added (unseeded:
clean vs. no-op-level predictions used independent random starting noise, so
they didn't match exactly) and is left untouched as a historical record, not
read or written by this version of the script.

Resumability: every (image, level) result is appended to
sweep_output_seeded/<name>/per_image_metrics.csv as soon as it's computed. On
restart, already-computed (image, level) pairs are skipped, so killing this
process and re-running it loses at most the one row in flight.

Run with: vsdiff_env/bin/python full_sweep.py [gaussian|blur|salt_pepper|contrast|all]
"""
import csv
import os
import sys
import time

import numpy as np
import torch
import lpips as lpips_lib
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure

sys.path.insert(0, os.path.dirname(__file__))
import virt_stain_utils2 as vsu
from vsdiff_model import build_model_and_scheduler, load_checkpoint, VirtualStainingDataset
from perturbations_torch import perturb_gaussian, perturb_blur, perturb_salt_pepper, perturb_contrast

BASE_DIR = "/cs/student/project_msc/2025/aibh/razarine"
CKPT_PATH = os.path.join(BASE_DIR, "runs/exp35/checkpoints/best.pth")
VAL_DIR = os.path.join(BASE_DIR, "datasets/polyps_v7/val")
STEPS = 50
ETA = 0.0

# New output root for the seeded design -- kept separate from the old
# sweep_output/ (unseeded) so the two are never mixed or confused.
OUT_ROOT = os.path.join(BASE_DIR, "sweep_output_seeded")

# One clean (unperturbed) prediction per image, shared across all 4 sweep
# types -- avoids recomputing the same 50-step DDIM sample 4x per image
# (gaussian/blur/salt_pepper/contrast each used to compute their own).
SHARED_CLEAN_DIR = os.path.join(OUT_ROOT, "clean")

SWEEPS = {
    "gaussian":    dict(fn=perturb_gaussian,     levels=[0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00], level_name="noise_std"),
    "blur":        dict(fn=perturb_blur,         levels=[1, 3, 5, 7, 11, 15, 21], level_name="kernel_size"),
    "salt_pepper": dict(fn=perturb_salt_pepper,  levels=[0.0, 0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.003, 0.005, 0.007, 0.01], level_name="amount"),
    "contrast":    dict(fn=perturb_contrast,     levels=[round(i * 0.05, 2) for i in range(0, 41)], level_name="contrast_factor"),
}


def denorm01(x):
    return x.clamp(-1, 1) * 0.5 + 0.5


def ddim_sample_seeded(model, phase, scheduler, device, seed, num_inference_steps, eta):
    """Same reverse process as vsu.ddim_sample_full, but the starting noise is
    drawn from a generator seeded per-image (seed=image_idx) instead of the
    global unseeded RNG. Reusing the same seed for an image's clean prediction
    and every one of its perturbation levels means the ONLY thing that differs
    between rows is the perturbed input itself -- not an independent random
    starting point too. (Only matters for eta=0.0, used here: eta>0 adds fresh
    noise at every reverse step regardless of the starting point.)"""
    model.eval()
    B = phase.size(0)
    g = torch.Generator(device=device).manual_seed(seed)
    sample = torch.randn(B, 3, phase.shape[2], phase.shape[3], device=device, generator=g)

    scheduler.set_timesteps(num_inference_steps, device=device)
    for t in scheduler.timesteps:
        inp = torch.cat([sample, phase], dim=1)
        out = model(inp, t)
        noise_pred = out.sample if hasattr(out, "sample") else out
        try:
            step_output = scheduler.step(noise_pred, t, sample, eta=eta)
        except TypeError:
            step_output = scheduler.step(noise_pred, t, sample)
        sample = step_output.prev_sample if hasattr(step_output, "prev_sample") else step_output["prev_sample"]
    model.train()
    return sample


def level_dirname(lv):
    return str(lv).replace('.', '_').replace('-', 'n')


def save_pred(pred01, path):
    arr = (pred01.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)


def get_or_compute_clean(model, phase, scheduler, device, fname, seed):
    """Loads the shared clean prediction for this image if another sweep
    already computed it; otherwise samples it once (50 steps, seeded on the
    image index) and saves it to SHARED_CLEAN_DIR for every other sweep type
    to reuse. Deterministic given (model, phase, seed), so loading a
    previously-saved file is exactly equivalent to recomputing it."""
    path = os.path.join(SHARED_CLEAN_DIR, fname)
    if os.path.exists(path):
        arr = np.asarray(Image.open(path)).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).to(device)

    with torch.no_grad():
        clean_pred = ddim_sample_seeded(model, phase, scheduler, device, seed,
                                         num_inference_steps=STEPS, eta=ETA)[0]
    clean_01 = denorm01(clean_pred)
    os.makedirs(SHARED_CLEAN_DIR, exist_ok=True)
    save_pred(clean_01, path)
    return clean_01


def load_done_pairs(csv_path):
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                done.add((int(row["image_idx"]), row["level"]))
    return done


def rebuild_summary(name, out_dir, per_image_csv, cfg):
    """(Re)builds summary.csv from whatever rows are in per_image_csv so far --
    called periodically during a run (partial progress) and once more at the
    end (final), so the notebook can plot live progress mid-run."""
    rows_by_level = {}
    with open(per_image_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            rows_by_level.setdefault(row["level"], []).append(row)

    summary_rows = []
    for lv in cfg["levels"]:
        rows = rows_by_level.get(str(lv), [])
        if not rows:
            continue
        ssim_l = [float(r["ssim"]) for r in rows]
        psnr_l = [float(r["psnr"]) for r in rows]
        lpips_l = [float(r["lpips"]) for r in rows]
        summary_rows.append({
            cfg["level_name"]: lv,
            "ssim_mean": float(np.mean(ssim_l)), "ssim_std": float(np.std(ssim_l)),
            "psnr_mean": float(np.mean(psnr_l)), "psnr_std": float(np.std(psnr_l)),
            "lpips_mean": float(np.mean(lpips_l)), "lpips_std": float(np.std(lpips_l)),
            "n_images": len(rows),
        })
    summary_csv = os.path.join(out_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=[cfg["level_name"], "ssim_mean", "ssim_std",
                                             "psnr_mean", "psnr_std", "lpips_mean", "lpips_std", "n_images"])
        wr.writeheader()
        wr.writerows(summary_rows)
    return summary_csv


def run_sweep(name, model, scheduler, device, ssim_metric, lpips_fn, dataset):
    cfg = SWEEPS[name]
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    per_image_csv = os.path.join(out_dir, "per_image_metrics.csv")
    fieldnames = ["image_idx", "file", "level", "ssim", "psnr", "lpips"]

    level_dirs = {lv: os.path.join(out_dir, level_dirname(lv)) for lv in cfg["levels"]}
    for d in level_dirs.values():
        os.makedirs(d, exist_ok=True)

    done = load_done_pairs(per_image_csv)
    write_header = not os.path.exists(per_image_csv)
    fh = open(per_image_csv, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        fh.flush()

    n_total = len(dataset)
    t0 = time.time()
    n_done_now = 0
    for i in range(n_total):
        levels_needed = [lv for lv in cfg["levels"] if (i, str(lv)) not in done]
        if not levels_needed:
            continue

        phase, _ = dataset[i]
        fname = os.path.basename(dataset.image_paths[i])
        phase = phase.unsqueeze(0).to(device)

        # same seed (image index) for the clean baseline and every perturbation
        # level of this image -- isolates the perturbation's effect from the
        # sampler's own initial-noise randomness (see ddim_sample_seeded)
        clean_01 = get_or_compute_clean(model, phase, scheduler, device, fname, seed=i)

        for lv in levels_needed:
            pert_phase = cfg["fn"](phase[0], lv).unsqueeze(0)
            with torch.no_grad():
                pred = ddim_sample_seeded(model, pert_phase, scheduler, device, i,
                                           num_inference_steps=STEPS, eta=ETA)[0]
            pred_01 = denorm01(pred)
            save_pred(pred_01, os.path.join(level_dirs[lv], fname))

            m = vsu.compute_metrics_batch(pred_01.unsqueeze(0), clean_01.unsqueeze(0), ssim_metric=ssim_metric)
            with torch.no_grad():
                lp = float(lpips_fn((pred_01 * 2 - 1).unsqueeze(0), (clean_01 * 2 - 1).unsqueeze(0)))

            writer.writerow({
                "image_idx": i, "file": fname, "level": lv,
                "ssim": m["ssim"][0], "psnr": m["psnr"][0], "lpips": lp,
            })
            fh.flush()
            n_done_now += 1

        if (i + 1) % 25 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t0
            rate = n_done_now / elapsed if elapsed > 0 else 0
            print(f"[{name}] image {i + 1}/{n_total} | {n_done_now} new rows this run | "
                  f"{rate:.2f} rows/s | elapsed {elapsed/60:.1f} min", flush=True)
            rebuild_summary(name, out_dir, per_image_csv, cfg)

    fh.close()
    summary_csv = rebuild_summary(name, out_dir, per_image_csv, cfg)
    print(f"[{name}] DONE. summary -> {summary_csv}", flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(SWEEPS.keys()) if which == "all" else [which]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scheduler = build_model_and_scheduler(device)
    load_checkpoint(model, CKPT_PATH, device)
    model.eval()

    dataset = VirtualStainingDataset(VAL_DIR)
    print(f"{len(dataset)} val images", flush=True)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips_lib.LPIPS(net="alex", verbose=False).to(device)

    for name in names:
        print(f"=== starting sweep: {name} ===", flush=True)
        run_sweep(name, model, scheduler, device, ssim_metric, lpips_fn, dataset)


if __name__ == "__main__":
    main()
