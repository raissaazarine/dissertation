"""Contrast robustness sweep for the VS-Diff model. See gaussian_sweep.py for
the clean_pred/floor design and the assumptions to check before running
(dataset path, real checkpoint weights, STEPS/ETA)."""
import os
import pathlib
import csv
import torch
import numpy as np
import lpips
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
import pandas as pd
import matplotlib.pyplot as plt

from vsdiff_model import VirtualStainingDataset, build_model_and_scheduler, load_checkpoint, sample
from perturbations_torch import perturb_contrast
import virt_stain_utils2 as vsu

DATA_ROOT = '../datasets/polyps_v7'
SPLIT = 'val'
CHECKPOINT_PATH = './checkpoints/best.pth'
STEPS = 100
ETA = 0.0
MAX_IMAGES = None  # e.g. 20 for a quick sanity run; None = full split

CONTRAST_FACTORS = [-2.00, -1.50, -1.00, -0.50, -0.25] + [round(i * 0.05, 2) for i in range(0, 41)]
# 1.00 = no-op (initial-noise floor)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

out_root = pathlib.Path('./contrast_sweep_output')
out_root.mkdir(parents=True, exist_ok=True)

dataset = VirtualStainingDataset(str(pathlib.Path(DATA_ROOT) / SPLIT))
n_total = len(dataset) if MAX_IMAGES is None else min(MAX_IMAGES, len(dataset))
print(f'{len(dataset)} {SPLIT} images found, using {n_total}')

model, scheduler = build_model_and_scheduler(device)
load_checkpoint(model, CHECKPOINT_PATH, device)

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
loss_fn = lpips.LPIPS(net='alex', verbose=False).to(device)


def cf_to_dirname(cf):
    return f'cf_{cf:+.2f}'.replace('.', '_').replace('+', 'p').replace('-', 'n')


def save_pred(pred01, path):
    arr = (pred01.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)


cf_dirs = {}
for cf in CONTRAST_FACTORS:
    d = out_root / cf_to_dirname(cf)
    d.mkdir(parents=True, exist_ok=True)
    cf_dirs[cf] = d

per_cf_rows = {cf: [] for cf in CONTRAST_FACTORS}

for i in range(n_total):
    phase, _ = dataset[i]
    fname = os.path.basename(dataset.image_paths[i])
    phase = phase.unsqueeze(0).to(device)  # [1, 1, H, W]

    with torch.no_grad():
        clean_pred = sample(model, phase, scheduler, device, STEPS, ETA)[0]
    clean_01 = clean_pred.clamp(-1, 1) * 0.5 + 0.5

    for cf in CONTRAST_FACTORS:
        cf_phase = perturb_contrast(phase[0], factor=cf).unsqueeze(0)
        with torch.no_grad():
            pred = sample(model, cf_phase, scheduler, device, STEPS, ETA)[0]
        pred_01 = pred.clamp(-1, 1) * 0.5 + 0.5

        save_pred(pred_01, cf_dirs[cf] / fname)

        metrics = vsu.compute_metrics_batch(pred_01.unsqueeze(0), clean_01.unsqueeze(0), ssim_metric=ssim_metric)
        with torch.no_grad():
            lpips_val = float(loss_fn((pred_01 * 2 - 1).unsqueeze(0), (clean_01 * 2 - 1).unsqueeze(0)))

        per_cf_rows[cf].append({
            'file': fname,
            'ssim': round(metrics['ssim'][0], 6),
            'psnr': round(metrics['psnr'][0], 4),
            'lpips': round(lpips_val, 6),
        })

    if (i + 1) % 10 == 0 or (i + 1) == n_total:
        print(f'  {i + 1}/{n_total} images')

summary_rows = []
for cf in CONTRAST_FACTORS:
    rows = per_cf_rows[cf]

    per_csv = cf_dirs[cf] / 'metrics.csv'
    with open(per_csv, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=['file', 'ssim', 'psnr', 'lpips'])
        wr.writeheader()
        wr.writerows(rows)

    ssim_list = [r['ssim'] for r in rows]
    psnr_list = [r['psnr'] for r in rows]
    lpips_list = [r['lpips'] for r in rows]

    row = {
        'contrast_factor': cf,
        'ssim_mean': round(float(np.mean(ssim_list)), 5),
        'ssim_std': round(float(np.std(ssim_list)), 5),
        'psnr_mean': round(float(np.mean(psnr_list)), 4),
        'psnr_std': round(float(np.std(psnr_list)), 4),
        'lpips_mean': round(float(np.mean(lpips_list)), 5),
        'lpips_std': round(float(np.std(lpips_list)), 5),
        'n_images': len(rows),
    }
    summary_rows.append(row)
    print(f"contrast_factor={cf:+.2f}  SSIM {row['ssim_mean']:.4f}  PSNR {row['psnr_mean']:.2f} dB  LPIPS {row['lpips_mean']:.4f}  (n={row['n_images']})")

summary_rows.sort(key=lambda r: r['contrast_factor'])
summary_csv = out_root / 'summary.csv'
fieldnames = ['contrast_factor', 'ssim_mean', 'ssim_std', 'psnr_mean', 'psnr_std', 'lpips_mean', 'lpips_std', 'n_images']
with open(summary_csv, 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=fieldnames)
    wr.writeheader()
    wr.writerows(summary_rows)
print()
print(f'Summary saved -> {summary_csv}')

# --- Plot ---
df = pd.read_csv(summary_csv)
df = df.sort_values('contrast_factor').reset_index(drop=True)
cf = df['contrast_factor'].values

fig, ax1 = plt.subplots(figsize=(14, 5))
ax2 = ax1.twinx()

ax1.plot(cf, df['ssim_mean'], marker='o', color='tab:blue', linewidth=2, markersize=5, label='SSIM (higher=better)')
ax1.fill_between(cf, df['ssim_mean'] - df['ssim_std'], df['ssim_mean'] + df['ssim_std'], alpha=0.15, color='tab:blue')

ax1.plot(cf, df['lpips_mean'], marker='s', color='tab:orange', linewidth=2, markersize=5, label='LPIPS (lower=better)')
ax1.fill_between(cf, df['lpips_mean'] - df['lpips_std'], df['lpips_mean'] + df['lpips_std'], alpha=0.15, color='tab:orange')

ax2.plot(cf, df['psnr_mean'], marker='^', color='tab:green', linewidth=2, markersize=5, label='PSNR dB (higher=better)', linestyle='--')
ax2.fill_between(cf, df['psnr_mean'] - df['psnr_std'], df['psnr_mean'] + df['psnr_std'], alpha=0.15, color='tab:green')

ax1.axvline(1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.8, label='Baseline (factor=1)')

ax1.set_xlabel('Contrast Factor', fontsize=12)
ax1.set_ylabel('SSIM / LPIPS', fontsize=12)
ax2.set_ylabel('PSNR (dB)', fontsize=12, color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')

ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='lower left')

plt.title('VS-Diff Contrast Robustness Sweep (Val Set) -- SSIM / LPIPS / PSNR vs Contrast Factor', fontsize=13)
plt.tight_layout()
plt.savefig(out_root / 'contrast_sweep_metrics.png', dpi=150, bbox_inches='tight')
plt.show()

print(df[['contrast_factor', 'ssim_mean', 'psnr_mean', 'lpips_mean']].to_string(index=False))
