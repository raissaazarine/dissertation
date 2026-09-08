import tensorflow as tf
import tensorflow_io as tfio
import torch
import lpips
import numpy as np
import pathlib
import csv
from timeit import default_timer as timer
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from pix2pix import Generator, INPUT_CHANNELS
from perturbations import perturb_gaussian

dataset_name = 'polyps_v7'
PATH = pathlib.Path(r'./datasets') / dataset_name
SPLIT = 'val'

generator = Generator()
_ckpt = tf.train.Checkpoint(generator=generator)
_ckpt.restore(tf.train.latest_checkpoint('./training_checkpoints')).expect_partial()
print('Checkpoint restored')

NOISE_STDS = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]

out_root = pathlib.Path('./gaussian_sweep_output')
out_root.mkdir(parents=True, exist_ok=True)

loss_fn = lpips.LPIPS(net='alex', verbose=False)
image_fns = sorted((PATH / SPLIT).glob('*.tif'))
print(f'{len(image_fns)} val images found')


def std_to_dirname(std):
    return f'std_{std:.2f}'.replace('.', '_')


def to_lpips_tensor(img_m11):
    t = tf.cast(tf.transpose(img_m11[tf.newaxis], [0, 3, 1, 2]), tf.float32)
    return torch.utils.dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(t))


std_dirs = {}
for std in NOISE_STDS:
    d = out_root / std_to_dirname(std)
    d.mkdir(parents=True, exist_ok=True)
    std_dirs[std] = d

per_std_rows = {std: [] for std in NOISE_STDS}

t0 = timer()
for i, fn in enumerate(image_fns):
    raw = tf.io.read_file(str(fn))
    image = tfio.experimental.image.decode_tiff(raw)
    half = tf.shape(image)[1] // 2

    inp = tf.cast(image[:, half:, :INPUT_CHANNELS], tf.float32)
    inp = (inp / 127.5) - 1.0

    clean_pred = generator(inp[tf.newaxis], training=True)[0]
    clean_01 = clean_pred * 0.5 + 0.5
    clean_t = to_lpips_tensor(clean_pred)

    for std in NOISE_STDS:
        noisy_inp = perturb_gaussian(inp, std=std)
        noisy_pred = generator(noisy_inp[tf.newaxis], training=True)[0]

        pred_np = ((noisy_pred.numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(pred_np).save(std_dirs[std] / fn.name)

        pred_01 = noisy_pred * 0.5 + 0.5
        pred_t = to_lpips_tensor(noisy_pred)

        ssim_val = float(tf.image.ssim(pred_01, clean_01, max_val=1))
        psnr_val = float(tf.image.psnr(pred_01, clean_01, max_val=1))
        with torch.no_grad():
            lpips_val = float(loss_fn(pred_t, clean_t))

        per_std_rows[std].append({
            'file': fn.name,
            'ssim': round(ssim_val, 6),
            'psnr': round(psnr_val, 4),
            'lpips': round(lpips_val, 6),
        })

    if (i + 1) % 50 == 0 or (i + 1) == len(image_fns):
        done = i + 1
        elapsed = timer() - t0
        rate = elapsed / done
        remaining = rate * (len(image_fns) - done)
        pct = 100 * done / len(image_fns)
        print(f'  {done}/{len(image_fns)} images ({pct:5.1f}%) | elapsed {elapsed/60:.1f} min | ETA {remaining/60:.1f} min')

summary_rows = []
for std in NOISE_STDS:
    rows = per_std_rows[std]

    per_csv = std_dirs[std] / 'metrics.csv'
    with open(per_csv, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=['file', 'ssim', 'psnr', 'lpips'])
        wr.writeheader()
        wr.writerows(rows)

    ssim_list = [r['ssim'] for r in rows]
    psnr_list = [r['psnr'] for r in rows]
    lpips_list = [r['lpips'] for r in rows]

    row = {
        'noise_std': std,
        'ssim_mean': round(float(np.mean(ssim_list)), 5),
        'ssim_std': round(float(np.std(ssim_list)), 5),
        'psnr_mean': round(float(np.mean(psnr_list)), 4),
        'psnr_std': round(float(np.std(psnr_list)), 4),
        'lpips_mean': round(float(np.mean(lpips_list)), 5),
        'lpips_std': round(float(np.std(lpips_list)), 5),
        'n_images': len(rows),
    }
    summary_rows.append(row)
    print(f"noise_std={std:.2f}  SSIM {row['ssim_mean']:.4f}  PSNR {row['psnr_mean']:.2f} dB  LPIPS {row['lpips_mean']:.4f}  (n={row['n_images']})")

summary_csv = out_root / 'summary.csv'
fieldnames = ['noise_std', 'ssim_mean', 'ssim_std', 'psnr_mean', 'psnr_std', 'lpips_mean', 'lpips_std', 'n_images']
with open(summary_csv, 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=fieldnames)
    wr.writeheader()
    wr.writerows(summary_rows)
print()
print(f'Summary saved -> {summary_csv}')

df = pd.read_csv(summary_csv)
std_vals = df['noise_std'].values

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

ax1.plot(std_vals, df['ssim_mean'], marker='o', color='tab:blue', linewidth=2, markersize=7, label='SSIM (higher=better)')
ax1.fill_between(std_vals, df['ssim_mean'] - df['ssim_std'], df['ssim_mean'] + df['ssim_std'], alpha=0.15, color='tab:blue')

ax1.plot(std_vals, df['lpips_mean'], marker='s', color='tab:orange', linewidth=2, markersize=7, label='LPIPS (lower=better)')
ax1.fill_between(std_vals, df['lpips_mean'] - df['lpips_std'], df['lpips_mean'] + df['lpips_std'], alpha=0.15, color='tab:orange')

ax2.plot(std_vals, df['psnr_mean'], marker='^', color='tab:green', linewidth=2, markersize=7, label='PSNR dB (higher=better)', linestyle='--')
ax2.fill_between(std_vals, df['psnr_mean'] - df['psnr_std'], df['psnr_mean'] + df['psnr_std'], alpha=0.15, color='tab:green')

ax1.axvline(0.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.8, label='Baseline (std=0)')

ax1.set_xlabel('Gaussian Noise Std', fontsize=12)
ax1.set_ylabel('SSIM / LPIPS', fontsize=12)
ax2.set_ylabel('PSNR (dB)', fontsize=12, color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')

ax1.set_xticks(std_vals)
ax1.set_xticklabels([f'{v:.2f}' for v in std_vals], fontsize=10)
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='lower left')

plt.title('Gaussian Noise Robustness Sweep (Val Set) -- SSIM / LPIPS / PSNR vs Noise Std', fontsize=13)
plt.tight_layout()
plt.savefig(out_root / 'gaussian_sweep_metrics.png', dpi=150, bbox_inches='tight')
plt.show()

print(df[['noise_std', 'ssim_mean', 'psnr_mean', 'lpips_mean']].to_string(index=False))
