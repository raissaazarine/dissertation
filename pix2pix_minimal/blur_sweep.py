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
from perturbations import perturb_blur

dataset_name = 'polyps_v7'
PATH = pathlib.Path(r'./datasets') / dataset_name
SPLIT = 'val'

generator = Generator()
_ckpt = tf.train.Checkpoint(generator=generator)
_ckpt.restore(tf.train.latest_checkpoint('./training_checkpoints')).expect_partial()
print('Checkpoint restored')

BLUR_KERNEL_SIZES = [1, 3, 5, 7, 11, 15, 21]

out_root = pathlib.Path('./blur_sweep_output')
out_root.mkdir(parents=True, exist_ok=True)

loss_fn = lpips.LPIPS(net='alex', verbose=False)
image_fns = sorted((PATH / SPLIT).glob('*.tif'))
print(f'{len(image_fns)} val images found')


def kernel_dirname(ks):
    return f'kernel_{ks}'


def to_lpips_tensor(img_m11):
    t = tf.cast(tf.transpose(img_m11[tf.newaxis], [0, 3, 1, 2]), tf.float32)
    return torch.utils.dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(t))


kernel_dirs = {}
for ks in BLUR_KERNEL_SIZES:
    d = out_root / kernel_dirname(ks)
    d.mkdir(parents=True, exist_ok=True)
    kernel_dirs[ks] = d

per_kernel_rows = {ks: [] for ks in BLUR_KERNEL_SIZES}

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

    for ks in BLUR_KERNEL_SIZES:
        blurred_inp = perturb_blur(inp, kernel_size=ks)
        pred = generator(blurred_inp[tf.newaxis], training=True)[0]

        pred_np = ((pred.numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(pred_np).save(kernel_dirs[ks] / fn.name)

        pred_01 = pred * 0.5 + 0.5
        pred_t = to_lpips_tensor(pred)

        ssim_val = float(tf.image.ssim(pred_01, clean_01, max_val=1))
        psnr_val = float(tf.image.psnr(pred_01, clean_01, max_val=1))
        with torch.no_grad():
            lpips_val = float(loss_fn(pred_t, clean_t))

        per_kernel_rows[ks].append({
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
for ks in BLUR_KERNEL_SIZES:
    rows = per_kernel_rows[ks]

    per_csv = kernel_dirs[ks] / 'metrics.csv'
    with open(per_csv, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=['file', 'ssim', 'psnr', 'lpips'])
        wr.writeheader()
        wr.writerows(rows)

    ssim_list = [r['ssim'] for r in rows]
    psnr_list = [r['psnr'] for r in rows]
    lpips_list = [r['lpips'] for r in rows]

    row = {
        'kernel_size': ks,
        'ssim_mean': round(float(np.mean(ssim_list)), 5),
        'ssim_std': round(float(np.std(ssim_list)), 5),
        'psnr_mean': round(float(np.mean(psnr_list)), 4),
        'psnr_std': round(float(np.std(psnr_list)), 4),
        'lpips_mean': round(float(np.mean(lpips_list)), 5),
        'lpips_std': round(float(np.std(lpips_list)), 5),
        'n_images': len(rows),
    }
    summary_rows.append(row)
    print(f"kernel_size={ks}  SSIM {row['ssim_mean']:.4f}  PSNR {row['psnr_mean']:.2f} dB  LPIPS {row['lpips_mean']:.4f}  (n={row['n_images']})")

summary_csv = out_root / 'summary.csv'
fieldnames = ['kernel_size', 'ssim_mean', 'ssim_std', 'psnr_mean', 'psnr_std', 'lpips_mean', 'lpips_std', 'n_images']
with open(summary_csv, 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=fieldnames)
    wr.writeheader()
    wr.writerows(summary_rows)
print()
print(f'Summary saved -> {summary_csv}')

df = pd.read_csv(summary_csv)
ks = df['kernel_size'].values

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.plot(ks, df['ssim_mean'], marker='o', color='tab:blue', linewidth=2, markersize=7, label='SSIM (higher=better)')
ax1.fill_between(ks, df['ssim_mean'] - df['ssim_std'], df['ssim_mean'] + df['ssim_std'], alpha=0.15, color='tab:blue')

ax1.plot(ks, df['lpips_mean'], marker='s', color='tab:orange', linewidth=2, markersize=7, label='LPIPS (lower=better)')
ax1.fill_between(ks, df['lpips_mean'] - df['lpips_std'], df['lpips_mean'] + df['lpips_std'], alpha=0.15, color='tab:orange')

ax2.plot(ks, df['psnr_mean'], marker='^', color='tab:green', linewidth=2, markersize=7, label='PSNR dB (higher=better)', linestyle='--')
ax2.fill_between(ks, df['psnr_mean'] - df['psnr_std'], df['psnr_mean'] + df['psnr_std'], alpha=0.15, color='tab:green')

ax1.set_xlabel('Blur Kernel Size', fontsize=12)
ax1.set_ylabel('SSIM / LPIPS', fontsize=12)
ax2.set_ylabel('PSNR (dB)', fontsize=12, color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')

ax1.set_xticks(ks)
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='lower left')

plt.title('Blur Robustness Sweep (Val Set) -- SSIM / LPIPS / PSNR vs Kernel Size', fontsize=13)
plt.tight_layout()
plt.savefig(out_root / 'blur_sweep_metrics.png', dpi=150, bbox_inches='tight')
plt.show()

print(df.to_string(index=False))
