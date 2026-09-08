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
from perturbations import perturb_contrast

dataset_name = 'polyps_v7'
PATH = pathlib.Path(r'./datasets') / dataset_name
SPLIT = 'val'

generator = Generator()
_ckpt = tf.train.Checkpoint(generator=generator)
_ckpt.restore(tf.train.latest_checkpoint('./training_checkpoints')).expect_partial()
print('Checkpoint restored')

CONTRAST_FACTORS = [-2.00, -1.50, -1.00, -0.50, -0.25] + [round(i * 0.05, 2) for i in range(0, 41)]

out_root = pathlib.Path('./contrast_sweep_output')
out_root.mkdir(parents=True, exist_ok=True)

loss_fn = lpips.LPIPS(net='alex', verbose=False)
image_fns = sorted((PATH / SPLIT).glob('*.tif'))
print(f'{len(image_fns)} val images found')


def cf_to_dirname(cf):
    return f'cf_{cf:+.2f}'.replace('.', '_').replace('+', 'p').replace('-', 'n')


def to_lpips_tensor(img_m11):
    t = tf.cast(tf.transpose(img_m11[tf.newaxis], [0, 3, 1, 2]), tf.float32)
    return torch.utils.dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(t))


cf_dirs = {}
for cf in CONTRAST_FACTORS:
    d = out_root / cf_to_dirname(cf)
    d.mkdir(parents=True, exist_ok=True)
    cf_dirs[cf] = d

per_cf_rows = {cf: [] for cf in CONTRAST_FACTORS}

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

    for cf in CONTRAST_FACTORS:
        cf_inp = perturb_contrast(inp, factor=cf)
        pred = generator(cf_inp[tf.newaxis], training=True)[0]

        pred_np = ((pred.numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(pred_np).save(cf_dirs[cf] / fn.name)

        pred_01 = pred * 0.5 + 0.5
        pred_t = to_lpips_tensor(pred)

        ssim_val = float(tf.image.ssim(pred_01, clean_01, max_val=1))
        psnr_val = float(tf.image.psnr(pred_01, clean_01, max_val=1))
        with torch.no_grad():
            lpips_val = float(loss_fn(pred_t, clean_t))

        per_cf_rows[cf].append({
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

plt.title('Contrast Robustness Sweep (Val Set) -- SSIM / LPIPS / PSNR vs Contrast Factor', fontsize=13)
plt.tight_layout()
plt.savefig(out_root / 'contrast_sweep_metrics.png', dpi=150, bbox_inches='tight')
plt.show()

print(df[['contrast_factor', 'ssim_mean', 'psnr_mean', 'lpips_mean']].to_string(index=False))
