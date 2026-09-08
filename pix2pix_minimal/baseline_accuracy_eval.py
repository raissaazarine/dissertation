"""Baseline accuracy of the checkpoint against the chemically stained ground truth,
over the full held-out set (no perturbation). This is the reference-point evaluation
for Section res-robustness, distinct from the amount=0.0 dropout-floor run (which
compares two predictions to each other, not to the ground truth).

Run with: pixi run python baseline_accuracy_eval.py
"""
import csv
import pathlib

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import torch
import lpips

from pix2pix import Generator, INPUT_CHANNELS

dataset_name = 'polyps_v7'
PATH = pathlib.Path(r'./datasets') / dataset_name
SPLIT = 'val'

out_root = pathlib.Path('./baseline_accuracy_output')
out_root.mkdir(parents=True, exist_ok=True)

generator = Generator()
_ckpt = tf.train.Checkpoint(generator=generator)
_ckpt.restore(tf.train.latest_checkpoint('./training_checkpoints')).expect_partial()
print('Checkpoint restored', flush=True)

loss_fn = lpips.LPIPS(net='alex', verbose=False)
image_fns = sorted((PATH / SPLIT).glob('*.tif'))
print(f'{len(image_fns)} val images found', flush=True)


def to_lpips_tensor(img_m11):
    t = tf.cast(tf.transpose(img_m11[tf.newaxis], [0, 3, 1, 2]), tf.float32)
    return torch.utils.dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(t))


per_csv = out_root / 'metrics.csv'
done_files = set()
if per_csv.exists():
    with open(per_csv, newline='') as fh:
        for row in csv.DictReader(fh):
            done_files.add(row['file'])
    print(f'resuming: {len(done_files)} images already done', flush=True)

write_header = not per_csv.exists()
out_fh = open(per_csv, 'a', newline='')
writer = csv.DictWriter(out_fh, fieldnames=['file', 'ssim', 'psnr', 'lpips'])
if write_header:
    writer.writeheader()
    out_fh.flush()

rows = []
for i, fn in enumerate(image_fns):
    if fn.name in done_files:
        continue
    raw = tf.io.read_file(str(fn))
    image = tfio.experimental.image.decode_tiff(raw)
    half = tf.shape(image)[1] // 2

    gt = tf.cast(image[:, :half, :3], tf.float32)
    gt = (gt / 127.5) - 1.0
    inp = tf.cast(image[:, half:, :INPUT_CHANNELS], tf.float32)
    inp = (inp / 127.5) - 1.0

    pred = generator(inp[tf.newaxis], training=True)[0]

    gt_01 = gt * 0.5 + 0.5
    pred_01 = pred * 0.5 + 0.5
    gt_t = to_lpips_tensor(gt)
    pred_t = to_lpips_tensor(pred)

    ssim_val = float(tf.image.ssim(pred_01, gt_01, max_val=1))
    psnr_val = float(tf.image.psnr(pred_01, gt_01, max_val=1))
    with torch.no_grad():
        lpips_val = float(loss_fn(pred_t, gt_t))

    row = {
        'file': fn.name,
        'ssim': round(ssim_val, 6),
        'psnr': round(psnr_val, 4),
        'lpips': round(lpips_val, 6),
    }
    rows.append(row)
    writer.writerow(row)
    out_fh.flush()

    if (i + 1) % 200 == 0 or (i + 1) == len(image_fns):
        print(f'  {i + 1}/{len(image_fns)} images', flush=True)

out_fh.close()

with open(per_csv, newline='') as fh:
    all_rows = list(csv.DictReader(fh))
ssim_list = [float(r['ssim']) for r in all_rows]
psnr_list = [float(r['psnr']) for r in all_rows]
lpips_list = [float(r['lpips']) for r in all_rows]

print()
print(f"SSIM  mean={np.mean(ssim_list):.4f}  std={np.std(ssim_list):.4f}")
print(f"PSNR  mean={np.mean(psnr_list):.2f} dB  std={np.std(psnr_list):.2f}")
print(f"LPIPS mean={np.mean(lpips_list):.4f}  std={np.std(lpips_list):.4f}")
print(f"n={len(rows)}")
print(f'saved -> {per_csv}', flush=True)
