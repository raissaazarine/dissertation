"""Builds a qualitative example figure: one held-out patch (idx=0) shown as
phase input -> pix2pix prediction, across the blur kernel sizes swept in
blur_sweep.py (1, 3, 5, 7, 11, 15, 21). Reuses the already-computed
predictions in blur_sweep_output/kernel_*/0.tif and regenerates the
perturbed *input* on the fly with the exact same perturb_blur() used for
the sweep, so the input row is faithful to what actually produced those
predictions.
"""
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
from PIL import Image

from pix2pix import INPUT_CHANNELS
from perturbations import perturb_blur

IDX = 50
KERNELS = [1, 3, 5, 7, 11, 15, 21]

PATH = pathlib.Path('./datasets/polyps_v7/val')
PRED_ROOT = pathlib.Path('./blur_sweep_output')
OUT_PNG = pathlib.Path('./blur_example_patch.png')

fn = PATH / f'{IDX}.tif'
raw = tf.io.read_file(str(fn))
image = tfio.experimental.image.decode_tiff(raw)
half = tf.shape(image)[1] // 2

target = tf.cast(image[:, :half, :3], tf.float32) / 255.0
inp = tf.cast(image[:, half:, :INPUT_CHANNELS], tf.float32)
inp = (inp / 127.5) - 1.0

def to01(img_m11):
    return np.clip((img_m11.numpy() * 0.5 + 0.5), 0, 1)

n = len(KERNELS)
fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.6))

for j, k in enumerate(KERNELS):
    blurred_inp = perturb_blur(inp, kernel_size=k)
    inp_img = to01(blurred_inp)
    if inp_img.shape[-1] == 1:
        inp_img = inp_img[..., 0]

    pred_path = PRED_ROOT / f'kernel_{k}' / f'{IDX}.tif'
    pred_img = np.asarray(Image.open(pred_path)) / 255.0

    axes[0, j].imshow(inp_img, cmap='gray', vmin=0, vmax=1)
    axes[0, j].set_title(f'$k={k}$', fontsize=12)
    axes[0, j].axis('off')

    axes[1, j].imshow(pred_img, vmin=0, vmax=1)
    axes[1, j].axis('off')

axes[0, 0].set_ylabel('Phase image', fontsize=11)
axes[1, 0].set_ylabel('pix2pix prediction', fontsize=11)
for row, label in [(0, 'Phase image\n(blurred)'), (1, 'pix2pix\nprediction')]:
    axes[row, 0].axis('on')
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])
    for spine in axes[row, 0].spines.values():
        spine.set_visible(False)
    axes[row, 0].set_ylabel(label, fontsize=11)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight')
print(f'Saved -> {OUT_PNG.resolve()}')
