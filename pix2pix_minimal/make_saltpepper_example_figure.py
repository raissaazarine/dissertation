"""Qualitative example figure for salt-and-pepper noise: same patch (idx=50)
used for the blur/contrast/gaussian examples, at 7 representative amounts,
plus ground truth and a genuinely fresh 'clean' prediction
(salt_pepper_sweep.py computes clean_pred as its own forward pass, separate
from amount=0.0's, so pix2pix's dropout-driven gap between the two is
visible here too).
"""
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
from PIL import Image

from pix2pix import Generator, INPUT_CHANNELS
from perturbations import perturb_salt_pepper

IDX = 50
AMOUNTS = [0.0, 0.0005, 0.001, 0.003, 0.005, 0.007, 0.01]

PATH = pathlib.Path('./datasets/polyps_v7/val')
PRED_ROOT = pathlib.Path('./salt_pepper_sweep_output')
OUT_PNG = pathlib.Path('./saltpepper_example_patch50.png')

generator = Generator()
_ckpt = tf.train.Checkpoint(generator=generator)
_ckpt.restore(tf.train.latest_checkpoint('./training_checkpoints')).expect_partial()

fn = PATH / f'{IDX}.tif'
raw = tf.io.read_file(str(fn))
image = tfio.experimental.image.decode_tiff(raw)
half = tf.shape(image)[1] // 2

gt = tf.cast(image[:, :half, :3], tf.float32).numpy() / 255.0
inp = tf.cast(image[:, half:, :INPUT_CHANNELS], tf.float32)
inp = (inp / 127.5) - 1.0


def to01(img_m11):
    a = np.clip(img_m11.numpy() * 0.5 + 0.5, 0, 1)
    return a[..., 0] if a.shape[-1] == 1 else a


def amt_dirname(a):
    return f'amt_{a:.4f}'.replace('.', '_')


clean_pred = generator(inp[tf.newaxis], training=True)[0]
clean_pred = to01(clean_pred)

col_labels = ['Ground\ntruth', 'Clean'] + [f'$a={a:g}$' for a in AMOUNTS]
n_cols = len(col_labels)

fig, axes = plt.subplots(2, n_cols, figsize=(2.0 * n_cols, 4.4))

axes[0, 0].axis('off')
axes[0, 1].imshow(to01(inp), cmap='gray', vmin=0, vmax=1)
tf.random.set_seed(0)
for j, a in enumerate(AMOUNTS):
    noisy = perturb_salt_pepper(inp, amount=a)
    axes[0, j + 2].imshow(to01(noisy), cmap='gray', vmin=0, vmax=1)

axes[1, 0].imshow(gt)
axes[1, 1].imshow(clean_pred)
for j, a in enumerate(AMOUNTS):
    pred = np.asarray(Image.open(PRED_ROOT / amt_dirname(a) / f'{IDX}.tif')) / 255.0
    axes[1, j + 2].imshow(pred)

row_labels = ['Phase image\n(salt & pepper)', 'pix2pix\nprediction']
for i in range(2):
    for j in range(n_cols):
        ax = axes[i, j]
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[i, 0].set_ylabel(row_labels[i], fontsize=11)

for j, label in enumerate(col_labels):
    axes[0, j].set_title(label, fontsize=12)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight')
print(f'Saved -> {OUT_PNG.resolve()}')
