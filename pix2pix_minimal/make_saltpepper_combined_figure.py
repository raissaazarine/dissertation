"""Combined qualitative salt-and-pepper figure for patch idx=50: ground
truth, each model's own clean prediction, and 7 representative amounts, for
both pix2pix and the diffusion model. Diffusion predictions are read from
../export_sp50_diff (exported elsewhere); ground truth and the phase input
come from the shared polyps_v7/val dataset used by both models. For the
diffusion model, 'Clean' and 'a=0' are the same image: sampling is seeded
per image and deterministic (eta=0), so the unperturbed and no-op-amount
predictions coincide by construction, unlike pix2pix where dropout stays
active at inference.
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
DIFF_ROOT = pathlib.Path('../export_sp50_diff')
OUT_PNG = pathlib.Path('./saltpepper_combined_patch50.png')

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


def load_tif01(path):
    return np.asarray(Image.open(path)) / 255.0


def amt_dirname(a):
    return f'amt_{a:.4f}'.replace('.', '_')


def diff_fname(a):
    return f'a{a:g}_50.tif'


clean_pix2pix = generator(inp[tf.newaxis], training=True)[0]
clean_pix2pix = to01(clean_pix2pix)
clean_diff = load_tif01(DIFF_ROOT / diff_fname(0.0))

col_labels = ['Ground\ntruth', 'Clean'] + [f'$a={a:g}$' for a in AMOUNTS]
n_cols = len(col_labels)

fig, axes = plt.subplots(3, n_cols, figsize=(2.0 * n_cols, 6.4))

axes[0, 0].axis('off')
axes[0, 1].imshow(to01(inp), cmap='gray', vmin=0, vmax=1)
tf.random.set_seed(0)
for j, a in enumerate(AMOUNTS):
    noisy = perturb_salt_pepper(inp, amount=a)
    axes[0, j + 2].imshow(to01(noisy), cmap='gray', vmin=0, vmax=1)

axes[1, 0].imshow(gt)
axes[1, 1].imshow(clean_pix2pix)
for j, a in enumerate(AMOUNTS):
    pred = load_tif01(PRED_ROOT / amt_dirname(a) / f'{IDX}.tif')
    axes[1, j + 2].imshow(pred)

axes[2, 0].imshow(gt)
axes[2, 1].imshow(clean_diff)
for j, a in enumerate(AMOUNTS):
    pred = load_tif01(DIFF_ROOT / diff_fname(a))
    axes[2, j + 2].imshow(pred)

row_labels = ['Phase image\n(salt & pepper)', 'pix2pix\nprediction', 'Diffusion\nprediction']
for i in range(3):
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
