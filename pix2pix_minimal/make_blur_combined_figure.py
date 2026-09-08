"""Combined qualitative blur-degradation figure for patch idx=50: ground
truth, a genuinely fresh 'clean' prediction (not reused from kernel_size=1,
so pix2pix's dropout-driven gap between the two is visible), and the
kernel_size in {1,3,5,7,11,15,21} sweep, for both pix2pix and the diffusion
model. Diffusion predictions are read from ../export_blur50_diff (exported
elsewhere, since this checkpoint copy is not usable locally); ground truth
and the phase input come from the shared polyps_v7/val dataset used by both
models. For the diffusion model, 'Clean' and 'k=1' are the same image: its
sampling is seeded per-image and deterministic (eta=0), so the unperturbed
and no-op-kernel predictions coincide by construction, unlike pix2pix where
dropout stays active at inference.
"""
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
from PIL import Image

from pix2pix import Generator, INPUT_CHANNELS
from perturbations import perturb_blur

IDX = 50
KERNELS = [1, 3, 5, 7, 11, 15, 21]

PATH = pathlib.Path('./datasets/polyps_v7/val')
PRED_ROOT = pathlib.Path('./blur_sweep_output')
DIFF_ROOT = pathlib.Path('../export_blur50_diff')
OUT_PNG = pathlib.Path('./blur_combined_patch50.png')

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


clean_pix2pix = generator(inp[tf.newaxis], training=True)[0]
clean_pix2pix = to01(clean_pix2pix)

clean_diff = load_tif01(DIFF_ROOT / 'k1_50.tif')

col_labels = ['Ground\ntruth', 'Clean'] + [f'$k={k}$' for k in KERNELS]
n_cols = len(col_labels)

fig, axes = plt.subplots(3, n_cols, figsize=(2.0 * n_cols, 6.4))

axes[0, 0].axis('off')
axes[0, 1].imshow(to01(inp), cmap='gray', vmin=0, vmax=1)
for j, k in enumerate(KERNELS):
    blurred = perturb_blur(inp, kernel_size=k)
    axes[0, j + 2].imshow(to01(blurred), cmap='gray', vmin=0, vmax=1)

axes[1, 0].imshow(gt)
axes[1, 1].imshow(clean_pix2pix)
for j, k in enumerate(KERNELS):
    pred = load_tif01(PRED_ROOT / f'kernel_{k}' / f'{IDX}.tif')
    axes[1, j + 2].imshow(pred)

axes[2, 0].imshow(gt)
axes[2, 1].imshow(clean_diff)
for j, k in enumerate(KERNELS):
    pred = load_tif01(DIFF_ROOT / f'k{k}_50.tif')
    axes[2, j + 2].imshow(pred)

row_labels = ['Phase image\n(blurred)', 'pix2pix\nprediction', 'Diffusion\nprediction']
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
