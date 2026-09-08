import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt

from pix2pix import INPUT_CHANNELS

PATH = pathlib.Path('./datasets/polyps_v7/val')
CANDIDATES = [50, 150, 300, 450, 600, 800, 1000, 1250, 1500, 1800,
              2100, 2400, 2700, 3000, 3300, 3600, 3900, 4200, 4500, 5000]

def load_input(idx):
    fn = PATH / f'{idx}.tif'
    raw = tf.io.read_file(str(fn))
    image = tfio.experimental.image.decode_tiff(raw)
    half = tf.shape(image)[1] // 2
    inp = tf.cast(image[:, half:, :INPUT_CHANNELS], tf.float32)
    inp = (inp / 127.5) - 1.0
    return (inp.numpy()[..., 0] * 0.5 + 0.5)

n = len(CANDIDATES)
cols = 5
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
for ax, idx in zip(axes.flat, CANDIDATES):
    img = load_input(idx)
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'idx={idx}  std={img.std():.3f}', fontsize=9)
    ax.axis('off')
for ax in axes.flat[n:]:
    ax.axis('off')
plt.tight_layout()
plt.savefig('candidates_preview.png', dpi=130, bbox_inches='tight')
print('saved candidates_preview.png')
