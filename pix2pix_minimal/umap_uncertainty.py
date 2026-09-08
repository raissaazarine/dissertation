"""
Uncertainty structure via UMAP: embeds per-pixel MC-Dropout prediction stacks and
colours by predicted brightness / tissue class (from ground truth) / uncertainty
(predictive variance) / confidence (median split of uncertainty).

Three parts, run in order (part 2 caches to disk so part 3 can reuse it without
repeating the slow whole-val-set feature extraction):
  1. Per-sample 4-panel UMAP for the 3 fixed samples.
  2. Whole-val-set UMAP (pooled across all val images, subsampled per image).
  3. Point-count vs. embedding-spread summary per tissue class, reusing part 2's cache.

NB: needs `umap-learn`. In this environment (numpy 1.22.4, pinned by
tensorflow 2.8) the compatible combo is:
    pip install "umap-learn==0.5.5" "scikit-learn<1.4" "numba<0.57" --no-deps
(newer umap-learn/scikit-learn/numba wheels are built against numpy>=1.24/2.0
and fail to import against this numpy version.)
"""
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import pathlib
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from scipy.spatial import ConvexHull

from pix2pix import Generator, INPUT_CHANNELS

dataset_name = 'polyps_v7'
PATH = pathlib.Path(r'./datasets') / dataset_name
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

out_root = pathlib.Path('./uncertainty')
out_root.mkdir(parents=True, exist_ok=True)


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tfio.experimental.image.decode_tiff(image)
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, w:, :3]
    real_image = image[:, :w, :3]
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    return input_image, real_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    input_image = input_image[:, :, :INPUT_CHANNELS]
    return input_image, real_image


val_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.tif'), shuffle=True, seed=42)
val_dataset = val_dataset.map(load_image_test)
val_dataset = val_dataset.batch(BATCH_SIZE)

generator = Generator()
_ckpt = tf.train.Checkpoint(generator=generator)
_ckpt.restore(tf.train.latest_checkpoint('./training_checkpoints')).expect_partial()
print('Checkpoint restored')

N_SAMPLES = 3
fixed_samples = [next(iter(val_dataset)) for _ in range(N_SAMPLES)]

STRIDE = 2
N_RUNS = 20

TISSUE_NAMES = ['Nucleus', 'Cytoplasm & stroma', 'Background/lumen']
TISSUE_COLORS = ['#7c1fd6', '#ff6fae', '#9a9a9a']
COLOUR_NOTE = ('H&E colour key  ->  purple = nucleus (hematoxylin-stained chromatin)   |   '
               'pink = cytoplasm & stroma (eosin-stained protein/collagen)   |   '
               'grey/white = background or gland lumen (no tissue)')

CONF_NAMES = ['High-confidence', 'Low-confidence']
CONF_COLORS = ['#9ecae1', '#08306b']


def classify_tissue(gt01):
    """Rough 3-way H&E tissue labelling from ground-truth colour (brightness percentile):
    darkest 25% -> nucleus, lightest 25% -> background/lumen, rest -> cytoplasm/stroma."""
    v = mcolors.rgb_to_hsv(gt01)[..., 2]
    p25, p75 = np.percentile(v, [25, 75])
    labels = np.full(v.shape, 1, dtype=np.int32)
    labels[v <= p25] = 0
    labels[v >= p75] = 2
    return labels


def mc_dropout_stack(model, test_input, n_runs=N_RUNS):
    """Like mc_dropout_uncertainty, but keeps every run's prediction (not just mean/var)."""
    preds = [model(test_input, training=True)[0].numpy() for _ in range(n_runs)]
    return np.stack(preds, axis=0)


for idx, (test_input, ground_truth) in enumerate(fixed_samples):
    stack01 = mc_dropout_stack(generator, test_input, N_RUNS) * 0.5 + 0.5

    gt01 = (ground_truth[0].numpy() * 0.5 + 0.5).clip(0, 1)
    labels = classify_tissue(gt01)
    uncertainty = stack01.var(axis=0).mean(axis=-1)
    mean_v = mcolors.rgb_to_hsv(stack01.mean(axis=0))[..., 2]

    sub = stack01[:, ::STRIDE, ::STRIDE, :]
    n, h, w, c = sub.shape
    X = sub.transpose(1, 2, 0, 3).reshape(h * w, n * c)
    y_tissue = labels[::STRIDE, ::STRIDE].reshape(-1)
    y_uncert = uncertainty[::STRIDE, ::STRIDE].reshape(-1)
    y_meanv = mean_v[::STRIDE, ::STRIDE].reshape(-1)

    print(f'sample {idx + 1}: {X.shape[0]} points, feature dim {X.shape[1]}')

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42, verbose=False)
    embedding = reducer.fit_transform(X)

    uncert_median = np.median(y_uncert)
    y_confidence = (y_uncert >= uncert_median).astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(15, 14))
    for ax, lbl in zip(axes.flat, ['a', 'b', 'c', 'd']):
        ax.text(-0.02, 1.03, lbl, transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom')
        ax.set_xticks([]); ax.set_yticks([])

    sc_a = axes[0, 0].scatter(embedding[:, 0], embedding[:, 1], s=6, c=y_meanv,
                               cmap='RdPu_r', alpha=0.7, edgecolors='none')
    plt.colorbar(sc_a, ax=axes[0, 0], fraction=0.046, pad=0.04, label='Mean-prediction brightness (V)')
    axes[0, 0].set_title('Prediction (mean brightness)', fontsize=12)

    for cls in range(3):
        m = y_tissue == cls
        axes[0, 1].scatter(embedding[m, 0], embedding[m, 1], s=6, alpha=0.6,
                            c=TISSUE_COLORS[cls], label=TISSUE_NAMES[cls], edgecolors='none')
    axes[0, 1].legend(markerscale=4, fontsize=9, loc='upper right')
    axes[0, 1].set_title('Tissue class (from ground truth)', fontsize=12)

    vmax = np.percentile(y_uncert, 97)
    order = np.argsort(y_uncert)
    sc_c = axes[1, 0].scatter(embedding[order, 0], embedding[order, 1], s=6, alpha=0.7,
                               c=y_uncert[order], cmap='inferno', vmin=0, vmax=vmax, edgecolors='none')
    plt.colorbar(sc_c, ax=axes[1, 0], fraction=0.046, pad=0.04, label='MC-Dropout variance (clipped at p97)')
    axes[1, 0].set_title('Uncertainty', fontsize=12)

    for cls in range(2):
        m = y_confidence == cls
        axes[1, 1].scatter(embedding[m, 0], embedding[m, 1], s=6, alpha=0.6,
                            c=CONF_COLORS[cls], label=CONF_NAMES[cls], edgecolors='none')
    axes[1, 1].legend(markerscale=4, fontsize=9, loc='upper right')
    axes[1, 1].set_title('Confidence (median split of uncertainty)', fontsize=12)

    plt.suptitle(f'UMAP embedding of per-pixel MC-Dropout predictions - Sample {idx + 1}', fontsize=14, y=1.0)
    fig.text(0.5, -0.01, COLOUR_NOTE, ha='center', va='top', fontsize=9.5, color='dimgray', wrap=True)
    plt.tight_layout()
    plt.savefig(out_root / f'umap_uncertainty_sample_{idx + 1}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

N_RUNS_ALL = 10
POINTS_PER_IMAGE = 150
RNG = np.random.default_rng(42)

cache_path = out_root / 'umap_valset_features.npz'

if cache_path.exists():
    cached = np.load(cache_path)
    X = cached['X']
    y_tissue = cached['y_tissue']
    y_uncert = cached['y_uncert']
    y_meanv = cached['y_meanv']
    sample_id = cached['sample_id']
    n_images = int(cached['n_images'])
    print(f'loaded cached features from {cache_path}: {n_images} images, {X.shape[0]} points total, feature dim {X.shape[1]}')
else:
    X_all, y_tissue_all, y_uncert_all, y_meanv_all, sample_id_all = [], [], [], [], []

    for img_idx, (test_input, ground_truth) in enumerate(val_dataset):
        stack01 = mc_dropout_stack(generator, test_input, N_RUNS_ALL) * 0.5 + 0.5

        gt01 = (ground_truth[0].numpy() * 0.5 + 0.5).clip(0, 1)
        labels = classify_tissue(gt01)
        uncertainty = stack01.var(axis=0).mean(axis=-1)
        mean_v = mcolors.rgb_to_hsv(stack01.mean(axis=0))[..., 2]

        h, w = uncertainty.shape
        n_pix = h * w
        n_take = min(POINTS_PER_IMAGE, n_pix)
        pix_idx = RNG.choice(n_pix, size=n_take, replace=False)

        X_img = stack01.transpose(1, 2, 0, 3).reshape(n_pix, N_RUNS_ALL * 3)[pix_idx]
        X_all.append(X_img)
        y_tissue_all.append(labels.reshape(-1)[pix_idx])
        y_uncert_all.append(uncertainty.reshape(-1)[pix_idx])
        y_meanv_all.append(mean_v.reshape(-1)[pix_idx])
        sample_id_all.append(np.full(n_take, img_idx))

        if (img_idx + 1) % 20 == 0:
            print(f'processed {img_idx + 1} val images...')

    n_images = img_idx + 1
    X = np.concatenate(X_all, axis=0)
    y_tissue = np.concatenate(y_tissue_all)
    y_uncert = np.concatenate(y_uncert_all)
    y_meanv = np.concatenate(y_meanv_all)
    sample_id = np.concatenate(sample_id_all)

    print(f'val set: {n_images} images, {X.shape[0]} points total, feature dim {X.shape[1]}')

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X=X, y_tissue=y_tissue, y_uncert=y_uncert, y_meanv=y_meanv,
             sample_id=sample_id, n_images=n_images)
    print(f'cached extracted features to {cache_path}')

MAX_UMAP_POINTS = 400_000
if X.shape[0] > MAX_UMAP_POINTS:
    sub_idx = np.random.default_rng(43).choice(X.shape[0], size=MAX_UMAP_POINTS, replace=False)
    X = X[sub_idx]
    y_tissue = y_tissue[sub_idx]
    y_uncert = y_uncert[sub_idx]
    y_meanv = y_meanv[sub_idx]
    sample_id = sample_id[sub_idx]
    print(f'subsampled to {X.shape[0]} points for UMAP fit')

reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42, verbose=True)
embedding = reducer.fit_transform(X)
uncert_median = np.median(y_uncert)
y_confidence = (y_uncert >= uncert_median).astype(int)

fig, axes = plt.subplots(2, 3, figsize=(21, 14))
for ax, lbl in zip(axes.flat, ['a', 'b', 'c', 'd', 'e', 'f']):
    ax.text(-0.02, 1.03, lbl, transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom')
    ax.set_xticks([]); ax.set_yticks([])

sc_a = axes[0, 0].scatter(embedding[:, 0], embedding[:, 1], s=4, c=y_meanv,
                           cmap='RdPu_r', alpha=0.5, edgecolors='none')
plt.colorbar(sc_a, ax=axes[0, 0], fraction=0.046, pad=0.04, label='Mean-prediction brightness (V)')
axes[0, 0].set_title('Prediction (mean brightness)', fontsize=12)

for cls in range(3):
    m = y_tissue == cls
    axes[0, 1].scatter(embedding[m, 0], embedding[m, 1], s=4, alpha=0.4,
                        c=TISSUE_COLORS[cls], label=TISSUE_NAMES[cls], edgecolors='none')
axes[0, 1].legend(markerscale=4, fontsize=9, loc='upper right')
axes[0, 1].set_title('Tissue class (from ground truth)', fontsize=12)

sc_id = axes[0, 2].scatter(embedding[:, 0], embedding[:, 1], s=4, c=sample_id,
                            cmap='nipy_spectral', alpha=0.5, edgecolors='none')
plt.colorbar(sc_id, ax=axes[0, 2], fraction=0.046, pad=0.04, label='Source image index')
axes[0, 2].set_title(f'Source image (n={n_images} val images)', fontsize=12)

vmax = np.percentile(y_uncert, 97)
order = np.argsort(y_uncert)
sc_c = axes[1, 0].scatter(embedding[order, 0], embedding[order, 1], s=4, alpha=0.5,
                           c=y_uncert[order], cmap='inferno', vmin=0, vmax=vmax, edgecolors='none')
plt.colorbar(sc_c, ax=axes[1, 0], fraction=0.046, pad=0.04, label='MC-Dropout variance (clipped at p97)')
axes[1, 0].set_title('Uncertainty', fontsize=12)

for cls in range(2):
    m = y_confidence == cls
    axes[1, 1].scatter(embedding[m, 0], embedding[m, 1], s=4, alpha=0.4,
                        c=CONF_COLORS[cls], label=CONF_NAMES[cls], edgecolors='none')
axes[1, 1].legend(markerscale=4, fontsize=9, loc='upper right')
axes[1, 1].set_title('Confidence (median split of uncertainty)', fontsize=12)

axes[1, 2].axis('off')

plt.suptitle(f'UMAP embedding of per-pixel MC-Dropout predictions - whole val set ({n_images} images)', fontsize=14, y=1.0)
fig.text(0.5, -0.01, COLOUR_NOTE, ha='center', va='top', fontsize=9.5, color='dimgray', wrap=True)
plt.tight_layout()
plt.savefig(out_root / 'umap_uncertainty_valset_all.png', dpi=150, bbox_inches='tight')
plt.close(fig)

assert cache_path.exists(), 'part 2 must run at least once first (it caches features here)'

cached = np.load(cache_path)
X = cached['X']
y_tissue = cached['y_tissue']

if X.shape[0] > MAX_UMAP_POINTS:
    sub_idx = np.random.default_rng(43).choice(X.shape[0], size=MAX_UMAP_POINTS, replace=False)
    X = X[sub_idx]
    y_tissue = y_tissue[sub_idx]

reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42, verbose=True)
embedding = reducer.fit_transform(X)

summary = []
print(f'{"Class":<20}{"Count":>10}{"% of points":>14}{"Mean dist to centroid":>24}{"Convex hull area":>18}')
for cls in range(3):
    m = y_tissue == cls
    pts = embedding[m]
    count = int(m.sum())
    pct = 100 * count / len(y_tissue)
    centroid = pts.mean(axis=0)
    mean_dist = float(np.linalg.norm(pts - centroid, axis=1).mean())
    hull_area = float(ConvexHull(pts).volume)
    summary.append((count, pct, mean_dist, hull_area))
    print(f'{TISSUE_NAMES[cls]:<20}{count:>10}{pct:>13.1f}%{mean_dist:>24.3f}{hull_area:>18.1f}')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(TISSUE_NAMES, [s[1] for s in summary], color=TISSUE_COLORS)
axes[0].set_ylabel('% of points')
axes[0].set_title('Point count share per class')

axes[1].bar(TISSUE_NAMES, [s[2] for s in summary], color=TISSUE_COLORS)
axes[1].set_ylabel('Mean distance to class centroid (embedding units)')
axes[1].set_title('Spread: mean distance to centroid')

axes[2].bar(TISSUE_NAMES, [s[3] for s in summary], color=TISSUE_COLORS)
axes[2].set_ylabel('Convex hull area (embedding units^2)')
axes[2].set_title('Spread: convex hull area')

for ax in axes:
    ax.tick_params(axis='x', rotation=15)

plt.suptitle('Point count vs. spread per tissue class in the whole-val-set UMAP embedding', fontsize=13)
plt.tight_layout()
plt.savefig(out_root / 'umap_class_spread_vs_count.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('DONE', flush=True)
