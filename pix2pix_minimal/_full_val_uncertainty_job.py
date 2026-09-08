import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import pathlib, time, csv
from sklearn.metrics import roc_curve, roc_auc_score

exec(open('_gen_stub7.py', encoding='utf-8').read())

PATH = pathlib.Path('./datasets/polyps_v7')
out_root = pathlib.Path('./uncertainty')
out_root.mkdir(parents=True, exist_ok=True)

generator = Generator()
ckpt = tf.train.Checkpoint(generator=generator)
ckpt.restore(tf.train.latest_checkpoint('./training_checkpoints')).expect_partial()
print('checkpoint restored', flush=True)

N_RUNS = 10
STRIDE = 8

image_fns = sorted((PATH / 'val').glob('*.tif'))
n_total = len(image_fns)
print(f'{n_total} val images found, N_RUNS={N_RUNS}, STRIDE={STRIDE}', flush=True)

all_uncert, all_error = [], []
per_image_rows = []

t_start = time.time()
for i, fn in enumerate(image_fns):
    raw = tf.io.read_file(str(fn))
    image = tfio.experimental.image.decode_tiff(raw)
    half = tf.shape(image)[1] // 2
    gt  = tf.cast(image[:, :half, :3], tf.float32); gt = (gt / 127.5) - 1.0
    inp = tf.cast(image[:, half:, :INPUT_CHANNELS], tf.float32); inp = (inp / 127.5) - 1.0
    inp = inp[tf.newaxis]

    inp_rep = tf.repeat(inp, N_RUNS, axis=0)
    stack = generator(inp_rep, training=True).numpy()
    stack01 = stack * 0.5 + 0.5
    mean_pred01 = stack01.mean(axis=0)
    gt01 = (gt.numpy() * 0.5 + 0.5).clip(0, 1)

    uncertainty = stack01.var(axis=0).mean(axis=-1)
    error = np.abs(mean_pred01 - gt01).mean(axis=-1)

    all_uncert.append(uncertainty[::STRIDE, ::STRIDE].reshape(-1))
    all_error.append(error[::STRIDE, ::STRIDE].reshape(-1))
    per_image_rows.append({
        'file': fn.name,
        'mean_uncertainty': round(float(uncertainty.mean()), 8),
        'mean_error': round(float(error.mean()), 6),
    })

    if (i + 1) % 200 == 0 or (i + 1) == n_total:
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        remaining = (n_total - (i + 1)) / rate if rate > 0 else float('nan')
        print(f'[{i + 1}/{n_total}]  elapsed={elapsed/60:.1f}min  '
              f'est. remaining={remaining/60:.1f}min  '
              f'mean_uncert={uncertainty.mean():.6f}  mean_error={error.mean():.4f}', flush=True)

with open(out_root / 'full_val_uncertainty_per_image.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=['file', 'mean_uncertainty', 'mean_error'])
    wr.writeheader()
    wr.writerows(per_image_rows)

y_uncert = np.concatenate(all_uncert)
y_error  = np.concatenate(all_error)
np.savez(out_root / 'full_val_uncertainty_pooled.npz', y_uncert=y_uncert, y_error=y_error)
print(f'\npooled pixel count: {y_uncert.shape[0]}', flush=True)

error_q75 = np.percentile(y_error, 75)
y_incorrect = (y_error >= error_q75).astype(int)

auc = roc_auc_score(y_incorrect, y_uncert)
fpr, tpr, thresholds = roc_curve(y_incorrect, y_uncert)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
theta = thresholds[best_idx]
median_theta = np.median(y_uncert)

print(f'AUC (uncertainty predicting worst-error-quartile pixels): {auc:.4f}', flush=True)
print(f"Youden's J theta: {theta:.6f}  (TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f})", flush=True)
print(f'Naive median-split threshold: {median_theta:.6f}', flush=True)

fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

bins = np.linspace(0, np.percentile(y_uncert, 99), 60)
axes[0].hist(y_uncert[y_incorrect == 0], bins=bins, density=True, alpha=0.5, color='seagreen', label='Correct (error < p75)')
axes[0].hist(y_uncert[y_incorrect == 1], bins=bins, density=True, alpha=0.5, color='darkorange', label='Incorrect (error >= p75)')
axes[0].axvline(theta, color='black', linestyle='--', linewidth=1.5, label=f"Youden's J theta={theta:.5f}")
axes[0].axvline(median_theta, color='gray', linestyle=':', linewidth=1.5, label=f'median split={median_theta:.5f}')
axes[0].set_xlabel('MC-Dropout uncertainty (variance)')
axes[0].set_ylabel('Density')
axes[0].set_title('Uncertainty distribution: correct vs incorrect pixels', fontsize=12)
axes[0].legend(fontsize=8)

hb = axes[1].hexbin(y_uncert, y_error, gridsize=60, bins='log', cmap='viridis',
                     extent=(0, np.percentile(y_uncert, 99), 0, np.percentile(y_error, 99)))
plt.colorbar(hb, ax=axes[1], fraction=0.046, pad=0.04, label='log10(pixel count)')
axes[1].axvline(theta, color='white', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('MC-Dropout uncertainty (variance)')
axes[1].set_ylabel('Pixel error |mean pred - GT|')
axes[1].set_title('Uncertainty vs. error (per pixel)', fontsize=12)

axes[2].plot(fpr, tpr, color='steelblue', linewidth=2, label=f'ROC (AUC={auc:.3f})')
axes[2].plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1)
axes[2].scatter([fpr[best_idx]], [tpr[best_idx]], color='black', zorder=5, label="Youden's J optimum")
axes[2].set_xlabel('False positive rate')
axes[2].set_ylabel('True positive rate')
axes[2].set_title('ROC: uncertainty predicting worst-error-quartile pixels', fontsize=12)
axes[2].legend(fontsize=9, loc='lower right')

plt.suptitle(f'Does MC-Dropout uncertainty track prediction error? (FULL val set, n={n_total} images, N_RUNS={N_RUNS})', fontsize=13)
plt.tight_layout()
plt.savefig(out_root / 'uncertainty_vs_error_validation_FULLVAL.png', dpi=150, bbox_inches='tight')
print('DONE', flush=True)
