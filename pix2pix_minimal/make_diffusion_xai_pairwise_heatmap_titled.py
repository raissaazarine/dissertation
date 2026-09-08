"""Regenerates the diffusion model's xai_pairwise_difference_heatmap.png with
'Diffusion Model' added to the title, matching the same fix already applied
to xai_method_comparison.png. Same data, same 4-panel per-metric layout,
same significance stars -- title text is the only change.
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = pathlib.Path('E:/Dissertation/xai_evaluation_diff/perturbation_metrics_per_image.csv')
OUT_PNG = pathlib.Path('E:/Dissertation/xai_evaluation_diff/xai_pairwise_difference_heatmap.png')

df = pd.read_csv(CSV_PATH)
metrics = ['deletion_auc', 'insertion_auc', 'sensitivity', 'entropy']
titles = {
    'deletion_auc': 'Deletion AUC (lower = better)',
    'insertion_auc': 'Insertion AUC (higher = better)',
    'sensitivity': 'Sensitivity (higher = better)',
    'entropy': 'Entropy (context-dependent)',
}
methods = [m for m in ['Saliency', 'Grad x Input', 'SmoothGrad'] if m in set(df['method'])]
n_m = len(methods)

wide = {m: df.pivot(index='image_idx', columns='method', values=m)[methods].dropna()
        for m in metrics}

try:
    from scipy import stats as _stats
    have_scipy = True
except ImportError:
    have_scipy = False

fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.8))
for ax, metric in zip(axes, metrics):
    W = wide[metric]
    diff = np.zeros((n_m, n_m))
    pval = np.ones((n_m, n_m))
    for i in range(n_m):
        for j in range(n_m):
            a, b = W.iloc[:, i].to_numpy(), W.iloc[:, j].to_numpy()
            diff[i, j] = np.mean(a - b)
            if have_scipy and i != j:
                pval[i, j] = _stats.ttest_rel(a, b).pvalue

    vmax = np.abs(diff).max() or 1.0
    im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_m)); ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.set_yticks(range(n_m)); ax.set_yticklabels(methods)
    ax.set_title(titles[metric], fontsize=10)
    for i in range(n_m):
        for j in range(n_m):
            star = ''
            if have_scipy and i != j:
                p = pval[i, j]
                star = '\n***' if p < 1e-3 else '\n**' if p < 1e-2 else '\n*' if p < 5e-2 else ''
            ax.text(j, i, f'{diff[i, j]:+.4f}{star}', ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

n_img = len(next(iter(wide.values())))
subtitle = f'paired over {n_img} val images  (row - column)'
if have_scipy:
    subtitle += '   * p<.05   ** p<.01   *** p<.001 (paired t-test)'
fig.suptitle('Diffusion Model -- Pairwise mean difference between XAI attribution methods\n' + subtitle,
             fontsize=12)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
print(f'Saved -> {OUT_PNG.resolve()}')
