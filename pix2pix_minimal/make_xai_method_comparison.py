"""Pix2Pix counterpart to VS_Diff-main/full_xai_evaluation.py's
summarize_xai_evaluation() plot -- same 4-panel bar layout, same colors, no
error bars, so it can sit side-by-side with the diffusion model's
xai_method_comparison.png for direct visual comparison. (summary_metrics.png
already covers the same data with added 95% CI error bars; this is the
plain, style-matched version.)
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = pathlib.Path('./xai_evaluation/full_val/perturbation_metrics_per_image.csv')
OUT_PNG = pathlib.Path('./xai_evaluation/full_val/xai_method_comparison.png')

df = pd.read_csv(CSV_PATH)
metrics = ['deletion_auc', 'insertion_auc', 'sensitivity', 'entropy']
titles = ['Deletion AUC (lower = better)', 'Insertion AUC (higher = better)',
          'Sensitivity (higher = better)', 'Entropy (context-dependent)']
methods = [m for m in ['Saliency', 'Grad x Input', 'SmoothGrad'] if m in set(df['method'])]

means = {m: df[df['method'] == m][metrics].mean() for m in methods}

print(f'{"Method":<16}{"n":>6}{"Deletion AUC":>16}{"Insertion AUC":>16}'
      f'{"Sensitivity":>14}{"Entropy":>12}')
for m in methods:
    n = (df['method'] == m).sum()
    row = means[m]
    print(f'{m:<16}{n:>6}{row["deletion_auc"]:>16.4f}{row["insertion_auc"]:>16.4f}'
          f'{row["sensitivity"]:>14.4f}{row["entropy"]:>12.4f}')

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for ax, metric, title in zip(axes, metrics, titles):
    ax.bar(methods, [means[m][metric] for m in methods], color=['#4c72b0', '#dd8452', '#55a868'])
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis='x', rotation=15)

plt.suptitle('Pix2Pix -- XAI attribution method comparison (perturbation-based faithfulness)', fontsize=13)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
print(f'Saved -> {OUT_PNG.resolve()}')
