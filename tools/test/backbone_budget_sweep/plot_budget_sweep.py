#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot the measured full Backbone budget sweep; no interpolation."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODES = ('roi_batch', 'adjacency_merge')

def write_budget_plots(out_dir, summary_rows, budgets):
    out_dir = Path(out_dir)
    for mode, title, filename in (
        ('roi_batch', 'ROI-Batch', 'roi_batch_budget_latency.png'),
        ('adjacency_merge', 'Adjacency-Merge', 'adjacency_merge_budget_latency.png'),
    ):
        rows = sorted((r for r in summary_rows if r['mode'] == mode),
                      key=lambda r: float(r['budget_pct']))
        if [float(r['budget_pct']) for r in rows] != [float(x) for x in budgets]:
            raise RuntimeError('Missing or duplicated measured budget rows for {}'.format(mode))
        x = np.asarray([float(r['budget_pct']) for r in rows], dtype=float)
        y = np.asarray([float(r['cnn_stream_ms_mean']) for r in rows], dtype=float)
        err = np.asarray([float(r['cnn_stream_ms_std']) for r in rows], dtype=float)
        fig, ax = plt.subplots(figsize=(10.5, 6.0))
        ax.errorbar(x, y, yerr=err, marker='o', capsize=3, linewidth=1.6,
                    markersize=5, label='Measured mean ± 1 SD')
        if len(x) and x[0] == 0.0:
            ax.scatter([0.0], [y[0]], s=90, facecolors='none', edgecolors='black',
                       linewidths=1.5, label='0% no-Backbone reference')
            ax.annotate('0%: ResNet/FPN bypass', (x[0], y[0]),
                        xytext=(8, 10), textcoords='offset points', fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 101, 5))
        ax.set_xlabel('Backbone packed-input budget (%)')
        ax.set_ylabel('Mean CNN Stream interval (ms; six cameras)')
        ax.set_title('{}: measured mean latency vs Backbone budget'.format(title))
        ax.grid(True, alpha=0.28)
        ax.legend()
        ax.text(0.01, 0.99,
                'All nonzero points are measured with real ResNet+FPN in UniAD E2E.\n'
                'Error bars show sample standard deviation; 0% skips ResNet/FPN.',
                transform=ax.transAxes, va='top', ha='left', fontsize=8,
                bbox={'boxstyle':'round,pad=0.35','facecolor':'white','alpha':0.85})
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=220, bbox_inches='tight')
        plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',required=True)
    p.add_argument('--budgets',default=','.join(str(x) for x in range(0,101,5)))
    a=p.parse_args()
    budgets=sorted(float(x.strip()) for x in a.budgets.split(',') if x.strip())
    path=Path(a.input_dir)/'budget_sweep_summary.csv'
    with path.open('r',newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    write_budget_plots(a.input_dir,rows,budgets)
    print('Generated two plots in',a.input_dir)

if __name__=='__main__':
    main()
