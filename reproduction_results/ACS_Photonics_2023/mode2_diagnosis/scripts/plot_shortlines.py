"""Render existing Jones/Stokes results with separated, shorter axis segments."""
from _paths import BASE, REPO, CACHE
import os
from pathlib import Path

os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = str(CACHE / 'matplotlib')

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = BASE
LENGTH = 0.0032  # 60% of the 0.005333 grid spacing; formerly 0.0068.


def load(name):
    with (ROOT / name).open() as f:
        rows = list(csv.DictReader(f))
    q = np.array([[float(r['qx']), float(r['qy'])] for r in rows])
    psi = np.array([float(r['psi']) for r in rows])
    valid = np.array([r['polarization_valid'] == 'True' for r in rows])
    return q, psi, valid


def draw(ax, data, title=''):
    q, psi, valid = data
    v = LENGTH / 2 * np.column_stack((np.cos(psi[valid]), np.sin(psi[valid])))
    ax.add_collection(LineCollection(np.stack((q[valid] - v, q[valid] + v), axis=1),
                                     colors='#173b64', linewidths=.85))
    ax.plot(0, 0, 'o', color='gray', ms=4, zorder=4)
    ax.set(xlim=(-.08, .08), ylim=(-.08, .08), aspect='equal', title=title,
           xlabel=r'$q_x=k_xa/(2\pi)$', ylabel=r'$q_y=k_ya/(2\pi)$')


def main():
    mode2 = load('data/fullgrid_mode2.csv')
    current = load('data/fullgrid_current.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.linewidth': .8, 'savefig.facecolor': 'white'})
    fig, ax = plt.subplots(figsize=(5.3, 5.0))
    draw(ax, mode2)
    ax.text(.025, .965, '(a)', transform=ax.transAxes, va='top', fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT / 'figures/mode2_polarization_shortlines.png', dpi=300)
    plt.close(fig)

    q, psi2, valid2 = mode2
    _, psi1, valid1 = current
    delta = (psi2 - psi1 + np.pi / 2) % np.pi - np.pi / 2
    n = int(round(np.sqrt(len(q))))
    step = .16 / (n - 1)
    fig, axes = plt.subplots(1, 3, figsize=(15.9, 5.0))
    draw(axes[0], current, 'Current mode (raw)')
    draw(axes[1], mode2, 'Candidate mode 2 (raw)')
    im = axes[2].imshow(np.where(valid1 & valid2, np.degrees(delta), np.nan).reshape(n, n),
                        origin='lower', extent=[-.08-step/2, .08+step/2]*2,
                        vmin=-90, vmax=90, cmap='coolwarm')
    axes[2].set(xlim=(-.08, .08), ylim=(-.08, .08), aspect='equal',
                xlabel=r'$q_x=k_xa/(2\pi)$', ylabel=r'$q_y=k_ya/(2\pi)$',
                title=r'Wrapped $\Delta\psi$ (deg)')
    axes[2].plot(0, 0, 'o', color='gray', ms=4)
    fig.colorbar(im, ax=axes[2], shrink=.8)
    fig.tight_layout()
    fig.savefig(ROOT / 'figures/current_vs_mode2_shortlines.png', dpi=250)
    plt.close(fig)
    print('Rendered both short-line figures from existing CSVs; segment length =', LENGTH)


if __name__ == '__main__':
    main()
