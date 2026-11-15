"""
Build Fig. 2(a)-(e) from the data written by run_fig2bc.py, run_fig2e.py and
run_fig2d.py.  Run from the repository root:
    python examples/paper_reproduction/janus_bic/make_figures.py [tag]
(tag defaults to g5_m8, i.e. gmax = 5, 8 guided modes)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # before numpy: workers are forked

import sys
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Ellipse as MEll, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import janus_core as jc  # noqa: E402

DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
TAG = sys.argv[1] if len(sys.argv) > 1 else "g5_m8"
QCAP = 1e12   # display cap for Q (GME losses below ~1e-16 relative are noise)
C_L, C_R, C_BIC = "#d62728", "#1f77b4", "0.15"


# ---------------------------------------------------------------- (a)
def fig_a(dd=12.0):
    a = jc.A_NM
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax = axs[0]
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            ax.add_patch(Rectangle((i - .5, j - .5), 1, 1, fc="#c9d3e6",
                                   ec="0.6", lw=.5))
    for sgn, ls, lab in ((+1, "-", "top layer: minor d+Δd"),
                         (-1, "--", "bottom layer: minor d−Δd")):
        m = jc.D_REL + sgn * dd / a
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                ax.add_patch(MEll((i, j), jc.BETA * m, m, fc="white" if sgn > 0
                                  else "none", ec="k", ls=ls, lw=1,
                                  label=lab if (i, j) == (0, 0) else None))
    ax.add_patch(Rectangle((-.5, -.5), 1, 1, fc="none", ec="k", lw=1.2))
    ax.set_xlim(-1, 1), ax.set_ylim(-1, 1), ax.set_aspect("equal")
    ax.set_xlabel("x / a"), ax.set_ylabel("y / a")
    ax.set_title(f"top view, Δd = {dd:g} nm (to scale)", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    ax = axs[1]
    t = jc.T_LAYER_NM / a
    for k, (sgn, z0) in enumerate(((-1, 0), (+1, t))):
        m = jc.D_REL + sgn * dd / a
        w = jc.BETA * m
        ax.add_patch(Rectangle((-1, z0), 2, t, fc="#c9d3e6", ec="k"))
        for i in (-1, 0, 1):
            ax.add_patch(Rectangle((i - w / 2, z0), w, t, fc="white", ec="k",
                                   clip_on=True))
        ax.text(0, z0 + t / 2, f"{'top' if sgn > 0 else 'bottom'}: hole major "
                f"{w*a:.0f} nm, minor {m*a:.0f} nm\nSi$_3$N$_4$ 240 nm",
                fontsize=6.5, va="center", ha="center")
    ax.set_xlim(-1, 1), ax.set_ylim(-0.5, 2 * t + .5), ax.set_aspect("equal")
    ax.set_xlabel("x / a"), ax.set_ylabel("z / a")
    ax.set_title("cross-section y = 0 (air above/below)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2a_geometry.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- (b,c)
def fig_bc():
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.4))
    d0 = np.load(os.path.join(DATA, f"fig2bc_dd00.0_{TAG}.npz"))
    s = d0["s"]
    ax = axs[0]
    F, T = d0["freqs_all"], d0["tmfrac_all"]
    for j in range(F.shape[1]):
        tm = T[:, j] > 0.5
        ax.scatter(s[tm], F[tm, j], s=3, c="k", lw=0)
        ax.scatter(s[~tm], F[~tm, j], s=3, c="0.75", lw=0)
    ax.plot(s, d0["freq"], c="C3", lw=1.5, label="TM$_1$ (tracked)")
    ax.set_ylim(0.52, 0.60), ax.set_xlim(-0.1, 0.1)
    ax.set_xlabel(r"X$'$ $\leftarrow$ Γ $\rightarrow$ X   ($k_y a/2\pi$ | $k_x a/2\pi$)")
    ax.set_ylabel(r"frequency $a/\lambda$")
    ax.set_title("(b) Δd = 0: TM-like (black) / TE-like (grey)", fontsize=9)
    ax.legend(fontsize=7)
    ax = axs[1]
    for dd, c in ((0.0, "k"), (6.0, "C3"), (12.0, "C0")):
        fn = os.path.join(DATA, f"fig2bc_dd{dd:04.1f}_{TAG}.npz")
        d = np.load(fn)
        ax.semilogy(d["s"], np.minimum(d["Q"], QCAP), c=c, lw=1,
                    label=f"Δd = {dd:g} nm")
    ax.set_ylim(1e3, QCAP * 2), ax.set_xlim(-0.1, 0.1)
    ax.set_xlabel(r"X$'$ $\leftarrow$ Γ $\rightarrow$ X")
    ax.set_ylabel("Q (capped at 1e12)")
    ax.set_title("(c) radiative Q of TM$_1$ (GME, 1st order)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig2bc_{TAG}.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- (d)
def load_2d(tag=TAG):
    fs = sorted(glob.glob(os.path.join(DATA, f"fig2d_{tag}*.json")))
    br, loops = {}, {}
    for fn in fs:
        d = json.load(open(fn))
        for k, lst in d["branches"].items():
            br.setdefault(k, []).extend(lst)
        loops.update(d["loops"])
    for k in br:
        br[k] = sorted({e["dd"]: e for e in br[k]}.values(),
                       key=lambda e: e["dd"])
    return br, loops


def merge_estimate(br, loops, side="down"):
    """dd interval of the merge on one side: last dd with C points found on
    that side, first larger dd where the smallest-loop charge on that side is
    -1."""
    j = 0 if side == "up" else 1
    found = [e["dd"] for k in (side + "_L", side + "_R") for e in br.get(k, [])
             if e["found"]]
    last = max(found) if found else np.nan
    flipped = sorted(float(dd) for dd, q in loops.items()
                     if round(q[min(q, key=float)][j]) == -1)
    first = min([x for x in flipped if x > last], default=np.nan)
    return last, first


def fig_d():
    br, loops = load_2d()
    if not br:
        return
    last, first = merge_estimate(br, loops, "down")
    last_u, first_u = merge_estimate(br, loops, "up")
    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(121, projection="3d")
    for key, e_list in br.items():
        side, hand = key.split("_")
        es = [e for e in e_list if e["found"]]
        if not es:
            continue
        u = np.array([e["u"] for e in es])
        v = np.array([e["v"] for e in es])
        z = np.array([e["dd"] for e in es])
        c = C_L if hand == "L" else C_R
        ls = "-" if side == "down" else "--"
        for su in (1, -1):          # root + its C2 partner
            ax.plot(su * u, su * v, z, ls=ls, c=c, lw=1.2)
    zmax = max(e["dd"] for l in br.values() for e in l)
    zb = first if np.isfinite(first) else zmax
    ze = first_u if np.isfinite(first_u) else zmax
    ax.plot([0, 0], [0, 0], [0, zb], c=C_BIC, lw=2)
    if np.isfinite(first):
        ax.plot([0, 0], [0, 0], [zb, ze], c="g", lw=3)
    if ze < zmax:
        ax.plot([0, 0], [0, 0], [ze, zmax], c=C_BIC, lw=2, ls=":")
    ax.set_xlabel(r"$k_xa/2\pi$"), ax.set_ylabel(r"$k_ya/2\pi$")
    ax.set_zlabel("Δd (nm)")
    ax.set_title("(d) C points: red=LCP blue=RCP (lab S3), solid=down, dashed=up;"
                 "\ngreen=Janus BIC (q_up=+1,q_down=-1), dotted: q=-1 both sides",
                 fontsize=8)
    ax2 = fig.add_subplot(122)
    for key, e_list in br.items():
        side, hand = key.split("_")
        es = [e for e in e_list if e["found"]]
        if not es:
            continue
        ax2.plot([e["dd"] for e in es], [np.hypot(e["u"], e["v"]) for e in es],
                 ls="-" if side == "down" else "--",
                 c=C_L if hand == "L" else C_R, marker=".", ms=3,
                 lw=3 if hand == "L" else 1.2,
                 label=f"{side} {'LCP' if hand == 'L' else 'RCP'}")
    for (a_, b_, lab) in ((last, first, "down"), (last_u, first_u, "up")):
        if np.isfinite(a_) and np.isfinite(b_):
            ax2.axvspan(a_, b_, color="g" if lab == "down" else "k", alpha=.3,
                        label=f"merge interval ({lab}): {a_:.2f}-{b_:.2f} nm")
    ax2.set_xlabel("Δd (nm)"), ax2.set_ylabel(r"$|k_C| a/2\pi$")
    ax2.legend(fontsize=7), ax2.set_title("distance of C points from Γ",
                                          fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig2d_{TAG}.png"), dpi=200)
    plt.close(fig)
    return dict(down=(last, first), up=(last_u, first_u))


# ---------------------------------------------------------------- (e)
def ellipses(ax, U, V, E, step, scale):
    S0, S1, S2, S3 = jc.stokes(E)
    phi = np.linspace(0, 2 * np.pi, 25)
    segs, cols = [], []
    for j in range(0, U.shape[0], step):
        for i in range(0, U.shape[1], step):
            e = E[j, i] / np.sqrt(S0[j, i] + 1e-300)
            if S0[j, i] < 1e-20:
                continue
            # physical trajectory of the field, exp(-i w t)
            x = np.real(e[0] * np.exp(-1j * phi))
            y = np.real(e[1] * np.exp(-1j * phi))
            segs.append(np.c_[U[j, i] + scale * x, V[j, i] + scale * y])
            cols.append(C_L if S3[j, i] > 0 else C_R)
    ax.add_collection(LineCollection(segs, colors=cols, lw=.6))


def cluster(sing, rad):
    """merge singular plaquettes closer than rad: (u, v, q_sum, <S3/S0>)"""
    out, used = [], [False] * len(sing)
    for i, s in enumerate(sing):
        if used[i]:
            continue
        grp = [j for j in range(len(sing)) if not used[j] and
               np.hypot(sing[j][0] - s[0], sing[j][1] - s[1]) < rad]
        for j in grp:
            used[j] = True
        out.append((np.mean([sing[j][0] for j in grp]),
                    np.mean([sing[j][1] for j in grp]),
                    sum(sing[j][2] for j in grp),
                    np.mean([sing[j][3] for j in grp])))
    return out


def fig_e(N=41, step=2, dds=(0.0, 6.0, 12.0), name=None):
    br, _ = load_2d()
    info = []
    fig, axs = plt.subplots(len(dds), 2, figsize=(7, 3.4 * len(dds)),
                            sharex=True, sharey=True, squeeze=False)
    for row, dd in enumerate(dds):
        fn = os.path.join(DATA, f"fig2e_dd{dd:04.1f}_N{N}_{TAG}.npz")
        if not os.path.exists(fn):
            continue
        d = np.load(fn)
        U, V = d["U"], d["V"]
        h = U[0, 1] - U[0, 0]
        for col, side in enumerate(("up", "down")):
            ax = axs[row, col]
            E = d["E_proj_" + side].reshape(N, N, 2)
            ellipses(ax, U, V, E, step, 0.45 * h * step)
            # singularities: plaquette scan of the transverse field, clustered
            # (a singularity sitting exactly on a grid node/line splits over
            # 2-4 plaquettes)
            sing = cluster(jc.plaquette_singularities(
                U, V, d["E_t_" + side].reshape(N, N, 2)), 1.6 * h)
            # refined C points from run_fig2d (same dd), else from plaquettes
            pts = []
            for key in (f"{side}_L", f"{side}_R"):
                for e in br.get(key, []):
                    if abs(e["dd"] - dd) < 1e-6 and e["found"]:
                        for su in (1, -1):
                            pts.append((su * e["u"], su * e["v"], key[-1]))
            for (u, v, q, s3) in sing:
                if abs(abs(q) - 1) < 1e-6:          # integer charge: BIC
                    ax.plot(u, v, "o", mfc="0.6", mec="k", ms=8, zorder=5)
                    ax.text(u + .003, v + .004, f"q={q:+g}", fontsize=7)
                elif abs(abs(q) - .5) < 1e-6 and not pts:
                    pts.append((u, v, "L" if s3 > 0 else "R"))
            for (u, v, hnd) in pts:
                ax.plot(u, v, "o", ms=6, mec="k", mfc=C_L if hnd == "L" else C_R,
                        zorder=6)
            info.append((dd, side, [(round(u, 4), round(v, 4), q) for u, v, q, _ in sing],
                         [(round(u, 5), round(v, 5), hnd) for u, v, hnd in pts]))
            ax.set_xlim(-0.062, 0.062), ax.set_ylim(-0.062, 0.062)
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(("upward" if side == "up" else "downward") +
                             " radiation", fontsize=10)
            if col == 1:
                ax.text(1.02, .5, f"Δd = {dd:g} nm", transform=ax.transAxes,
                        rotation=90, va="center")
            if col == 0:
                ax.set_ylabel(r"$k_ya/2\pi$")
            if row == len(dds) - 1:
                ax.set_xlabel(r"$k_xa/2\pi$")
    fig.suptitle("(e) far-field polarization, TM$_1$  (red: S3>0 'LCP', blue: "
                 "S3<0 'RCP';\nlab frame viewed from +z, exp(-iωt); grey: BIC, "
                 "dots: C points)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name or f"fig2e_{TAG}.png"), dpi=220)
    plt.close(fig)
    return info


if __name__ == "__main__":
    fig_a()
    extra = lambda: fig_e(dds=(2.0,), name=f"fig2e_extra_dd2_{TAG}.png")
    for f in (fig_bc, fig_d, fig_e, extra):
        try:
            r = f()
            print(f.__name__, "ok", r if r is not None else "")
        except FileNotFoundError as err:
            print(f.__name__, "skipped (missing data):", err)
