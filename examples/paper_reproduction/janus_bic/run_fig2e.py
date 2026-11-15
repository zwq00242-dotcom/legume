"""
Fig. 2(e): far-field polarization maps of the TM1 band, upward and downward,
for dd = 0, 6, 12 nm on (u, v) in [-0.06, 0.06]^2.
Also locates all polarization singularities on the grid (plaquette winding).

Run from the repository root:
    python examples/paper_reproduction/janus_bic/run_fig2e.py [N] [gmax n_gmodes]
N = grid points per axis (default 41 -> step 0.003, Gamma on the grid).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # before numpy: workers are forked

import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import janus_core as jc  # noqa: E402

OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)
NPROC = int(os.environ.get("NPROC", "4"))


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 41
    opts = {}
    if len(sys.argv) > 3:
        opts = dict(gmax=float(sys.argv[2]), n_gmodes=int(sys.argv[3]))
    tag = "g%g_m%d" % (opts.get("gmax", jc.DEFAULT["gmax"]),
                       opts.get("n_gmodes", jc.DEFAULT["n_gmodes"]))
    ax = np.linspace(-0.06, 0.06, N)
    U, V = np.meshgrid(ax, ax)
    uv = np.c_[U.ravel(), V.ravel()]
    dds = [float(x) for x in os.environ.get("DDS", "0,6,12").split(",")]
    for dd in dds:
        d = jc.eval_points(dd, uv, NPROC, opts)
        d["U"], d["V"] = U, V
        fn = os.path.join(OUT, f"fig2e_dd{dd:04.1f}_N{N}_{tag}.npz")
        np.savez(fn, **d)
        print(f"dd={dd}: saved {fn}; min overlap={d['overlap'].min():.3f}, "
              f"min margin={np.min(d['overlap'] - d['overlap2']):.3f}, "
              f"f range {d['freq'].min():.4f}..{d['freq'].max():.4f}")
        for side in ("up", "down"):
            E = d["E_t_" + side].reshape(N, N, 2)
            sing = jc.plaquette_singularities(U, V, E)
            print(f"  {side}: {len(sing)} singular plaquettes:")
            for s in sing:
                print(f"     (u,v)=({s[0]:+.4f},{s[1]:+.4f}) q={s[2]:+.1f} "
                      f"<S3/S0>={s[3]:+.3f} <S0>={s[4]:.2e}")


if __name__ == "__main__":
    main()
