"""
Fig. 2(b,c): TM-like bands (dd = 0) and radiative Q of the tracked TM1 band
for dd = 0, 6, 12 nm along  X' <- Gamma -> X.
Horizontal coordinate s in [-0.1, 0.1]:  s < 0 -> (u, v) = (0, |s|)  (Gamma-X'),
                                          s > 0 -> (u, v) = (s, 0)    (Gamma-X).
Run from the repository root:
    python examples/paper_reproduction/janus_bic/run_fig2bc.py [gmax n_gmodes]
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


def path_points(n=41, smax=0.1, extra=()):
    s = np.unique(np.r_[np.linspace(-smax, smax, 2 * n - 1), list(extra)])
    uv = np.c_[np.where(s > 0, s, 0.0), np.where(s < 0, -s, 0.0)]
    return s, uv


def main():
    opts = {}
    if len(sys.argv) > 2:
        opts = dict(gmax=float(sys.argv[1]), n_gmodes=int(sys.argv[2]))
    tag = "g%g_m%d" % (opts.get("gmax", jc.DEFAULT["gmax"]),
                       opts.get("n_gmodes", jc.DEFAULT["n_gmodes"]))
    for dd in (0.0, 6.0, 12.0):
        s, uv = path_points()
        d = jc.eval_points(dd, uv, NPROC, opts)
        # refine around the Q maximum on each branch (off-Gamma peak)
        extra = []
        for sgn in (-1, 1):
            m = (np.sign(s) == sgn) & (np.abs(s) > 0.006)
            j = np.where(m)[0][np.argmax(d["Q"][m])]
            extra += list(np.linspace(s[j] - 0.004, s[j] + 0.004, 33))
        extra += list(np.linspace(-0.006, 0.006, 25))
        s, uv = path_points(extra=extra)
        d = jc.eval_points(dd, uv, NPROC, opts)
        d["s"] = s
        fn = os.path.join(OUT, f"fig2bc_dd{dd:04.1f}_{tag}.npz")
        np.savez(fn, **d)
        i = np.argsort(-d["Q"] * (np.abs(s) > 0.004))[:3]
        print(f"dd={dd}: f_Gamma={d['gamma_f']:.5f}; largest-Q points (|s|>0.004):"
              f" {[(round(s[k], 5), '%.2e' % d['Q'][k]) for k in i]}; "
              f"min overlap={d['overlap'].min():.3f} -> {fn}")


if __name__ == "__main__":
    main()
