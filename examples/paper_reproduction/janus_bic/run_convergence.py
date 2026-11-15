"""
Convergence of the key numbers w.r.t. the GME truncation:
  * Gamma frequency of TM1 and off-Gamma BIC position (dd = 0),
  * downward and upward merge dd (C points absorbed by the Gamma BIC),
    found by bisection on the charge of a tiny loop (r = 5e-4) around Gamma:
    q_down: +1 -> -1 at the downward merge, q_up: +1 -> -1 at the upward one.
    (bias of the loop radius: r / (d|k_C|/d dd) ~ 0.03 nm)

Run from the repository root:
    python examples/paper_reproduction/janus_bic/run_convergence.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # before numpy: workers are forked

import sys
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import janus_core as jc  # noqa: E402
from run_checks import bic_on_axis  # noqa: E402
from run_fig2d import to_json  # noqa: E402

OUT = os.path.join(HERE, "data")
NPROC = int(os.environ.get("NPROC", "4"))
R_LOOP = 5e-4

VARIANTS = [dict(gmax=3.0, n_gmodes=8), dict(gmax=4.0, n_gmodes=8),
            dict(gmax=5.0, n_gmodes=8), dict(gmax=6.0, n_gmodes=8),
            dict(gmax=7.0, n_gmodes=8),
            dict(gmax=5.0, n_gmodes=6), dict(gmax=5.0, n_gmodes=10),
            dict(gmax=5.0, n_gmodes=12),
            dict(gmax=5.0, n_gmodes=8, eps_eff="background")]


def small_loop_q(dd, opts, n=24):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    d = jc.eval_points(dd, np.c_[R_LOOP * np.cos(t), R_LOOP * np.sin(t)],
                       NPROC, opts)
    return jc.loop_charge(d["E_t_up"]), jc.loop_charge(d["E_t_down"])


def bisect(opts, side, lo, hi, tol=0.02):
    j = 0 if side == "up" else 1
    qlo, qhi = small_loop_q(lo, opts)[j], small_loop_q(hi, opts)[j]
    if not (round(qlo) == 1 and round(qhi) == -1):
        return (np.nan, np.nan, qlo, qhi)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        q = small_loop_q(mid, opts)[j]
        if round(q) == 1:
            lo = mid
        else:
            hi = mid
    return (lo, hi, 1, -1)


def main():
    fn = os.path.join(OUT, "convergence.json")
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    for v in VARIANTS:
        opts = dict(jc.DEFAULT, **v)
        key = "gmax%g_m%d_%s" % (opts["gmax"], opts["n_gmodes"],
                                 opts.get("eps_eff", "average"))
        if key in res:
            print(key, res[key])
            continue
        t0 = time.time()
        f0, ub, Qb = bic_on_axis(0.0, opts)
        dn = bisect(opts, "down", 1.0, 12.0)
        up = bisect(opts, "up", dn[1] if np.isfinite(dn[1]) else 1.0, 18.0)
        res[key] = dict(f_gamma=f0, u_bic=ub, Q_bic=Qb, merge_down=dn[:2],
                        merge_up=up[:2], q_ends_down=dn[2:], q_ends_up=up[2:],
                        n_pw=int(jc.Solver(0, opts["gmax"]).gvec.shape[1]),
                        seconds=time.time() - t0)
        print(key, res[key], flush=True)
        json.dump(res, open(fn, "w"), indent=1, default=to_json)


if __name__ == "__main__":
    main()
