"""
Diagnostic only (NOT the paper geometry): does the merge dd follow the
position of the off-Gamma BIC?  GME puts the dd=0 accidental BIC at
u ~ 0.022-0.031 instead of the paper's ~0.05. Here the layer thickness / beta
are detuned to move the GME BIC outward, and the Gamma charges q_up/q_down on a
small loop (r = 0.002) are swept in dd.  gmax = 4, 8 guided modes (fast).
Run from the repository root:
    python examples/paper_reproduction/janus_bic/run_diagnostic_geometry.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # before numpy: workers are forked

import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import janus_core as jc  # noqa: E402
from run_checks import bic_on_axis  # noqa: E402
from run_fig2d import to_json  # noqa: E402

GEOMS = ([dict()] + [dict(t_layer_nm=x) for x in (220.0, 244.0, 248.0, 252.0,
          256.0, 260.0)] + [dict(beta=1.30), dict(beta=1.40)])


def main():
    res = []
    for g in GEOMS:
        opts = dict(jc.DEFAULT, gmax=4.0, geom=g)
        f0, ub, Qb = bic_on_axis(0.0, opts)
        t = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        loop = np.c_[0.002 * np.cos(t), 0.002 * np.sin(t)]
        qs = []
        for dd in np.arange(0.0, 18.01, 1.0):
            d = jc.eval_points(dd, loop, 4, opts)
            qs.append((dd, jc.loop_charge(d["E_t_up"]),
                       jc.loop_charge(d["E_t_down"])))
        flip = lambda j: next((q[0] for q in qs if round(q[j]) == -1), None)  # noqa
        r = dict(geom=g, f_gamma=f0, u_bic=ub, Q_bic=Qb, first_dd_down_m1=flip(2),
                 first_dd_up_m1=flip(1), q=qs)
        res.append(r)
        print(g, f"f={f0:.4f} u_BIC={ub:.4f} (Q={Qb:.1e})  q_down=-1 from dd="
              f"{r['first_dd_down_m1']}  q_up=-1 from dd={r['first_dd_up_m1']}",
              flush=True)
    json.dump(res, open(os.path.join(HERE, "data", "diagnostic_geometry.json"),
                        "w"), indent=1, default=to_json)


if __name__ == "__main__":
    main()
