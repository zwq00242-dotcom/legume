"""
Pre-flight checks before the 2D maps (Fig. 2e) and the C-point tracking.

  1. dd = 0: up/down far-field polarization must coincide (sigma_z symmetry),
     Gamma BIC charge +1, off-Gamma BIC charge -1 on both sides.
  2. complex (Ex, Ey) and (c_te, c_tm) at (u,v) = (0.01,0) and (0,0.01)
     -> axis / s,p bookkeeping.
  3. dd = 12 nm: charge on small loops around Gamma, several radii.
  (+) Ellipse (analytic FT) vs inscribed equal-area polygon, Gamma frequency
      and off-Gamma BIC position, as a check of the shape implementation.

Run from the repository root:
    python examples/paper_reproduction/janus_bic/run_checks.py
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

OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)
NPROC = int(os.environ.get("NPROC", "4"))


def ring(r, n=36, c=(0.0, 0.0)):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t)]


def axis_angle(E):
    _, S1, S2, _ = jc.stokes(E)
    return 0.5 * np.degrees(np.angle(S1 + 1j * S2))


def fmt(z):
    return f"{z.real:+.4e}{z.imag:+.4e}j"


def bic_on_axis(dd, opts=None):
    from scipy.optimize import minimize_scalar
    tr = jc.make_tracker(dd, **dict(jc.DEFAULT, **(opts or {})))
    us = np.linspace(0.0, 0.1, 21)
    Q = np.array([tr.eval(u, 0)["Q"] for u in us])
    i = int(np.argmax(Q[2:]) + 2)
    f = lambda u: -np.log(tr.eval(u, 0, store=False)["Q"])  # noqa: E731
    try:
        r = minimize_scalar(f, bracket=(us[i - 1], us[i], us[min(i + 1, 20)]),
                            tol=1e-7)
    except ValueError:   # peak not bracketed on the grid
        r = minimize_scalar(f, bounds=(us[i - 1], us[min(i + 1, 20)]),
                            method="bounded", options=dict(xatol=1e-7))
    return tr.gamma["f"], float(r.x), float(np.exp(-r.fun))


def main():
    rep = {}
    lines = []
    P = lines.append

    # ---------------- check 1 -----------------
    f0, ub, Qb = bic_on_axis(0.0)
    P(f"[1] dd=0: Gamma TM1 f={f0:.5f}; off-Gamma BIC on +kx axis at u={ub:.5f} "
      f"(Q={Qb:.2e})")
    pts = np.vstack([ring(0.01, 24), ring(0.02, 24), ring(0.004, 24, (ub, 0)),
                     ring(0.004, 24, (-ub, 0)),
                     np.c_[np.linspace(-0.06, 0.06, 13), 0.03 * np.ones(13)]])
    d = jc.eval_points(0.0, pts, NPROC)
    Eu, Ed = d["E_t_up"], d["E_t_down"]
    fid = np.abs(np.sum(Eu.conj() * Ed, 1)) / (np.linalg.norm(Eu, axis=1) *
                                                 np.linalg.norm(Ed, axis=1))
    rel = np.abs(np.sum(Eu.conj() * Ed, 1)) / np.linalg.norm(Eu, axis=1)**2
    dang = axis_angle(Eu) - axis_angle(Ed)
    dang = (dang + 90) % 180 - 90
    S3u = jc.stokes(Eu)[3] / jc.stokes(Eu)[0]
    S3d = jc.stokes(Ed)[3] / jc.stokes(Ed)[0]
    P(f"    up/down state fidelity |<Eu|Ed>|/(|Eu||Ed|): min={fid.min():.10f}")
    P(f"    |Ed|/|Eu| range: {rel.min():.6f}..{rel.max():.6f}")
    P(f"    max |major-axis angle difference| = {np.abs(dang).max():.2e} deg")
    P(f"    max |S3_up - S3_down| = {np.abs(S3u - S3d).max():.2e}; "
      f"max|S3| = {np.abs(S3u).max():.2e}")
    ph = np.angle(np.sum(Eu.conj() * Ed, 1))
    P(f"    relative phase Ed/Eu (deg): {np.unique(np.round(np.degrees(ph)))}")
    q = {}
    for name, sl in (("Gamma r=0.01", slice(0, 24)), ("Gamma r=0.02", slice(24, 48)),
                     ("+u BIC", slice(48, 72)), ("-u BIC", slice(72, 96))):
        q[name] = (jc.loop_charge(Eu[sl]), jc.loop_charge(Ed[sl]))
        P(f"    loop {name:13s}: q_up={q[name][0]:+.3f}  q_down={q[name][1]:+.3f}")
    rep["check1"] = dict(f_gamma=f0, u_bic=ub, Q_bic=Qb, fid_min=float(fid.min()),
                         max_dangle=float(np.abs(dang).max()), charges=q)

    # ---------------- check 2 -----------------
    P("[2] complex far field at (0.01,0) and (0,0.01)  [exp(-i w t), physical k]")
    rep["check2"] = {}
    for dd in (0.0, 12.0):
        d2 = jc.eval_points(dd, np.array([[0.01, 0.0], [0.0, 0.01]]), 1)
        for i, lab in enumerate(("(0.01,0)", "(0,0.01)")):
            P(f"    dd={dd:4.1f} {lab}: f={d2['freq'][i]:.5f} Q={d2['Q'][i]:.3e}")
            for side in ("up", "down"):
                E = d2["E_t_" + side][i]
                P(f"       {side:4s} c_te={fmt(d2['c_te_'+side][i])} "
                  f"c_tm={fmt(d2['c_tm_'+side][i])}  Ex={fmt(E[0])} Ey={fmt(E[1])}"
                  f"  |Ey/Ex|={abs(E[1])/abs(E[0]):.3e}")
                rep["check2"][f"{dd}_{lab}_{side}"] = [str(E[0]), str(E[1])]
    P("    bookkeeping: in-plane p-hat || k, s-hat _|_ k "
      "(E || x at k || x is p, E || y at k || y is p).  On the mirror lines (sigma_x, sigma_y) the "
      "radiation must be pure s or pure p (linear).")

    # ---------------- check 3 -----------------
    P("[3] dd=12 nm: charges on loops around Gamma")
    radii = [0.002, 0.004, 0.006, 0.01, 0.015, 0.02, 0.03]
    pts = np.vstack([ring(r, 48) for r in radii])
    d3 = jc.eval_points(12.0, pts, NPROC)
    rep["check3"] = {}
    for i, r in enumerate(radii):
        sl = slice(48 * i, 48 * (i + 1))
        qu = jc.loop_charge(d3["E_t_up"][sl])
        qd = jc.loop_charge(d3["E_t_down"][sl])
        qu_p = jc.loop_charge(d3["E_proj_up"][sl])
        qd_p = jc.loop_charge(d3["E_proj_down"][sl])
        P(f"    r={r:.3f}: q_up={qu:+.3f} q_down={qd:+.3f}   (projected field: "
          f"{qu_p:+.3f}, {qd_p:+.3f})   min overlap={d3['overlap'][sl].min():.3f}")
        rep["check3"][r] = (qu, qd)

    # ---------------- shape check -----------------
    P("[+] Ellipse vs equal-area polygon (gmax=4, 8 guided modes)")
    for shape, npoly in (("ellipse", 0), ("poly", 32), ("poly", 64), ("poly", 128)):
        f, u, Qm = bic_on_axis(0.0, dict(gmax=4.0, shape=shape, npoly=npoly))
        P(f"    {shape:7s} N={npoly:3d}: f_Gamma={f:.6f}  u_BIC={u:.5f}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "checks.txt"), "w") as fh:
        fh.write(txt + "\n")
    with open(os.path.join(OUT, "checks.json"), "w") as fh:
        json.dump(rep, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
