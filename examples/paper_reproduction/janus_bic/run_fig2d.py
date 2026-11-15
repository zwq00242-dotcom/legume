"""
Fig. 2(d): trajectories of the C points (upward / downward, LCP / RCP) versus
dd, and the charge of the Gamma BIC on each side.

Method
------
* C point of handedness h on side s: zero of the complex ratio
      r_h(u, v) = E_h / E_other ,   E_h = lab-frame circular component of the
  transverse field E_t (E_minus=0 -> LCP, S3=+1; E_plus=0 -> RCP, S3=-1).
  The ratio is independent of the arbitrary eigenvector phase and finite
  wherever the mode radiates, so it is a smooth function of (u, v).
* Damped Newton iteration (forward-difference Jacobian), continued in dd from
  the previous root.  Each (side, handedness) branch is one process.
* At every dd: residual |r_h| at the C2 partner (-u,-v) (same handedness) and
  at the sigma_y image (u,-v) of the opposite handedness; total radiated
  amplitude and Q at the root; Stokes S3 there.
* Charges: small loops around Gamma (r = 0.0015, 0.003, 0.006) and a large loop
  (r = 0.06) for both sides; a merge shows up as q_down(Gamma) jumping from +1
  to -1 while the down C points disappear.

Run from the repository root:
    python examples/paper_reproduction/janus_bic/run_fig2d.py [gmax n_gmodes]
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


def to_json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, complex):
        return [o.real, o.imag]
    raise TypeError(type(o))


def ratio(res, side, hand):
    Ep, Em = jc.circ(res["E_t_" + side])
    return Em / Ep if hand == "L" else Ep / Em


def newton(tr, side, hand, x0, tol=1e-9, h=2e-5, maxit=25, rmin=4e-4,
           rmax=0.08):
    x = np.array(x0, float)
    nev = 0
    try:
        for it in range(maxit):
            r0 = tr.eval(*x, store=False)
            nev += 1
            f0 = ratio(r0, side, hand)
            if abs(f0) < tol:
                tr.eval(*x)            # store as continuation reference
                return x, r0, abs(f0), nev, True
            J = np.zeros((2, 2))
            for j in range(2):
                xp = x.copy()
                xp[j] += h
                fj = ratio(tr.eval(*xp, store=False), side, hand)
                nev += 1
                J[:, j] = [(fj - f0).real / h, (fj - f0).imag / h]
            dx = -np.linalg.solve(J, [f0.real, f0.imag])
            lim = min(max(0.25 * np.hypot(*x), 0.002), 0.01)
            n = np.hypot(*dx)
            if n > lim:
                dx *= lim / n
            x = x + dx
            if np.hypot(*x) < rmin:  # ran into the Gamma BIC: no C point here
                return x, None, np.inf, nev, False
            if np.hypot(*x) > rmax:
                return x, None, np.inf, nev, False
        r0 = tr.eval(*x, store=False)
        res = abs(ratio(r0, side, hand))
        return x, r0, res, nev + 1, res < 1e-6
    except (AssertionError, RuntimeError, np.linalg.LinAlgError):
        return x, None, np.inf, nev, False


def branch(args):
    """Continue one (side, handedness) C point in dd, starting at dd0 from the
    scan seed, forward (dd > dd0) and backward (dd < dd0)."""
    side, hand, dd_fwd, dd_bwd, seed, opts = args
    out = []
    for dds in (dd_fwd, dd_bwd):
        prev = np.array(seed, float)
        lost = False
        for dd in dds:
            if lost:   # absorbed by the Gamma BIC (confirmed by loop charges)
                out.append(dict(dd=dd, found=False, absorbed=True))
                continue
            tr = jc.make_tracker(dd, **opts)
            x, r, res, nev, ok = newton(tr, side, hand, prev)
            if not ok:
                near = np.hypot(*prev) < float(os.environ.get("RABS", "0.02"))
                out.append(dict(dd=dd, found=False, absorbed=bool(near),
                                last_u=prev[0], last_v=prev[1]))
                print(f"[{side} {hand}] dd={dd:5.2f}: no C point (last |k|="
                      f"{np.hypot(*prev):.4f}{', absorbed' if near else ''})",
                      flush=True)
                lost = near
                continue
            rc2 = tr.eval(-x[0], -x[1], store=False)
            ry = tr.eval(x[0], -x[1], store=False)
            other = "R" if hand == "L" else "L"
            S0, _, _, S3 = jc.stokes(r["E_t_" + side])
            S0p, _, _, S3p = jc.stokes(r["E_proj_" + side])
            amp = np.linalg.norm(r["E_t_" + side])
            out.append(dict(dd=dd, found=True, u=x[0], v=x[1], resid=res,
                            nev=nev, S3=S3 / S0, S3_proj=S3p / S0p, amp=amp,
                            Q=r["Q"], freq=r["freq"], overlap=r["overlap"],
                            c2_resid=abs(ratio(rc2, side, hand)),
                            sy_resid=abs(ratio(ry, side, other))))
            print(f"[{side} {hand}] dd={dd:5.2f}: (u,v)=({x[0]:+.5f},{x[1]:+.5f})"
                  f" |r|={res:.1e} S3={S3/S0:+.6f} Q={r['Q']:.3e} C2 res="
                  f"{out[-1]['c2_resid']:.1e} sy res={out[-1]['sy_resid']:.1e}"
                  f" nev={nev}", flush=True)
            prev = x
    return side, hand, sorted(out, key=lambda e: e["dd"])


def scan_seeds(dd0, opts, n=(23, 17)):
    """Coarse plaquette scan (u>0 half plane) to seed the C points at dd0."""
    U, V = np.meshgrid(np.linspace(0.004, 0.06, n[0]),
                       np.linspace(-0.02, 0.02, n[1]))
    d = jc.eval_points(dd0, np.c_[U.ravel(), V.ravel()], NPROC, opts)
    seeds = {}
    for side in ("up", "down"):
        sing = jc.plaquette_singularities(U, V, d["E_t_" + side].reshape(
            U.shape + (2,)))
        print(f"scan dd={dd0} {side}: {[(round(a,4), round(b,4), q, round(s,2)) for a, b, q, s, _ in sing]}")
        for hand in ("L", "R"):
            c = [(a, b) for a, b, q, s3, _ in sing
                 if abs(q + 0.5) < 1e-6 and (s3 > 0) == (hand == "L")]
            if len(c) != 1:
                print(f"  WARNING: {len(c)} candidate {side}-{hand} C points")
            if c:
                seeds[(side, hand)] = min(c, key=lambda p: abs(p[1]))
    return seeds


def ring(r, n, c=(0.0, 0.0)):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t)]


def loops(dds, opts, radii=(0.0015, 0.003, 0.006, 0.06), n=32):
    res = {}
    for dd in dds:
        pts = np.vstack([ring(r, n if r < 0.05 else 64) for r in radii])
        d = jc.eval_points(dd, pts, NPROC, opts)
        q = {}
        i0 = 0
        for r in radii:
            m = n if r < 0.05 else 64
            sl = slice(i0, i0 + m)
            i0 += m
            q[r] = (jc.loop_charge(d["E_t_up"][sl]),
                    jc.loop_charge(d["E_t_down"][sl]))
        res[dd] = q
        print("loops dd=%5.2f: " % dd + "  ".join(
            f"r={r}: ({q[r][0]:+.2f},{q[r][1]:+.2f})" for r in radii), flush=True)
    return res


def main():
    import multiprocessing as mp
    opts = dict(jc.DEFAULT)
    if len(sys.argv) > 2:
        opts.update(gmax=float(sys.argv[1]), n_gmodes=int(sys.argv[2]))
    tag = "g%g_m%d" % (opts["gmax"], opts["n_gmodes"])
    dds = np.r_[0.25, 0.5, 0.75, np.arange(1.0, 18.01, 0.5)]
    dd0 = float(os.environ.get("DD0", "1.0"))
    if "DDS" in os.environ:   # e.g. DDS="3:4.5:0.05" for a dense rerun
        a, b, s = map(float, os.environ["DDS"].split(":"))
        dds = np.round(np.arange(a, b + 1e-9, s), 6)
        dd0 = float(os.environ.get("DD0", a))
    from run_checks import bic_on_axis
    f0, ub, Qb = bic_on_axis(0.0, opts)
    print(f"dd=0: f_Gamma={f0:.5f}, off-Gamma BIC at u={ub:.5f} (Q={Qb:.1e})")
    seeds = scan_seeds(dd0, opts)
    dd_fwd = [d for d in dds if d >= dd0]
    dd_bwd = sorted([d for d in dds if d < dd0], reverse=True)
    jobs = [(s, h, dd_fwd, dd_bwd, seeds[(s, h)], opts)
            for s in ("up", "down") for h in ("L", "R") if (s, h) in seeds]
    with mp.get_context("fork").Pool(4) as pool:
        res = pool.map(branch, jobs)
    lp = loops(dds, opts)
    data = dict(dds=list(map(float, dds)), u_bic0=ub, f_gamma0=f0, opts=opts, branches={})
    for side, hand, out in res:
        data["branches"][f"{side}_{hand}"] = out
    data["loops"] = {str(k): {str(r): v for r, v in q.items()}
                     for k, q in lp.items()}
    suffix = os.environ.get("SUFFIX", "")
    fn = os.path.join(OUT, f"fig2d_{tag}{suffix}.json")
    with open(fn, "w") as fh:
        json.dump(data, fh, indent=1, default=to_json)
    print("saved", fn)


if __name__ == "__main__":
    main()
