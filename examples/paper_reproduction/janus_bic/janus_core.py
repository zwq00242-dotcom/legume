"""
Core routines for reproducing Fig. 2 of
M. Kang, M. Xiao, C. T. Chan, "Janus Bound States in the Continuum with
Asymmetric Topological Charges", PRL 134, 013805 (2025)
with legume's guided-mode expansion (GME).

Units / conventions (read this before using the data files)
-----------------------------------------------------------
* Lengths are normalized by the lattice constant a = 336 nm.
* Reduced wave vector  (u, v) = (kx a / 2pi, ky a / 2pi).
  legume expects k in units of 1/a, so  kpoints = 2*pi*(u, v).
* Reduced frequency  f = a / lambda = omega a / (2 pi c)  (= legume `freqs`).
* Q = Re(f) / (2 Im(f)),  Im(f) = legume `freqs_im` (first-order / Fermi-golden-
  rule radiative rate of GME: the guided-mode basis is diagonalized without
  losses, and the coupling to the radiative continuum of the effective
  (layer-averaged eps) slab is evaluated perturbatively).

Far-field convention (derived from legume/gme/gme.py and slab_modes.py)
-----------------------------------------------------------------------
legume builds D = -(i/omega) curl H (see `ft_field_xy`), i.e. its fields obey
curl H = +i omega D, which is the exp(+i omega t) time convention, and its
radiative modes `rad_modes(..., clad)` are the states with a single *outgoing*
plane wave (in that convention) in cladding `clad`.  `_compute_rad_components`
returns
      c_tm = < psi | Theta | rad_tm >  = conj(physical amplitude a_tm)
      c_te = -1j < psi | Theta | rad_te > ,  i.e. a_te = -1j*conj(c_te).
Evaluating the outgoing E-field of the radiative modes (E = D/eps in the
cladding, TE normalized by 1/(sqrt(eps) omega), TM by 1) gives, in the
exp(+i omega t) convention and for in-plane wave vector g = k + G (legume label)

    upper cladding:  F_up   = (1/sqrt(eps)) [ conj(c_te) q  - cos(th) conj(c_tm) p ]
    lower cladding:  F_down = (1/sqrt(eps)) [ conj(c_te) q  + cos(th) conj(c_tm) p ]

with p = g/|g| (in-plane), q = (p_y, -p_x), cos(th) = chi/(sqrt(eps) omega).
(The extra minus sign of the upper-cladding p-wave comes from E_p ~ (chi p + g z)
for a wave leaving through the top versus (chi p - g z) through the bottom.)
The outgoing wave in that convention has in-plane phase exp(+i g.r + i omega t),
i.e. it travels along -g.  Converting to the standard physics convention
exp(-i omega t): the complex amplitude is conj(F) and the physical in-plane wave
vector is  k_phys = -k_label.  Hence, for G = 0,

    E_up(k_phys = -k_label)   ∝  c_te q - cos(th) c_tm p
    E_down(k_phys = -k_label) ∝  c_te q + cos(th) c_tm p        (exp(-i w t))

Everything stored by this module is expressed in the *physical* k_phys and the
exp(-i omega t) convention: we simply run legume at k_label = -k_phys.
`E_*_proj` is the in-plane (xy) projection (with the cos(th) factor) used for
the polarization-ellipse maps; `E_*_t` ("transverse") drops cos(th), i.e. the
transverse field rotated rigidly into the xy plane along the plane of
incidence; it is used for the C-point search (circular polarization in the
plane transverse to the propagation direction).

Handedness:  S3 = 2 Im(conj(Ex) Ey) in the fixed lab frame (x, y), exp(-i w t).
S3 > 0  <=> the E vector rotates counter-clockwise when viewed from +z (looking
down onto the xy plane).  We call S3>0 "LCP" and S3<0 "RCP" on BOTH sides
(lab-frame convention, so that sigma_z-symmetric structures give identical
labels up and down).  For the downward wave the helicity w.r.t. its own
propagation direction is the opposite; this is stated in the plots.
Circular components in the lab frame:  E_pm = (Ex -/+ i Ey)/sqrt(2);
E_minus = 0  <=> E ∝ (1, i)  <=> S3 = +1 (LCP);  E_plus = 0 <=> RCP.
Topological charge of the major axis: q = (1/4pi) ∮ d arg(S1 + i S2) along a
counter-clockwise loop in the (kx, ky) plane, same axes for both sides.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import legume  # noqa: E402

A_NM = 336.0
N_SIN = 2.02
EPS_SIN = N_SIN ** 2
T_LAYER_NM = 240.0
D_REL = 0.6           # d = 0.6 a  (minor-axis full length)
BETA = 1.35           # major / minor full-length ratio


def build_phc(dd_nm=0.0, shape="ellipse", npoly=64, a_nm=A_NM,
              t_layer_nm=T_LAYER_NM, d_rel=D_REL, beta=BETA, eps=EPS_SIN):
    """Bilayer Si3N4 PhC slab in air. Layer 0 = bottom (z in [0, t]),
    layer 1 = top (z in [t, 2t]).  Minor (y) full axis of top/bottom hole is
    d +/- dd; major (x) full axis = beta * minor.  Semi-axes passed to legume.
    """
    lat = legume.Lattice("square")
    phc = legume.PhotCryst(lat, eps_l=1.0, eps_u=1.0)
    t = t_layer_nm / a_nm
    for sgn in (-1.0, +1.0):          # bottom (d - dd), then top (d + dd)
        minor_full = d_rel + sgn * dd_nm / a_nm
        ry = 0.5 * minor_full
        rx = 0.5 * beta * minor_full
        phc.add_layer(d=t, eps_b=eps)
        if shape == "ellipse":
            phc.add_shape(legume.Ellipse(eps=1.0, rx=rx, ry=ry))
        elif shape == "poly":
            th = np.linspace(0, 2 * np.pi, npoly, endpoint=False)
            # equal-area inscribed polygon correction: scale radius so that the
            # polygon area equals the ellipse area
            s = np.sqrt(2 * np.pi / (npoly * np.sin(2 * np.pi / npoly)))
            phc.add_shape(legume.Poly(eps=1.0, x_edges=s * rx * np.cos(th),
                                      y_edges=s * ry * np.sin(th)))
        else:
            raise ValueError(shape)
    return phc


class Solver:
    """Thin wrapper: one GME object per (dd, gmax), single-k evaluations."""

    def __init__(self, dd_nm=0.0, gmax=4.0, n_gmodes=6, numeig=8,
                 sigma=0.565, shape="ellipse", npoly=64, eps_eff="average",
                 geom=None):
        self.eps_eff = eps_eff
        self.dd_nm = dd_nm
        self.gmax = gmax
        self.n_gmodes = n_gmodes
        self.numeig = numeig
        self.sigma = sigma
        self.phc = build_phc(dd_nm, shape=shape, npoly=npoly, **(geom or {}))
        self.gme = legume.GuidedModeExp(self.phc, gmax=gmax + 1e-6)
        self.gvec = self.gme.gvec
        self.i_g0 = int(np.argmin(np.hypot(self.gvec[0], self.gvec[1])))
        assert np.hypot(*self.gvec[:, self.i_g0]) < 1e-12

    # ------------------------------------------------------------------
    def solve(self, u, v):
        """Run GME at physical reduced wave vector (u, v).  Returns dict with
        freqs and embedded eigenvectors for overlap tracking."""
        kl = -2 * np.pi * np.array([[u], [v]], dtype=float)  # k_label = -k_phys
        g = self.gme
        g.run(kpoints=kl, gmode_inds=list(range(self.n_gmodes)),
              numeig=self.numeig, eig_sigma=self.sigma, compute_im=False,
              eps_eff=self.eps_eff, verbose=False)
        freqs = np.array(g.freqs[0])
        V = np.array(g.eigvecs[0])
        emb, tmfrac = self._embed(V)
        return dict(u=u, v=v, freqs=freqs, emb=emb, tmfrac=tmfrac)

    def _embed(self, V):
        """Map eigenvectors (basis: guided mode x G subset) onto a fixed
        (n_gmodes x NG) grid so that vectors at different k can be overlapped;
        also return the weight on TM (odd-index) guided modes."""
        g = self.gme
        NG = self.gvec.shape[1]
        gk = np.hypot(self.gvec[0] + g.kpoints[0, 0] + g.delta_gx,
                      self.gvec[1] + g.kpoints[1, 0])
        emb = np.zeros((self.n_gmodes * NG, V.shape[1]), dtype=complex)
        tm = np.zeros(V.shape[1])
        cnt = 0
        for im1, mode in enumerate(g.gmode_include[0]):
            ind = g._get_guided(gk, 0, mode)[0]
            n = g.modes_numg[0][im1]
            blk = V[cnt:cnt + n]
            emb[mode * NG + np.asarray(ind)] = blk
            if mode % 2 == 1:
                tm += np.sum(np.abs(blk) ** 2, axis=0)
            cnt += n
        nrm = np.linalg.norm(emb, axis=0)
        return emb / nrm, tm / nrm ** 2

    # ------------------------------------------------------------------
    def far_field(self, im):
        """Zero-order (G = 0) far field of mode `im` of the last solve().
        Returns dict in the exp(-i w t), physical-k convention."""
        g = self.gme
        fim, rc, rv = g.compute_rad(0, [im])
        f = float(g.freqs[0][im])
        out = dict(freq=f, freq_im=float(fim[0]),
                   Q=f / (2 * float(fim[0])) if fim[0] > 0 else np.inf)
        k = g.kpoints[:, 0]
        for side, key in (("up", "u"), ("down", "l")):
            gv = np.array(rv[key][0])
            if gv.size == 0:
                raise RuntimeError("no radiative channel")
            dist = np.hypot(gv[0], gv[1])
            j = int(np.argmin(dist))
            assert dist[j] < 1e-9, "G=0 order not found in rad_gvec"
            cte = complex(rc[key + "_te"][0][j])
            ctm = complex(rc[key + "_tm"][0][j])
            gx, gy = k[0] + g.delta_gx, k[1]
            gn = np.hypot(gx, gy)
            px, py = gx / gn, gy / gn
            qx, qy = py, -px
            om = 2 * np.pi * f
            cos_th = np.sqrt(max(om ** 2 - gn ** 2, 0.0)) / om
            sgn = -1.0 if side == "up" else +1.0
            Et = np.array([cte * qx + sgn * ctm * px, cte * qy + sgn * ctm * py])
            Ep = np.array([cte * qx + sgn * cos_th * ctm * px,
                           cte * qy + sgn * cos_th * ctm * py])
            out["c_te_" + side] = cte
            out["c_tm_" + side] = ctm
            out["E_t_" + side] = Et
            out["E_proj_" + side] = Ep
            out["n_orders_" + side] = int(gv.shape[1])
        out["cos_th"] = cos_th
        return out


# ----------------------------------------------------------------------
def stokes(E):
    """E: (..., 2) complex lab-frame (Ex, Ey). Returns S0..S3 (unnormalized)."""
    Ex, Ey = E[..., 0], E[..., 1]
    S0 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    S1 = np.abs(Ex) ** 2 - np.abs(Ey) ** 2
    S2 = 2 * np.real(np.conj(Ex) * Ey)
    S3 = 2 * np.imag(np.conj(Ex) * Ey)
    return S0, S1, S2, S3


def circ(E):
    """Lab-frame circular components: E_plus (zero at RCP), E_minus (zero at
    LCP)."""
    Ex, Ey = E[..., 0], E[..., 1]
    return (Ex - 1j * Ey) / np.sqrt(2), (Ex + 1j * Ey) / np.sqrt(2)


def loop_charge(E_loop):
    """q = (1/4pi) sum of wrapped increments of arg(S1 + i S2) around a closed
    counter-clockwise loop (E_loop: (N, 2), first point not repeated)."""
    _, S1, S2, _ = stokes(np.asarray(E_loop))
    ph = np.angle(S1 + 1j * S2)
    dph = np.diff(np.concatenate([ph, ph[:1]]))
    dph = (dph + np.pi) % (2 * np.pi) - np.pi
    return np.sum(dph) / (4 * np.pi)


# ----------------------------------------------------------------------
class Tracker:
    """Evaluate the tracked TM1 mode at arbitrary (u, v): picks the eigenmode
    with the largest eigenvector overlap with a reference vector (taken from
    the nearest previously evaluated point, by default)."""

    def __init__(self, solver, ref_emb, ref_uv=(0.0, 0.0), min_overlap=0.5):
        self.s = solver
        self.refs = [(np.array(ref_uv, float), ref_emb)]
        self.min_overlap = min_overlap

    def nearest_ref(self, u, v):
        d = [np.hypot(*(r[0] - (u, v))) for r in self.refs]
        return self.refs[int(np.argmin(d))][1]

    def eval(self, u, v, ref=None, store=True, far=True):
        r = self.s.solve(u, v)
        ref = self.nearest_ref(u, v) if ref is None else ref
        ov = np.abs(ref.conj() @ r["emb"])
        im = int(np.argmax(ov))
        srt = np.sort(ov)[::-1]
        res = dict(u=u, v=v, im=im, overlap=float(ov[im]),
                   overlap2=float(srt[1]) if srt.size > 1 else 0.0,
                   freqs=r["freqs"], tmfrac=float(r["tmfrac"][im]),
                   tmfrac_all=r["tmfrac"])
        if store:
            self.refs.append((np.array([u, v], float), r["emb"][:, im]))
        if far:
            res.update(self.s.far_field(im))
        return res


def gamma_tm1(solver, f_target=0.565, u0=0.0):
    """Identify TM1 at Gamma: closest mode to f_target among those with
    TM-dominated guided-mode content. Returns (im, freqs, tmfrac, emb)."""
    r = solver.solve(u0, 0.0)
    cand = np.where(r["tmfrac"] > 0.5)[0]
    im = int(cand[np.argmin(np.abs(r["freqs"][cand] - f_target))])
    return im, r


def make_tracker(dd_nm, gmax=4.0, n_gmodes=6, numeig=8, f_target=0.565,
                 shape="ellipse", npoly=64, ref_emb=None, eps_eff="average",
                 geom=None):
    """geom: optional overrides of build_phc geometry (diagnostics only)."""
    s = Solver(dd_nm, gmax=gmax, n_gmodes=n_gmodes, numeig=numeig,
               sigma=f_target, shape=shape, npoly=npoly, eps_eff=eps_eff,
               geom=geom)
    im, r = gamma_tm1(s, f_target)
    emb = r["emb"][:, im] if ref_emb is None else ref_emb
    tr = Tracker(s, emb)
    tr.gamma = dict(im=im, freqs=r["freqs"], tmfrac=r["tmfrac"],
                    f=r["freqs"][im])
    return tr


# ----------------------------------------------------------------------
# Parallel evaluation helpers
# ----------------------------------------------------------------------
DEFAULT = dict(gmax=5.0, n_gmodes=8, numeig=8, f_target=0.565,
               shape="ellipse", npoly=64)

FIELDS_SCALAR = ["u", "v", "im", "overlap", "overlap2", "tmfrac", "freq",
                 "freq_im", "Q", "cos_th", "c_te_up", "c_tm_up",
                 "c_te_down", "c_tm_down", "n_orders_up", "n_orders_down"]
FIELDS_VEC = ["E_t_up", "E_t_down", "E_proj_up", "E_proj_down"]


def _worker(args):
    dd_nm, pts, opts = args
    tr = make_tracker(dd_nm, **opts)
    out = []
    for (u, v) in pts:            # pts are sorted by |k| within a sector
        out.append(tr.eval(u, v))
    return out, tr.gamma


def eval_points(dd_nm, uv, nproc=4, opts=None):
    """Evaluate the tracked TM1 mode at the points uv (N,2). Points are split
    into angular sectors; each sector is swept outward from Gamma so that the
    eigenvector-overlap continuation always starts from the Gamma TM1 mode.
    Returns a dict of arrays (same order as uv) + gamma info."""
    import multiprocessing as mp
    opts = dict(DEFAULT, **(opts or {}))
    uv = np.asarray(uv, float)
    ang = np.arctan2(uv[:, 1], uv[:, 0])
    rad = np.hypot(uv[:, 0], uv[:, 1])
    nsec = max(nproc * 2, 8)
    sec = np.floor((ang + np.pi) / (2 * np.pi) * nsec).astype(int) % nsec
    jobs, idx = [], []
    for s in range(nsec):
        ii = np.where(sec == s)[0]
        if ii.size == 0:
            continue
        ii = ii[np.argsort(rad[ii])]
        idx.append(ii)
        jobs.append((dd_nm, [tuple(p) for p in uv[ii]], opts))
    if nproc > 1:
        with mp.get_context("fork").Pool(nproc) as pool:
            res = pool.map(_worker, jobs)
    else:
        res = [_worker(j) for j in jobs]
    N = uv.shape[0]
    data = {}
    for f in FIELDS_SCALAR:
        dt = complex if f.startswith("c_") else float
        data[f] = np.zeros(N, dtype=dt)
    for f in FIELDS_VEC:
        data[f] = np.zeros((N, 2), dtype=complex)
    ne = opts["numeig"]
    data["freqs_all"] = np.zeros((N, ne))
    data["tmfrac_all"] = np.zeros((N, ne))
    for ii, (lst, gam) in zip(idx, res):
        for i, r in zip(ii, lst):
            for f in FIELDS_SCALAR + FIELDS_VEC:
                data[f][i] = r[f]
            data["freqs_all"][i] = r["freqs"]
            data["tmfrac_all"][i] = r["tmfrac_all"]
    data["gamma_f"] = res[0][1]["f"]
    data["dd_nm"] = dd_nm
    for k, val in opts.items():
        data["opt_" + k] = val
    return data


def plaquette_singularities(U, V, E):
    """Locate polarization singularities on a regular grid.
    U, V: (Ny, Nx) meshgrid, E: (Ny, Nx, 2).  For every plaquette compute the
    winding of arg(S1 + i S2) (counter-clockwise) -> q in multiples of 1/2.
    Returns list of (u_c, v_c, q, mean S3/S0, mean S0)."""
    S0, S1, S2, S3 = stokes(E)
    ph = np.angle(S1 + 1j * S2)

    def d(a, b):
        x = b - a
        return (x + np.pi) % (2 * np.pi) - np.pi
    w = (d(ph[:-1, :-1], ph[:-1, 1:]) + d(ph[:-1, 1:], ph[1:, 1:]) +
         d(ph[1:, 1:], ph[1:, :-1]) + d(ph[1:, :-1], ph[:-1, :-1]))
    q = w / (4 * np.pi)
    out = []
    for j, i in zip(*np.where(np.abs(q) > 0.25)):
        sl = (slice(j, j + 2), slice(i, i + 2))
        out.append((U[sl].mean(), V[sl].mean(), round(2 * q[j, i]) / 2,
                    float(np.mean(S3[sl] / S0[sl])), float(np.mean(S0[sl]))))
    return out
