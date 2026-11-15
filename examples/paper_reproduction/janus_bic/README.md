# Janus BIC – reproduction of Fig. 2 of Kang, Xiao & Chan, PRL 134, 013805 (2025) with legume GME

All numbers below come from the scripts in this folder (legume 1.0.3, this repository).
Nothing from the paper is hard-coded (paper values only appear in the comparison table).

## How to rerun (from the repository root)

```bash
pip install -e .            # legume + numpy/scipy/matplotlib
bash examples/paper_reproduction/janus_bic/run_all.sh
```

or step by step (each step writes to `examples/paper_reproduction/janus_bic/data/`):

| script | what | wall time (4 cores) |
|---|---|---|
| `run_checks.py` | checks 1–3 of the task + ellipse/polygon check → `data/checks.txt` | 3 min |
| `run_fig2bc.py [gmax nmodes]` | bands + Q along X′←Γ→X, Δd = 0, 6, 12 nm | 5 min |
| `run_fig2e.py [N] [gmax nmodes]` | N×N polarization maps, Δd from env `DDS` (default `0,6,12`) | 4 min per Δd (N=41) |
| `run_fig2d.py [gmax nmodes]` | C-point tracking Δd = 0.25…18 nm (env `DDS=a:b:step` for dense reruns, `SUFFIX`) | 35 min |
| `run_convergence.py` | Γ frequency, BIC position, merge Δd vs. gmax / number of guided modes | ~1 h |
| `make_figures.py [tag]` | `figures/fig2a_geometry.png`, `fig2bc_*.png`, `fig2d_*.png`, `fig2e_*.png` | 10 s |

`NPROC` sets the number of worker processes. The scripts set `OMP_NUM_THREADS=1`
*before* importing numpy (the workers are forked; otherwise BLAS threads oversubscribe
the CPU and runs become 10–50× slower).

## Model

* square lattice, a = 336 nm; everything normalized by a.
* two Si₃N₄ layers (n = 2.02, ε = 4.0804), 240 nm each (0.7143 a), directly stacked,
  air above/below. Layer 0 = bottom, layer 1 = top.
* elliptic air holes, `legume.Ellipse` (analytic Fourier transform; `rx`, `ry` are
  **semi**-axes, checked: area = π rx ry). Minor (y) full axis d ± Δd
  (top: +, bottom: −), d = 0.6a; major (x) full axis = β × minor, β = 1.35; no rotation.
  An equal-area inscribed polygon (`legume.Poly`, 32/64/128 vertices) gives the same
  Γ frequency and off-Γ BIC position to 6 digits (`data/checks.txt`), so no polygon is needed.
* GME: `gmax = 5` (81 plane waves), 8 guided modes (TE0, TM0, …, TE3, TM3) of the
  layer-averaged slab (`eps_eff='average'`), `gmode_compute='exact'`.
  Convergence variants: gmax = 3…7, 6/8/10/12 guided modes, `eps_eff='background'`.
* wave vector: (u, v) = (kₓa/2π, k_ya/2π). legume takes k in units of 1/a, so
  `kpoints = 2π(u, v)` (and, because of the time convention below, `kpoints = −2π(u, v)`).

## TM₁ identification and tracking

At Γ, the eigenmode closest to a/λ = 0.565 whose eigenvector weight on TM guided modes
(odd legume `gmode` index) exceeds 0.5 is taken (weight 0.99; it is the lowest TM-like
band at Γ in this window and is a symmetry-protected BIC). Every other k-point is
assigned by maximum eigenvector overlap with the nearest already-evaluated point
(eigenvectors are embedded into a fixed (guided mode × G) grid so they can be compared
across k). Points are swept outward from Γ in angular sectors. The overlap and the
gap to the second-best overlap are stored for every point (`overlap`, `overlap2`).
The same Γ-reference procedure is repeated for every Δd (no fixed band index anywhere).

## Q factor

`Q = Re f / (2 Im f)`, `Im f` = legume `compute_rad` (first-order, golden-rule
coupling of the lossless GME mode to the radiative continuum of the *effective* slab).
For symmetry-protected / accidental BICs Im f reaches ~1e-21…1e-32, i.e. the numerical
noise floor; plots cap Q at 1e12.

## Far field: s/p → (Eₓ, E_y) (derived from the legume source, see `janus_core.py` docstring)

* legume builds D = −(i/ω)∇×H (`ft_field_xy`), i.e. ∇×H = +iωD: an **exp(+iωt)** convention,
  and its radiative modes (`rad_modes`, `clad=0/1`) have one *outgoing* plane wave
  (in that convention) in the lower/upper cladding.
* `_compute_rad_components` returns `c_tm = ⟨ψ|Θ|rad⟩ = conj(a_tm)` and
  `c_te = −i⟨ψ|Θ|rad⟩`, i.e. the conjugates of the physical amplitudes a, with the TE one
  multiplied by −i.
* Evaluating the outgoing E-field of the radiative modes in the cladding (TE normalized by
  1/(√ε ω), TM by 1) gives, for g = k + G, p̂ = g/|g|, q̂ = (p̂_y, −p̂ₓ), cos θ = χ/(√ε ω):

  ```
  exp(+iwt) amplitude, in-plane phase exp(+i g.r):
     F_up   = [ conj(c_te) q̂ − cosθ conj(c_tm) p̂ ] / √ε
     F_down = [ conj(c_te) q̂ + cosθ conj(c_tm) p̂ ] / √ε
  ```
  (the sign flip of the p term comes from E_p ∝ (χp̂ + gẑ) through the top vs (χp̂ − gẑ)
  through the bottom).
* Converting to the standard **exp(−iωt)** convention (E → conj(E)), the wave
  travels with in-plane wave vector −g. So the code runs legume at
  `k_label = −k_phys` and uses, for the zero order G = 0 (selected by |G| < 1e-9 in
  `rad_gvec`, not by array position):

  ```
  E_up(k_phys)   ∝ c_te q̂ − cosθ c_tm p̂       (p̂, q̂ evaluated at k_label)
  E_down(k_phys) ∝ c_te q̂ + cosθ c_tm p̂
  ```
  `E_proj_*` (with cos θ) is the xy projection used for the ellipse plots; `E_t_*`
  (without cos θ) is the transverse field rotated rigidly into the xy plane
  and is used for C points and charges. Both give the same charges (`checks.txt`).
* Handedness: S₃ = 2 Im(Eₓ* E_y) in the fixed lab frame, exp(−iωt). S₃ > 0 ⇔ the
  field rotates counter-clockwise **viewed from +z**. We label S₃ > 0 "LCP" (red) and
  S₃ < 0 "RCP" (blue) **on both sides** (lab-frame labels, so σ_z-symmetric structures get
  identical labels up and down, as in the paper's Fig. 2(e)). For the downward wave the
  helicity w.r.t. its own propagation direction is the opposite.
  Circular components: E∓ = (Eₓ ± iE_y)/√2; E₋ = 0 ⇔ LCP, E₊ = 0 ⇔ RCP.
* Charge: q = (1/4π)∮ d arg(S₁ + iS₂) on counter-clockwise loops in the same (kₓ, k_y)
  axes for both sides (no rotations, flips or swaps).

## C-point search

Zero of the eigenvector-phase–independent ratio r_L = E₋/E₊ (LCP) or r_R = E₊/E₋ (RCP)
of `E_t`, by damped Newton (forward-difference Jacobian, step 2×10⁻⁵), |r| < 10⁻⁹.
Seeds: plaquette-winding scan of the u > 0 half plane at the start Δd; then continuation
in Δd (forward and backward). At every root: S₃/S₀ (= ±1 to 6 digits), Q, |r| at the C₂
partner (−u, −v) (same handedness) and at the σ_y image (u, −v) (opposite handedness) are
stored (all ≲ 10⁻⁹). Missing roots are checked with (i) loop charges around Γ
(r = 0.0015, 0.003, 0.006) and a large loop (r = 0.06, total must be −1),
(ii) full plaquette scans of the 41×41 maps of Fig. 2(e).
A branch is declared absorbed when Newton fails starting within |k| < 0.02 of Γ and the
small-loop charge on that side has flipped.

## Results (production: gmax = 5, 8 guided modes)

Figures: `figures/fig2a_geometry.png`, `figures/fig2bc_g5_m8.png`, `figures/fig2d_g5_m8.png`,
`figures/fig2e_g5_m8.png` (Δd = 0, 6, 12 nm) and `figures/fig2e_extra_dd2_g5_m8.png`
(Δd = 2 nm, where all eight C points are inside the window).

Raw data (`data/`):
* `fig2bc_dd{Δd}_g5_m8.npz`, `fig2e_dd{Δd}_N41_g5_m8.npz`: per k-point `u, v, im` (eigen-index
  of the tracked mode at that k), `freq, freq_im, Q, overlap, overlap2, tmfrac`, the complex
  zero-order couplings `c_te_up, c_tm_up, c_te_down, c_tm_down` (legume convention, at
  k_label = −k_phys), the physical fields `E_t_up/down`, `E_proj_up/down` (exp(−iωt)),
  `freqs_all, tmfrac_all`, and all options (`opt_*`).
* `fig2d_g5_m8*.json`: every C point (side, handedness, Δd, u, v, residual, S₃, Q, f, C₂ and
  σ_y partner residuals) and the loop charges per Δd.
* `checks.txt/json`, `convergence.json`, `diagnostic_geometry.json`.

### Checks before the maps (`data/checks.txt`)
1. Δd = 0: up/down far fields are identical up to a global sign (state fidelity 1 − 1e−10,
   major-axis difference 4e−9°); charges Γ: +1/+1, off-Γ BICs (±0.0283, 0): −1/−1 on both sides.
2. (u,v) = (0.01, 0): E ∥ x (p, |E_y/E_x| ~ 1e−12); (0, 0.01): E ∥ y (p). The TM₁ radiation is
   p-polarized on both mirror lines, i.e. radial around Γ ⇒ q = +1, consistent with check 1.
3. Δd = 12 nm, loops r = 0.002 … 0.03 around Γ: **q_up = −1, q_down = −1** for every radius
   (both the transverse and the projected field). The paper's q_up = +1 is *not* reproduced at
   12 nm in GME — see below; q_up = +1, q_down = −1 is obtained for 3.83 nm < Δd < 6.78 nm.
* `legume.Ellipse` vs 32/64/128-gon: identical f_Γ and u_BIC to 1e−6.

### Comparison with the paper

| quantity | paper (Fig. 2) | this work, GME gmax=5, 8 modes | spread over convergence variants |
|---|---|---|---|
| TM₁ frequency at Γ (a/λ) | ≈ 0.564 | **0.56473** | 0.56463 – 0.56497 (converged to ~3e−4) |
| off-Γ BIC (Δd = 0) | (±≈0.05, 0) on ΓX, none on ΓX′ | **(±0.0283, 0)** on ΓX, none on ΓX′ | 0.022 – 0.031 (not converged; ≥ 8 modes needed, 6 modes → 0.094, `eps_eff=background` → no BIC) |
| Q at Γ / off-Γ BIC | ∞ / ∞ | 1e31 / 1e21 (noise floor) | – |
| Δd = 6, 12 nm on ΓX | off-Γ peak becomes finite, Γ BIC kept | same (Fig. 2c) | – |
| splitting into C points | 4 per side, q = −1/2, opposite handedness in σ_y pairs | same; |r| < 1e−9, S₃ = ±1.000000, C₂/σ_y partner residuals < 1e−9 (96 roots) | – |
| **downward merge** into Γ | ≈ 10 nm | **3.80 < Δd ≤ 3.85 nm** (|k_C| = 0.0031 at 3.80 nm; bisection on r = 5e−4 loop: 3.825–3.836 nm) | 3.0 – 4.3 nm: gmax 3/4/5/6/7 → 3.11/3.54/3.83/4.02/4.29; 10 modes 3.93, 12 modes 2.99 |
| upward C points after the down merge | move away from Γ | first move outward (max |k| = 0.0326 at 3 nm), then turn back | – |
| **upward merge** into Γ | not seen in the plotted range (Δd ≤ 18 nm) | **6.75 < Δd ≤ 6.80 nm** (bisection 6.768–6.782 nm) | 6.55 – 6.83 nm |
| Janus BIC (q_u = +1, q_d = −1) | Δd ≥ 10 nm | **3.83 nm < Δd < 6.78 nm** (e.g. Δd = 6 nm in Fig. 2e) | window opens 3.0–4.3, closes 6.55–6.83 |
| charges at Δd = 12 nm | q_u = +1, q_d = −1 | q_u = −1, q_d = −1 (ordinary BIC, all C points absorbed) | same in all variants with ≥ 8 modes |
| total charge, loop r = 0.06 | −1 | −1 for every Δd | – |

C-point positions (u, v) of the root in the half plane u > 0 (others by C₂: (−u, −v), same handedness):

| Δd (nm) | down LCP | down RCP | up LCP | up RCP |
|---|---|---|---|---|
| 1 | (0.0252, −0.0018) | (0.0252, +0.0018) | (0.0304, +0.0022) | (0.0304, −0.0022) |
| 2 | (0.0208, −0.0030) | (0.0208, +0.0030) | (0.0316, +0.0047) | (0.0316, −0.0047) |
| 3 | (0.0143, −0.0031) | (0.0143, +0.0031) | (0.0318, +0.0073) | (0.0318, −0.0073) |
| 3.8 | (0.0030, −0.0009) | (0.0030, +0.0009) | (0.0311, +0.0093) | (0.0311, −0.0093) |
| 6 | – (absorbed) | – | (0.0219, +0.0119) | (0.0219, −0.0119) |
| 6.75 | – | – | (0.0051, +0.0037) | (0.0051, −0.0037) |

### What is converged and what is not
* **Converged / robust:** Γ frequency (±3e−4); TM₁ band shape (Fig. 2b); symmetry-protected Γ BIC
  for all Δd; σ_z symmetry of the far field at Δd = 0; Γ charge +1 and off-Γ charges −1;
  splitting of each off-Γ BIC into an LCP/RCP pair of −1/2 C points on each side; the
  *downward* C points merging first, producing a Janus BIC with q_up = +1, q_down = −1; the
  subsequent *upward* merge at 6.6–6.8 nm (stable to ±0.15 nm across all variants with ≥ 8
  guided modes). These come out identically from Newton roots, loop charges and plaquette scans.
* **Not converged / deviating from the paper:** the position of the accidental off-Γ BIC
  (u ≈ 0.022–0.031 vs ≈ 0.05) and therefore the downward merge Δd (3.0–4.3 nm vs 10 nm). It
  drifts upward with gmax (0.024 → 0.031 for gmax 3 → 7) and moves non-monotonically with the
  number of guided modes; with `eps_eff='background'` the off-Γ BIC disappears from ΓX entirely.
  Accidental BICs and C-point trajectories are fine-tuned features; GME (effective-slab
  guided basis, first-order radiation, thick 1.43a bilayer) is not accurate enough for them.
* **Diagnostic** (`run_diagnostic_geometry.py`, gmax = 4, *not* the paper geometry): changing
  the layer thickness by a few nm moves u_BIC through 0.042 (t = 244 nm: both sides flip at
  ≈ 6 nm) and 0.055 (t = 248 nm: the *upward* side merges first, ≈ 4 nm, the downward at ≈ 6 nm);
  for t ≥ 252 nm the Γ charge is already −1 at Δd = 0. Hence a ~3 % model error is enough to
  change both the merge Δd and which side merges first. The paper's "upward C points move away
  for all Δd ≤ 18 nm" is therefore not something GME can confirm or refute for this design; a
  full-wave check (RCWA/FEM) of the off-Γ BIC position is the next step.
* Polygon vs analytic ellipse is not a source of error (identical to 1e−6). Mode tracking is
  not a source of error (overlap ≥ 0.993, gap to the second-best overlap ≥ 0.88 on all maps).

