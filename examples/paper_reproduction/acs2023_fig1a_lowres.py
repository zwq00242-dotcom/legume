"""Low-resolution quasi-TE bands for the user-supplied ACS 2023 geometry.

API references: docs/faq.rst (horizontal mirror symmetry),
legume/phc/lattice.py (hexagonal lattice and BZ points).
This is a single-guided-band approximation, not a convergence study.
"""

import csv
import json
import os
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results" / "acs2023_fig1a_lowres"
RESULTS.mkdir(parents=True, exist_ok=True)
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(RESULTS / ".cache" / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(RESULTS / ".cache")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import legume


def main():
    a_nm = 724.0
    thickness = 100.0 / a_nm
    radius = 200.0 / a_nm
    lattice = legume.Lattice("hexagonal")
    phc = legume.PhotCryst(lattice, eps_l=1.0, eps_u=1.0)
    phc.add_layer(d=thickness, eps_b=4.0)
    phc.add_shape(legume.Circle(eps=1.0, r=radius))
    path = lattice.bz_path(["K", "G", "M"], [12, 12])
    kpoints = path["kpoints"]  # Cartesian k*a in radians, not k*a/(2*pi).
    gamma = path["indexes"][1]
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(kpoints), axis=0))]
    signed_distance = (distance - distance[gamma]) / (2 * np.pi)
    gme = legume.GuidedModeExp(phc, gmax=3.5, truncate_g="abs")
    # FAQ: the lowest positive horizontal-parity guided band is quasi-TE.
    # kz_symmetry would select vertical parity and is deliberately left unset.
    gme.run(kpoints=kpoints, gmode_inds=[0], numeig=10,
            compute_im=False, verbose=False)
    frequencies = np.asarray(gme.freqs)
    assert frequencies.shape == (25, 10), frequencies.shape
    assert np.all(np.isfinite(frequencies)) and np.all(frequencies >= 0)
    # Legume returns f*a/c = omega*a/(2*pi*c) = a/lambda.
    wavelengths = np.full_like(frequencies, np.inf)
    np.divide(a_nm, frequencies, out=wavelengths, where=frequencies > 0)
    csv_path = RESULTS / "bands.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["k_index", "segment", "path_signed_ka_over_2pi",
                         "kx_a", "ky_a", "kx_a_over_2pi", "ky_a_over_2pi",
                         "band_1based", "normalized_frequency_a_over_lambda",
                         "wavelength_nm"])
        for ik in range(len(distance)):
            segment = "K-Gamma" if ik < gamma else ("Gamma" if ik == gamma else "Gamma-M")
            for ib in range(frequencies.shape[1]):
                writer.writerow([ik, segment, signed_distance[ik], *kpoints[:, ik],
                                 *(kpoints[:, ik] / (2 * np.pi)), ib + 1,
                                 frequencies[ik, ib], wavelengths[ik, ib]])

    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    in_window = (wavelengths >= 740) & (wavelengths <= 880)
    relevant = []
    for ib in range(frequencies.shape[1]):
        if np.any(in_window[:, ib]):
            relevant.append(ib + 1)
            ax.plot(signed_distance, wavelengths[:, ib], ".-", ms=4,
                    lw=1.2, label=f"Band {ib + 1}")
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xticks(signed_distance[path["indexes"]], ["K", r"$\Gamma$", "M"])
    ax.set_xlim(signed_distance[0], signed_distance[-1])
    ax.set_ylim(740, 880)
    ax.set_xlabel(r"Path distance from $\Gamma$ ($ka/2\pi$, signed)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title("Low-resolution quasi-TE bands: air / Si3N4 / air\n"
                 "a=724 nm, h=100 nm, r=200 nm; eps=4, gmax=3.5, basis=[0]")
    ax.grid(alpha=0.25)
    if relevant:
        ax.legend(ncol=2, fontsize=8)
    fig.savefig(RESULTS / "bands.png", dpi=180)
    plt.close(fig)

    metadata = dict(a_nm=a_nm, thickness_over_a=thickness, radius_over_a=radius,
                    slab_eps=4.0, cladding_eps=1.0, gmax=3.5, truncate_g="abs",
                    gmode_inds=[0], numeig=10, compute_im=False,
                    kpoints_total=25, kpoints_per_segment_including_endpoints=13,
                    high_symmetry_ka={label: kpoints[:, idx].tolist()
                                     for label, idx in zip(["K", "Gamma", "M"], path["indexes"])},
                    plane_waves=int(gme.gvec.shape[1]),
                    gamma_frequencies=frequencies[gamma].tolist(),
                    gamma_wavelengths_nm=wavelengths[gamma].tolist(),
                    bands_sampled_in_740_880_nm=relevant,
                    caveats=["No gmax or guided-basis convergence study.",
                             "Band indices are frequency sorted, not overlap tracked.",
                             "Constant eps=4; no material dispersion.",
                             "Original Figure 1(a) was not supplied for visual comparison."])
    (RESULTS / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("legume:", legume.__file__)
    print("gme.run completed; shape:", frequencies.shape)
    print("Normalized thickness, radius:", thickness, radius)
    print("High-symmetry k*a:", metadata["high_symmetry_ka"])
    print("Plane waves:", metadata["plane_waves"])
    print("Gamma frequencies:", frequencies[gamma])
    print("Gamma wavelengths (nm):", wavelengths[gamma])
    for ib in relevant:
        selected = wavelengths[in_window[:, ib - 1], ib - 1]
        print(f"Band {ib}: sampled wavelengths in window "
              f"{selected.min():.6f}--{selected.max():.6f} nm "
              f"({len(selected)} points); Gamma={wavelengths[gamma, ib - 1]:.6f} nm")
    print("Near Gamma (signed k*a/2pi, wavelengths of bands in target window):")
    for ik in [gamma - 1, gamma, gamma + 1]:
        print(signed_distance[ik], wavelengths[ik, np.array(relevant, dtype=int) - 1])
    print("All 250 frequency/wavelength/k-point records:", csv_path)
    print("=== low-resolution run completed successfully ===")


if __name__ == "__main__":
    main()
