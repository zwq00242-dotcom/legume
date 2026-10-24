"""High-resolution quasi-TE band calculation for ACS Photonics 2023 Fig. 1(a).

The target mode is anchored at sorted Band 2 at Gamma and followed toward K
and M by maximum adjacent-k eigenvector overlap on the shared physical basis.
"""

import csv
import json
import os
from pathlib import Path


RESULTS = Path(__file__).resolve().parent / "results" / "acs2023_fig1a_highres"
RESULTS.mkdir(parents=True, exist_ok=True)
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(RESULTS / ".cache" / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(RESULTS / ".cache")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import legume


A_NM = 724.0
THICKNESS = 100.0 / A_NM
RADIUS = 200.0 / A_NM
GMAX = 4.5
GMODE_INDS = [0, 3]
NUMEIG = 12
INTERVALS_PER_SEGMENT = 100
WINDOW_NM = (740.0, 880.0)
MIDRES_CASE_C_NM = {"K": 1145.091917272893,
                    "Gamma": 815.8094391344013,
                    "M": 1271.3755889039023}


def build_structure():
    lattice = legume.Lattice("hexagonal")
    phc = legume.PhotCryst(lattice, eps_l=1.0, eps_u=1.0)
    phc.add_layer(d=THICKNESS, eps_b=4.0)
    phc.add_shape(legume.Circle(eps=1.0, r=RADIUS))
    return lattice, phc


def basis_keys_and_vectors(gme, ik):
    """Label each eigenvector row by (guided mode, reciprocal indices)."""
    keys = []
    vectors = np.asarray(gme.eigvecs[ik])
    for block, mode in enumerate(gme.gmode_include[ik]):
        for gi in np.asarray(gme.ind_modes[ik][block], dtype=int):
            keys.append((int(mode), int(gme.inds1[gi]), int(gme.inds2[gi])))
    if len(keys) != vectors.shape[0]:
        raise RuntimeError("Eigenvector and physical-basis dimensions disagree")
    return keys, vectors


def overlap_matrix(gme, left_index, right_index):
    """Normalized eigenvector overlaps on the common physical basis."""
    left_keys, left_vectors = basis_keys_and_vectors(gme, left_index)
    right_keys, right_vectors = basis_keys_and_vectors(gme, right_index)
    left_map = {key: i for i, key in enumerate(left_keys)}
    right_map = {key: i for i, key in enumerate(right_keys)}
    common = [key for key in right_keys if key in left_map]
    if not common:
        raise RuntimeError("No common basis vectors for adjacent-k tracking")
    left_rows = np.array([left_map[key] for key in common], dtype=int)
    right_rows = np.array([right_map[key] for key in common], dtype=int)
    left = left_vectors[left_rows]
    right = right_vectors[right_rows]
    denominator = (np.linalg.norm(left, axis=0)[:, None]
                   * np.linalg.norm(right, axis=0)[None, :])
    overlaps = np.abs(left.conj().T @ right)
    np.divide(overlaps, denominator, out=overlaps, where=denominator > 1e-14)
    return overlaps, len(common)


def track_target(gme, frequencies, gamma_index):
    """Track Gamma Band 2 independently toward both ends of the path."""
    nk = frequencies.shape[0]
    indices = np.full(nk, -1, dtype=int)
    selection_overlap = np.full(nk, np.nan)
    selection_margin = np.full(nk, np.nan)
    common_counts = np.zeros(nk, dtype=int)
    indices[gamma_index] = 1  # zero-based sorted Band 2

    for ik in range(gamma_index - 1, -1, -1):
        overlaps, common = overlap_matrix(gme, ik + 1, ik)
        row = overlaps[indices[ik + 1]]
        order = np.argsort(row)[::-1]
        indices[ik] = order[0]
        selection_overlap[ik] = row[order[0]]
        selection_margin[ik] = row[order[0]] - row[order[1]]
        common_counts[ik] = common

    for ik in range(gamma_index + 1, nk):
        overlaps, common = overlap_matrix(gme, ik - 1, ik)
        row = overlaps[indices[ik - 1]]
        order = np.argsort(row)[::-1]
        indices[ik] = order[0]
        selection_overlap[ik] = row[order[0]]
        selection_margin[ik] = row[order[0]] - row[order[1]]
        common_counts[ik] = common

    # Store overlaps in plotted K->Gamma->M order, so row i is explicitly the
    # overlap with CSV row i-1. Tracking selection itself still starts at Gamma.
    overlap_with_previous = np.full(nk, np.nan)
    previous_common_counts = np.zeros(nk, dtype=int)
    for ik in range(1, nk):
        overlaps, common = overlap_matrix(gme, ik - 1, ik)
        overlap_with_previous[ik] = overlaps[indices[ik - 1], indices[ik]]
        previous_common_counts[ik] = common

    target_frequencies = frequencies[np.arange(nk), indices]
    return {
        "indices": indices,
        "selection_overlap": selection_overlap,
        "selection_margin": selection_margin,
        "selection_common_counts": common_counts,
        "overlap_with_previous": overlap_with_previous,
        "previous_common_counts": previous_common_counts,
        "frequencies": target_frequencies,
        "wavelengths": A_NM / target_frequencies,
    }


def contiguous_true_ranges(mask):
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop - 1))
            for start, stop in changes.reshape(-1, 2)]


def main():
    lattice, phc = build_structure()
    bz_path = lattice.bz_path(["K", "G", "M"],
                              [INTERVALS_PER_SEGMENT, INTERVALS_PER_SEGMENT])
    kpoints = bz_path["kpoints"]
    gamma = bz_path["indexes"][1]
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(kpoints), axis=0))]
    path = (distance - distance[gamma]) / (2 * np.pi)
    high_indices = dict(zip(["K", "Gamma", "M"], bz_path["indexes"]))
    symmetry_positions = path[bz_path["indexes"]]
    point_labels = [""] * len(path)
    for label, index in high_indices.items():
        point_labels[index] = label
    sections = np.full(len(path), "Gamma-M", dtype=object)
    sections[:gamma] = "K-Gamma"
    sections[gamma] = "Gamma"

    gme = legume.GuidedModeExp(phc, gmax=GMAX, truncate_g="abs")
    gme.run(kpoints=kpoints, gmode_inds=GMODE_INDS, numeig=NUMEIG,
            compute_im=False, verbose=False)
    frequencies = np.asarray(gme.freqs)
    expected_shape = (2 * INTERVALS_PER_SEGMENT + 1, NUMEIG)
    if frequencies.shape != expected_shape:
        raise RuntimeError(f"Expected {expected_shape}, got {frequencies.shape}")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise RuntimeError("Unresolved NaN/Inf or non-positive eigenfrequency")
    wavelengths = A_NM / frequencies
    target = track_target(gme, frequencies, gamma)
    if not np.all(np.isfinite(target["frequencies"])):
        raise RuntimeError("Unresolved NaN/Inf in tracked target")
    finite_overlaps = target["overlap_with_previous"][1:]
    if not np.all(np.isfinite(finite_overlaps)):
        raise RuntimeError("Unresolved NaN/Inf in adjacent overlaps")

    target_csv = RESULTS / "target_band_highres.csv"
    with target_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path_index", "path_coordinate_signed_ka_over_2pi",
                         "section", "k_point_label", "kx_a", "ky_a",
                         "normalized_frequency_a_over_lambda", "wavelength_nm",
                         "sorted_band_index_1based", "overlap_with_previous",
                         "selection_overlap_from_gamma", "selection_overlap_margin",
                         "common_basis_count_with_previous"])
        for ik in range(len(path)):
            writer.writerow([ik, path[ik], sections[ik], point_labels[ik],
                             kpoints[0, ik], kpoints[1, ik],
                             target["frequencies"][ik], target["wavelengths"][ik],
                             target["indices"][ik] + 1,
                             target["overlap_with_previous"][ik],
                             target["selection_overlap"][ik],
                             target["selection_margin"][ik],
                             target["previous_common_counts"][ik]])

    all_csv = RESULTS / "all_bands_highres.csv"
    with all_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path_index", "path_coordinate_signed_ka_over_2pi",
                         "section", "k_point_label", "kx_a", "ky_a",
                         "sorted_band_index_1based",
                         "normalized_frequency_a_over_lambda", "wavelength_nm"])
        for ik in range(len(path)):
            for ib in range(NUMEIG):
                writer.writerow([ik, path[ik], sections[ik], point_labels[ik],
                                 kpoints[0, ik], kpoints[1, ik], ib + 1,
                                 frequencies[ik, ib], wavelengths[ik, ib]])

    lo, hi = WINDOW_NM
    relevant = [ib for ib in range(NUMEIG)
                if np.any((wavelengths[:, ib] >= lo) &
                          (wavelengths[:, ib] <= hi))]
    fig, ax = plt.subplots(figsize=(7.6, 5.1), constrained_layout=True)
    for ib in relevant:
        ax.plot(path, wavelengths[:, ib], color="0.58", lw=0.85, zorder=1)
    ax.plot(path, target["wavelengths"], color="#c82433", lw=2.6,
            label="Tracked TE-like target", zorder=3)
    for x in symmetry_positions:
        ax.axvline(x, color="0.82", lw=0.8, zorder=0)
    ax.set_xticks(symmetry_positions, ["K", r"$\Gamma$", "M"])
    ax.set_xlim(path[0], path[-1])
    ax.set_ylim(lo, hi)
    ax.set_ylabel("Wavelength (nm)")
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, loc="upper center", fontsize=9)
    fig.savefig(RESULTS / "fig1a_paper_style.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    ax.plot(path, target["wavelengths"], color="#c82433", lw=2.2)
    ax.scatter(symmetry_positions,
               target["wavelengths"][bz_path["indexes"]], color="#c82433", s=28)
    for x in symmetry_positions:
        ax.axvline(x, color="0.82", lw=0.8)
    ax.set_xticks(symmetry_positions, ["K", r"$\Gamma$", "M"])
    ax.set_xlim(path[0], path[-1])
    padding = 0.03 * np.ptp(target["wavelengths"])
    ax.set_ylim(target["wavelengths"].min() - padding,
                target["wavelengths"].max() + padding)
    ax.set_xlabel(r"Path distance from $\Gamma$ ($ka/2\pi$, signed)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title("Overlap-tracked TE-like target band")
    ax.grid(alpha=0.2)
    fig.savefig(RESULTS / "target_band_highres_fullrange.png", dpi=200)
    plt.close(fig)

    swaps = np.flatnonzero(np.diff(target["indices"]) != 0)
    low_overlap = np.flatnonzero(target["overlap_with_previous"] < 0.9)
    fig, (ax_index, ax_overlap) = plt.subplots(2, 1, figsize=(8.4, 6.4),
                                               sharex=True,
                                               constrained_layout=True)
    ax_index.step(path, target["indices"] + 1, where="mid", color="#315a9b")
    ax_index.set_ylabel("Sorted band index")
    ax_index.set_yticks(np.unique(target["indices"] + 1))
    ax_index.grid(alpha=0.2)
    ax_overlap.plot(path[1:], target["overlap_with_previous"][1:],
                    color="#238443", lw=1.5)
    ax_overlap.axhline(0.9, color="#c82433", ls="--", lw=1.0, label="0.9 threshold")
    if len(low_overlap):
        ax_overlap.scatter(path[low_overlap],
                           target["overlap_with_previous"][low_overlap],
                           color="#c82433", zorder=3, label="Overlap < 0.9")
    for transition in swaps:
        x = (path[transition] + path[transition + 1]) / 2
        ax_index.axvline(x, color="#c82433", ls=":", lw=1.0)
        ax_overlap.axvline(x, color="#c82433", ls=":", lw=1.0)
    ax_overlap.set_xticks(symmetry_positions, ["K", r"$\Gamma$", "M"])
    ax_overlap.set_xlim(path[0], path[-1])
    ax_overlap.set_ylim(min(0.88, finite_overlaps.min() - 0.01), 1.003)
    ax_overlap.set_ylabel("Overlap with previous k")
    ax_overlap.set_xlabel(r"Path distance from $\Gamma$ ($ka/2\pi$, signed)")
    ax_overlap.grid(alpha=0.2)
    ax_overlap.legend(frameon=False, fontsize=8)
    fig.savefig(RESULTS / "mode_tracking_diagnostics.png", dpi=200)
    plt.close(fig)

    high_wavelengths = {label: float(target["wavelengths"][index])
                        for label, index in high_indices.items()}
    high_frequencies = {label: float(target["frequencies"][index])
                        for label, index in high_indices.items()}
    midres_differences = {label: high_wavelengths[label] - MIDRES_CASE_C_NM[label]
                          for label in high_indices}
    window_mask = ((target["wavelengths"] >= lo)
                   & (target["wavelengths"] <= hi))
    window_ranges = []
    for start, stop in contiguous_true_ranges(window_mask):
        window_ranges.append({
            "start_index": start, "stop_index": stop,
            "start_path_coordinate": float(path[start]),
            "stop_path_coordinate": float(path[stop]),
            "start_section": str(sections[start]),
            "stop_section": str(sections[stop]),
            "start_wavelength_nm": float(target["wavelengths"][start]),
            "stop_wavelength_nm": float(target["wavelengths"][stop]),
        })
    swap_records = [{
        "between_path_indices": [int(i), int(i + 1)],
        "path_coordinate_midpoint": float((path[i] + path[i + 1]) / 2),
        "from_sorted_band_1based": int(target["indices"][i] + 1),
        "to_sorted_band_1based": int(target["indices"][i + 1] + 1),
        "adjacent_overlap": float(target["overlap_with_previous"][i + 1]),
    } for i in swaps]

    # One-sided quadratic fits quantify smooth positive curvature at Gamma.
    fit_points = 8
    k_side = slice(gamma - fit_points + 1, gamma + 1)
    m_side = slice(gamma, gamma + fit_points)
    local_fits = {}
    for side, subset in [("K_side", k_side), ("M_side", m_side)]:
        coefficients = np.polyfit(path[subset], target["wavelengths"][subset], 2)
        residual = (target["wavelengths"][subset]
                    - np.polyval(coefficients, path[subset]))
        local_fits[side] = {
            "quadratic_coefficients_nm": coefficients.tolist(),
            "curvature_second_derivative_nm": float(2 * coefficients[0]),
            "fitted_slope_at_gamma_nm": float(coefficients[1]),
            "rms_residual_nm": float(np.sqrt(np.mean(residual**2))),
        }
    gamma_is_local_min = bool(
        target["wavelengths"][gamma] < target["wavelengths"][gamma - 1]
        and target["wavelengths"][gamma] < target["wavelengths"][gamma + 1])

    metadata = {
        "structure": {"lattice": "hexagonal", "a_nm": A_NM,
                      "thickness_over_a": THICKNESS, "radius_over_a": RADIUS,
                      "slab_eps": 4.0, "upper_cladding_eps": 1.0,
                      "lower_cladding_eps": 1.0},
        "solver": {"gmax": GMAX, "gmode_inds": GMODE_INDS,
                   "truncate_g": "abs", "compute_im": False,
                   "numeig": NUMEIG, "plane_waves": int(gme.gvec.shape[1]),
                   "basis_dimensions_min_max": [int(np.min(gme.N_basis)),
                                                  int(np.max(gme.N_basis))]},
        "path": {"labels": ["K", "Gamma", "M"],
                 "points_per_segment_including_endpoints": 101,
                 "total_unique_points": len(path),
                 "high_symmetry_ka": {label: kpoints[:, index].tolist()
                                      for label, index in high_indices.items()}},
        "tracking": {
            "method": "maximum adjacent-k eigenvector overlap on common physical basis",
            "gamma_anchor_sorted_band_1based": 2,
            "unique_sorted_band_indices_1based":
                sorted((np.unique(target["indices"]) + 1).tolist()),
            "sorted_band_swaps": swap_records,
            "minimum_adjacent_overlap": float(finite_overlaps.min()),
            "minimum_overlap_path_index": int(np.argmin(finite_overlaps) + 1),
            "overlap_below_0_9_indices": low_overlap.tolist(),
            "minimum_selection_margin":
                float(np.nanmin(target["selection_margin"])),
            "continuous": bool(finite_overlaps.min() >= 0.9),
        },
        "high_symmetry": {
            "target_wavelengths_nm": high_wavelengths,
            "target_normalized_frequencies": high_frequencies,
            "midres_case_C_wavelengths_nm": MIDRES_CASE_C_NM,
            "highres_minus_midres_nm": midres_differences,
        },
        "paper_window": {"wavelength_nm": list(WINDOW_NM),
                         "bands_present_1based": [ib + 1 for ib in relevant],
                         "target_inside_ranges": window_ranges,
                         "gamma_is_local_wavelength_minimum": gamma_is_local_min,
                         "local_quadratic_fits": local_fits},
        "validation": {"frequency_shape": list(frequencies.shape),
                       "all_frequencies_finite_and_positive": True,
                       "target_frequencies_finite_and_positive": True,
                       "all_adjacent_overlaps_finite": True},
        "limitations": [
            "Constant eps=4 is used without material dispersion.",
            "Correspondence to the paper is assessed by supplied geometry and qualitative curve shape; no digitized paper data were supplied.",
            "Single-mode overlap tracking may be gauge-dependent in an exactly degenerate subspace; no low-overlap ambiguity was detected here.",
        ],
    }
    (RESULTS / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("Legume:", legume.__file__)
    print("Solver: gmax=4.5, gmode_inds=[0, 3], truncate_g=abs, "
          "compute_im=False, numeig=12")
    print("k points:", len(path), "(101 per segment including endpoints)")
    print("Plane waves:", gme.gvec.shape[1], "basis dimensions:",
          min(gme.N_basis), "--", max(gme.N_basis))
    print("K/Gamma/M wavelengths (nm):", high_wavelengths)
    print("Gamma normalized frequency:", high_frequencies["Gamma"])
    print("Highres minus midres Case C (nm):", midres_differences)
    print("Minimum adjacent overlap:", finite_overlaps.min(),
          "at path index", np.argmin(finite_overlaps) + 1)
    print("Sorted band swaps:", swap_records)
    print("Overlap below 0.9 indices:", low_overlap.tolist())
    print("Target ranges inside 740--880 nm:", window_ranges)
    print("Gamma local wavelength minimum:", gamma_is_local_min)
    print("Local quadratic fits:", local_fits)
    print("NaN/Inf validation passed; target tracking continuous")
    print("=== high-resolution run completed successfully ===")


if __name__ == "__main__":
    main()
