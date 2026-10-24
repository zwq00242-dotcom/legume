"""Mid-resolution convergence check for the ACS 2023 Fig. 1(a) geometry.

The candidate band is initialized from low-resolution Band 2 at Gamma and
tracked independently toward K and M using adjacent-k eigenvector overlaps.
Case-to-case initialization at Gamma also uses eigenvector overlap on the
common reciprocal-vector/guided-mode basis.
"""

import csv
import json
import os
from pathlib import Path


RESULTS = Path(__file__).resolve().parent / "results" / "acs2023_fig1a_midres"
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
NUMEIG = 12
WINDOW_NM = (740.0, 880.0)
CASES = {
    "A": {"gmax": 3.5, "gmode_inds": [0]},
    "B": {"gmax": 4.5, "gmode_inds": [0]},
    # Official positive horizontal-parity basis: docs/faq.rst and example 1.1.
    "C": {"gmax": 4.5, "gmode_inds": [0, 3]},
}


def build_structure():
    lattice = legume.Lattice("hexagonal")
    phc = legume.PhotCryst(lattice, eps_l=1.0, eps_u=1.0)
    phc.add_layer(d=THICKNESS, eps_b=4.0)
    phc.add_shape(legume.Circle(eps=1.0, r=RADIUS))
    return lattice, phc


def basis_keys_and_vectors(run, ik):
    """Return physical basis labels and eigenvectors for one k point.

    A label is (guided-mode index, reciprocal-index 1, reciprocal-index 2).
    This makes overlaps well-defined when guided modes enter/leave the cutoff
    or when two cases use different gmax values.
    """
    keys = []
    offset = 0
    vectors = np.asarray(run["gme"].eigvecs[ik])
    for block, mode in enumerate(run["gme"].gmode_include[ik]):
        g_indices = np.asarray(run["gme"].ind_modes[ik][block], dtype=int)
        count = len(g_indices)
        for gi in g_indices:
            keys.append((int(mode), int(run["gme"].inds1[gi]),
                         int(run["gme"].inds2[gi])))
        offset += count
    assert offset == vectors.shape[0] == len(keys)
    return keys, vectors


def overlap_matrix(left, il, right, ir):
    """Normalized modal overlaps on the basis shared by two calculations."""
    lkeys, lvec = basis_keys_and_vectors(left, il)
    rkeys, rvec = basis_keys_and_vectors(right, ir)
    lmap = {key: i for i, key in enumerate(lkeys)}
    common = [key for key in rkeys if key in lmap]
    if not common:
        raise RuntimeError("No common basis vectors for mode-overlap tracking")
    li = np.array([lmap[key] for key in common], dtype=int)
    rmap = {key: i for i, key in enumerate(rkeys)}
    ri = np.array([rmap[key] for key in common], dtype=int)
    lv = lvec[li]
    rv = rvec[ri]
    denom = np.linalg.norm(lv, axis=0)[:, None] * np.linalg.norm(rv, axis=0)[None, :]
    overlaps = np.abs(lv.conj().T @ rv)
    np.divide(overlaps, denom, out=overlaps, where=denom > 1e-14)
    return overlaps, len(common)


def track_from_gamma(run, gamma_index, initial_band):
    """Greedily follow maximum eigenvector overlap away from Gamma."""
    nk = run["frequencies"].shape[0]
    tracked = np.full(nk, -1, dtype=int)
    scores = np.full(nk, np.nan)
    margins = np.full(nk, np.nan)
    common_counts = np.zeros(nk, dtype=int)
    tracked[gamma_index] = initial_band
    for ik in range(gamma_index - 1, -1, -1):
        overlaps, common = overlap_matrix(run, ik + 1, run, ik)
        row = overlaps[tracked[ik + 1]]
        order = np.argsort(row)[::-1]
        tracked[ik] = order[0]
        scores[ik] = row[order[0]]
        margins[ik] = row[order[0]] - row[order[1]]
        common_counts[ik] = common
    for ik in range(gamma_index + 1, nk):
        overlaps, common = overlap_matrix(run, ik - 1, run, ik)
        row = overlaps[tracked[ik - 1]]
        order = np.argsort(row)[::-1]
        tracked[ik] = order[0]
        scores[ik] = row[order[0]]
        margins[ik] = row[order[0]] - row[order[1]]
        common_counts[ik] = common
    run["tracked_indices"] = tracked
    run["tracking_overlaps"] = scores
    run["tracking_margins"] = margins
    run["tracking_common_basis_counts"] = common_counts
    run["target_frequencies"] = run["frequencies"][np.arange(nk), tracked]
    run["target_wavelengths"] = A_NM / run["target_frequencies"]


def run_case(name, config, kpoints):
    _, phc = build_structure()
    gme = legume.GuidedModeExp(phc, gmax=config["gmax"], truncate_g="abs")
    gme.run(kpoints=kpoints, gmode_inds=config["gmode_inds"], numeig=NUMEIG,
            compute_im=False, verbose=False)
    frequencies = np.asarray(gme.freqs)
    if frequencies.shape != (kpoints.shape[1], NUMEIG):
        raise RuntimeError(f"Unexpected {name} frequency shape {frequencies.shape}")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise RuntimeError(f"Non-finite or non-positive frequencies in Case {name}")
    wavelengths = A_NM / frequencies
    lo, hi = WINDOW_NM
    relevant = [ib + 1 for ib in range(NUMEIG)
                if np.any((wavelengths[:, ib] >= lo) &
                          (wavelengths[:, ib] <= hi))]
    return {"name": name, "config": config, "gme": gme,
            "frequencies": frequencies, "wavelengths": wavelengths,
            "bands_in_window": relevant}


def save_all_bands_csv(run, path, kpoints, labels):
    output = RESULTS / f"bands_midres_case{run['name']}.csv"
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "k_index", "k_point_label", "path_signed_ka_over_2pi",
                         "kx_a", "ky_a", "band_1based",
                         "normalized_frequency_a_over_lambda", "wavelength_nm"])
        for ik in range(len(path)):
            for ib in range(NUMEIG):
                writer.writerow([run["name"], ik, labels[ik], path[ik],
                                 kpoints[0, ik], kpoints[1, ik], ib + 1,
                                 run["frequencies"][ik, ib], run["wavelengths"][ik, ib]])
    return output


def plot_case(run, path, symmetry_positions):
    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    lo, hi = WINDOW_NM
    relevant = []
    for ib in range(NUMEIG):
        wave = run["wavelengths"][:, ib]
        if np.any((wave >= lo) & (wave <= hi)):
            relevant.append(ib + 1)
            ax.plot(path, wave, lw=1.0, label=f"Band {ib + 1}")
    ax.plot(path, run["target_wavelengths"], "k--", lw=2.0,
            label="Tracked target")
    ax.set_xticks(symmetry_positions, ["K", r"$\Gamma$", "M"])
    ax.set_xlim(path[0], path[-1])
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Path distance from $\Gamma$ ($ka/2\pi$, signed)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title(f"Case {run['name']}: gmax={run['config']['gmax']}, "
                 f"basis={run['config']['gmode_inds']}")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    output = RESULTS / f"bands_midres_case{run['name']}.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    run["bands_in_window"] = relevant
    return output


def main():
    lattice, _ = build_structure()
    # 30 intervals per segment -> 31 points including both endpoints, 61 unique.
    bz_path = lattice.bz_path(["K", "G", "M"], [30, 30])
    kpoints = bz_path["kpoints"]
    gamma = bz_path["indexes"][1]
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(kpoints), axis=0))]
    path = (distance - distance[gamma]) / (2 * np.pi)
    symmetry_positions = path[bz_path["indexes"]]
    labels = [""] * len(path)
    for label, index in zip(["K", "Gamma", "M"], bz_path["indexes"]):
        labels[index] = label

    runs = {name: run_case(name, config, kpoints)
            for name, config in CASES.items()}

    # Case A starts from prior Band 2. Other cases are matched at Gamma by
    # common-basis eigenvector overlap rather than by sorted eigenvalue index.
    initial = {"A": 1}
    ab, ab_common = overlap_matrix(runs["A"], gamma, runs["B"], gamma)
    initial["B"] = int(np.argmax(ab[initial["A"]]))
    bc, bc_common = overlap_matrix(runs["B"], gamma, runs["C"], gamma)
    initial["C"] = int(np.argmax(bc[initial["B"]]))
    gamma_match_overlaps = {
        "A_to_B": float(ab[initial["A"], initial["B"]]),
        "B_to_C": float(bc[initial["B"], initial["C"]]),
    }
    for name, run in runs.items():
        track_from_gamma(run, gamma, initial[name])
        save_all_bands_csv(run, path, kpoints, labels)
    plot_case(runs["A"], path, symmetry_positions)
    plot_case(runs["B"], path, symmetry_positions)

    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    styles = {"A": "-", "B": "--", "C": ":"}
    for name, run in runs.items():
        label = f"Case {name}: gmax={run['config']['gmax']}, {run['config']['gmode_inds']}"
        ax.plot(path, run["target_wavelengths"], styles[name], lw=2.0, label=label)
        ax.scatter(symmetry_positions,
                   run["target_wavelengths"][bz_path["indexes"]], s=25)
    ax.set_xticks(symmetry_positions, ["K", r"$\Gamma$", "M"])
    ax.set_xlim(path[0], path[-1])
    target_values = np.concatenate([run["target_wavelengths"]
                                    for run in runs.values()])
    padding = 0.03 * (target_values.max() - target_values.min())
    ax.set_ylim(target_values.min() - padding, target_values.max() + padding)
    ax.set_xlabel(r"Path distance from $\Gamma$ ($ka/2\pi$, signed)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title("Tracked target-band convergence")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(RESULTS / "target_band_convergence.png", dpi=180)
    plt.close(fig)

    convergence_csv = RESULTS / "target_band_convergence.csv"
    with convergence_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "k_index", "k_point_label",
                         "path_signed_ka_over_2pi", "kx_a", "ky_a",
                         "sorted_band_1based", "normalized_frequency_a_over_lambda",
                         "wavelength_nm", "adjacent_mode_overlap", "overlap_margin"])
        for name, run in runs.items():
            for ik in range(len(path)):
                writer.writerow([name, ik, labels[ik], path[ik], kpoints[0, ik],
                                 kpoints[1, ik], run["tracked_indices"][ik] + 1,
                                 run["target_frequencies"][ik],
                                 run["target_wavelengths"][ik],
                                 run["tracking_overlaps"][ik],
                                 run["tracking_margins"][ik]])

    high_indices = dict(zip(["K", "Gamma", "M"], bz_path["indexes"]))
    high_values = {name: {label: float(run["target_wavelengths"][index])
                          for label, index in high_indices.items()}
                   for name, run in runs.items()}
    differences = {
        "A_vs_B_nm": {label: high_values["A"][label] - high_values["B"][label]
                       for label in high_indices},
        "B_vs_C_nm": {label: high_values["B"][label] - high_values["C"][label]
                       for label in high_indices},
    }
    tracking = {}
    for name, run in runs.items():
        valid = np.isfinite(run["tracking_overlaps"])
        index_changes = np.flatnonzero(np.diff(run["tracked_indices"]) != 0).tolist()
        ambiguous = np.flatnonzero(valid & ((run["tracking_overlaps"] < 0.8) |
                                             (run["tracking_margins"] < 0.1))).tolist()
        tracking[name] = {
            "gamma_initial_sorted_band_1based": int(initial[name] + 1),
            "unique_sorted_band_indices_1based":
                sorted((np.unique(run["tracked_indices"]) + 1).tolist()),
            "sorted_index_change_steps": index_changes,
            "ambiguous_overlap_steps": ambiguous,
            "minimum_adjacent_overlap": float(np.min(run["tracking_overlaps"][valid])),
            "minimum_overlap_margin": float(np.min(run["tracking_margins"][valid])),
        }

    near_gamma = [gamma - 1, gamma, gamma + 1]
    shape_checks = {
        name: {
            "gamma_local_wavelength_minimum": bool(
                run["target_wavelengths"][gamma] < run["target_wavelengths"][gamma - 1]
                and run["target_wavelengths"][gamma] < run["target_wavelengths"][gamma + 1]),
            "near_gamma_wavelengths_nm_K_side_Gamma_M_side":
                [float(run["target_wavelengths"][i]) for i in near_gamma],
        } for name, run in runs.items()
    }
    metadata = {
        "structure": {"a_nm": A_NM, "thickness_over_a": THICKNESS,
                      "radius_over_a": RADIUS, "slab_eps": 4.0,
                      "upper_cladding_eps": 1.0, "lower_cladding_eps": 1.0},
        "path": {"labels": ["K", "Gamma", "M"], "intervals_per_segment": 30,
                 "points_per_segment_including_endpoints": 31,
                 "total_unique_points": len(path),
                 "high_symmetry_ka": {label: kpoints[:, index].tolist()
                                      for label, index in high_indices.items()}},
        "solver": {name: {**run["config"], "truncate_g": "abs",
                            "numeig": NUMEIG, "compute_im": False,
                            "plane_waves": int(run["gme"].gvec.shape[1]),
                            "basis_dimensions_min_max":
                                [int(np.min(run["gme"].N_basis)),
                                 int(np.max(run["gme"].N_basis))]}
                   for name, run in runs.items()},
        "basis_rationale": ("[0,3] is the next positive horizontal-parity basis "
                            "documented in docs/faq.rst and examples 0.4/1.1; [1,2] "
                            "is the negative-parity quasi-TM basis."),
        "gamma_cross_case_common_basis_counts": {"A_to_B": ab_common,
                                                   "B_to_C": bc_common},
        "gamma_cross_case_match_overlaps": gamma_match_overlaps,
        "high_symmetry_target_wavelengths_nm": high_values,
        "signed_wavelength_differences_nm": differences,
        "tracking": tracking,
        "shape_checks": shape_checks,
        "bands_sampled_in_740_880_nm":
            {name: run["bands_in_window"] for name, run in runs.items()},
        "limitations": [
            "Adjacent-k tracking uses eigenvector overlap on the shared guided-mode/Fourier basis.",
            "Greedy single-mode tracking can be ambiguous inside an exactly degenerate subspace.",
            "The paper figure/data were not supplied, so correspondence is assessed by dispersion shape only.",
            "This is a mid-resolution check, not a final gmax or basis convergence study.",
            "Constant eps=4 is used without material dispersion.",
        ],
    }
    (RESULTS / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("Legume:", legume.__file__)
    print("k points:", len(path), "(31 per segment including endpoints)")
    for name, run in runs.items():
        print(f"Case {name}: gmax={run['config']['gmax']}, "
              f"gmode_inds={run['config']['gmode_inds']}, numeig={NUMEIG}, "
              f"plane_waves={run['gme'].gvec.shape[1]}, "
              f"basis_dim={min(run['gme'].N_basis)}--{max(run['gme'].N_basis)}")
        print("  Gamma matched sorted band:", initial[name] + 1)
        print("  K/Gamma/M wavelengths (nm):", high_values[name])
        print("  tracking:", tracking[name])
        print("  Gamma-neighbor shape:", shape_checks[name])
    print("Gamma cross-case match overlaps:", gamma_match_overlaps)
    print("Signed wavelength differences (first minus second, nm):", differences)
    print("All gme.run calls completed; compute_im=False")
    print("=== mid-resolution convergence run completed successfully ===")


if __name__ == "__main__":
    main()
