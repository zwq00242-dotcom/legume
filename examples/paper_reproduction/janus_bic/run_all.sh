#!/usr/bin/env bash
# Full reproduction of Fig. 2 (Kang, Xiao, Chan, PRL 134, 013805 (2025)).
# Run from the repository root:   bash examples/paper_reproduction/janus_bic/run_all.sh
# Uses NPROC=4 worker processes by default (NPROC=... to change).
# Wall time on 4 cores: ~1.5-2 h for the production setting (gmax=5, 8 guided
# modes) plus ~1 h for the convergence variants.
set -e
D=examples/paper_reproduction/janus_bic

python $D/run_checks.py                         # checks 1-3 (+ ellipse vs polygon)
python $D/run_fig2bc.py                         # bands + Q, dd = 0, 6, 12 nm
python $D/run_fig2e.py 41                       # 41x41 polarization maps
DDS="2" python $D/run_fig2e.py 41               # extra row: dd = 2 nm (C points visible on both sides)
python $D/run_fig2d.py                          # C-point tracking, dd = 0.25..18 nm
DDS="3.5:4.0:0.05" SUFFIX="_dense_down" python $D/run_fig2d.py   # dense, downward merge
DDS="6.5:7.0:0.05" SUFFIX="_dense_up"   python $D/run_fig2d.py   # dense, upward merge
python $D/run_convergence.py                    # merge interval vs gmax / guided modes / eps_eff
python $D/run_diagnostic_geometry.py            # diagnostic only: detuned t / beta
python $D/make_figures.py                       # figures/ *.png
