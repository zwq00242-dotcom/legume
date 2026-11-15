"""Synthetic basis algebra only; this is not an ACS modal calculation."""
from _paths import BASE, REPO, CACHE
import csv
import numpy as np

def stokes(ex, ey):
    s0 = abs(ex)**2 + abs(ey)**2
    s1 = (abs(ex)**2 - abs(ey)**2) / s0
    s2 = 2*np.real(ex*np.conj(ey)) / s0
    s3 = 2*np.imag(ex*np.conj(ey)) / s0
    return s1, s2, s3, np.degrees(0.5*np.arctan2(s2, s1))

def main():
    # Chosen synthetic Jones amplitudes; these are NOT legume/ACS modes.
    s, p = 0.7+0.2j, 0.3-0.4j
    rows = []
    for phi in np.arange(0, 360, 30):
        a = np.radians(phi)
        u = np.array([np.cos(a), np.sin(a)])
        v = np.array([np.sin(a), -np.cos(a)])
        e = s*v + p*u
        # Reversing BOTH real in-plane basis axes with unchanged amplitudes.
        eminus = s*(-v) + p*(-u)
        np.testing.assert_allclose(stokes(*e), stokes(*eminus), atol=1e-12)
        rows.append([phi, *stokes(*e), *stokes(*eminus)])
    with (BASE/'data/basis_algebra_check.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['synthetic_phi_deg','plus_S1','plus_S2','plus_S3','plus_psi_deg',
                    'minus_S1','minus_S2','minus_S3','minus_psi_deg'])
        w.writerows(rows)
    print('PASS: 12 synthetic basis reversals preserve all Stokes ratios and psi.')
    print('This is not an ACS single-point or 12-point modal validation.')

if __name__ == '__main__':
    main()
