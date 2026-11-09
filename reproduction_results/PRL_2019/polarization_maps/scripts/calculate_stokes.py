"""Raw upper G=0 coupling (local Jones estimate NOT USED for broken C2), in the fixed Cartesian laboratory frame.

See README for reciprocity and basis signs; Cartesian fields are assembled
from reciprocal partners in compute_farfield.py.
Amplitudes retain Legume's loss normalization, not absolute electric-field units.
"""
import numpy as np


def extract(gme, ik, mode):
    fi, fs, fp, coup, gv = gme.compute_rad_sp(ik, [int(mode)])
    raw = {}; channels = {}
    for side in ('u', 'l'):
        vectors = np.asarray(gv[side][0])
        zero = np.flatnonzero(np.linalg.norm(vectors, axis=0) < 1e-12)
        if len(zero) != 1:
            raise RuntimeError('Exactly one open G=0 channel is required')
        for pol in ('te', 'tm'):
            raw[side+'_'+pol] = complex(coup[side+'_'+pol][0][zero[0]])
        channels[side] = vectors.T
    f = float(gme.freqs[ik, mode]); om = 2*np.pi*f
    loss_sum = sum(np.sum(abs(coup[key][0])**2) for key in coup)
    reconstructed_fi = np.sqrt(om**2+1j*loss_sum).imag/(2*np.pi)
    if not np.isclose(reconstructed_fi, fi[0], rtol=1e-10, atol=1e-30):
        raise RuntimeError('Radiation normalization mismatch')
    return dict(**raw, freq_im=float(fi[0]),
                freq_im_s=float(fs[0]), freq_im_p=float(fp[0]),
                nopen_u=len(channels['u']), nopen_l=len(channels['l']),
                channels_u=channels['u'], channels_l=channels['l'],
                loss_sum=float(loss_sum))


def stokes(ex, ey, kxy, relative_floor=1e-12):
    s0 = abs(ex)**2+abs(ey)**2
    s1 = abs(ex)**2-abs(ey)**2
    s2 = 2*np.real(ex.conj()*ey)
    s3 = 2*np.imag(ex.conj()*ey)
    floor = max(float(s0.max())*relative_floor, 1e-28)
    valid = (s0 > floor)
    normalized = np.full((len(s0),3), np.nan)
    normalized[valid] = np.array([s1,s2,s3]).T[valid]/s0[valid,None]
    psi = np.full(len(s0),np.nan); chi=psi.copy()
    psi[valid] = .5*np.arctan2(s2[valid],s1[valid])
    chi[valid] = .5*np.arcsin(np.clip(normalized[valid,2],-1,1))
    return dict(S0=s0,S1=s1,S2=s2,S3=s3,normalized_stokes=normalized,
                psi=psi,ellipticity_angle=chi,valid=valid,intensity_floor=floor)
