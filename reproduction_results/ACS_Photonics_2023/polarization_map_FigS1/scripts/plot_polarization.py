"""Generate all figures from stored numerical data; no solver calls."""
from pathlib import Path
import os
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/workspace/setup-legume/matplotlib')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
ROOT=Path(__file__).resolve().parents[1]


def main():
    d=np.load(ROOT/'data/grid31.npz');k=d['kxy'];valid=d['valid'];psi=d['psi']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.linewidth':.8,'savefig.facecolor':'white'})
    fig,ax=plt.subplots(figsize=(5.3,5.0));length=.0068
    v=.5*length*np.column_stack((np.cos(psi[valid]),np.sin(psi[valid])))
    segments=np.stack((k[valid]-v,k[valid]+v),axis=1)
    ax.add_collection(LineCollection(segments,colors='#173b64',linewidths=.85))
    ax.plot(0,0,'ko',ms=4,zorder=4)
    ax.set(xlim=(-.15,.15),ylim=(-.15,.15),xlabel=r'$k_x a/(2\pi)$',ylabel=r'$k_y a/(2\pi)$',aspect='equal')
    ax.set_xticks([-.15,-.1,-.05,0,.05,.1,.15]);ax.set_yticks([-.15,-.1,-.05,0,.05,.1,.15])
    ax.text(.025,.965,'(a)',transform=ax.transAxes,va='top',fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT/'figures/FigS1a_polarization_map.png',dpi=300)
    fig.savefig(ROOT/'figures/FigS1a_polarization_map.pdf')
    plt.close(fig)
    n=31;extent=[-.155,.155,-.155,.155]
    fig,ax=plt.subplots(figsize=(5.3,4.8));q=np.where(valid,np.log10(d['Q']),np.nan).reshape(n,n)
    im=ax.imshow(q,origin='lower',extent=extent,cmap='magma',aspect='equal');ax.plot(0,0,'ko',ms=3)
    ax.set(xlim=(-.15,.15),ylim=(-.15,.15),xlabel=r'$k_x a/(2\pi)$',ylabel=r'$k_y a/(2\pi)$')
    fig.colorbar(im,ax=ax,label=r'$\log_{10} Q_{\mathrm{rad}}$');fig.tight_layout();fig.savefig(ROOT/'figures/FigS1b_Q_factor.png',dpi=300);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,8))
    panels=[(724/d['frequency'],'Wavelength (nm)','viridis'),(d['overlap'],'Parent overlap','viridis'),(d['normalized_stokes'][:,2],r'$S_3/S_0$','coolwarm'),(d['gap'],'Nearest frequency gap (a/λ)','viridis')]
    for ax,(values,title,cmap) in zip(axes.ravel(),panels):
        im=ax.imshow(values.reshape(n,n),origin='lower',extent=extent,cmap=cmap)
        ax.set(title=title,xlim=(-.15,.15),ylim=(-.15,.15),xlabel=r'$k_x a/(2\pi)$',ylabel=r'$k_y a/(2\pi)$');fig.colorbar(im,ax=ax,shrink=.8)
    fig.tight_layout();fig.savefig(ROOT/'figures/mode_tracking_diagnostic.png',dpi=220);plt.close(fig)
    # Explicit data subsets make the archived workflow easy to reuse.
    subsets={'farfield_complex':['kxy','frequency','freq_im','freq_im_s','freq_im_p','Q','u_te','u_tm','l_te','l_tm','Ex','Ey','rad_gvec_u','rad_gvec_l','nopen_u','nopen_l','loss_sum'],
             'polarization_stokes':['kxy','S0','S1','S2','S3','normalized_stokes','psi','ellipticity_angle','valid','intensity_floor'],
             'mode_tracking':['kxy','mode','parent','overlap','margin','gap','all_frequencies','selected_eigenvectors','basis_keys']}
    for name,keys in subsets.items():np.savez_compressed(ROOT/'data'/f'{name}.npz',**{key:d[key] for key in keys})
    print('Figures regenerated from',ROOT/'data/grid31.npz')
if __name__=='__main__':main()
