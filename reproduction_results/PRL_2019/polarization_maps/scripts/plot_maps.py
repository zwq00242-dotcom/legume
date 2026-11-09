"""Polarization ellipses from saved complex eigenmode radiation, no solver."""
import os
os.environ.setdefault('MPLCONFIGDIR','/workspace/setup-legume/matplotlib')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap,Normalize
ROOT=Path(__file__).resolve().parents[1]
CMAP=LinearSegmentedColormap.from_list('helicity',['#2157b0','#999999','#cc272d'])

def draw(ax,z,alpha,cpoints):
    k=z['kxy'];n=int(np.sqrt(len(k)));valid=z['valid'];norm=z['normalized_stokes'];extent=np.max(abs(k))
    # Plot every second point of a 51 grid to keep the ellipses legible.
    stride=2 if n>40 else 1;mask=np.zeros((n,n),bool);mask[::stride,::stride]=True
    ix=np.flatnonzero(valid & mask.ravel());psi=z['psi'][ix];chi=z['ellipticity_angle'][ix]
    aa=(2*extent/(n-1))*stride*.39;bb=aa*abs(np.tan(chi))
    t=np.linspace(0,2*np.pi,37);x=aa*np.cos(t);y=bb[:,None]*np.sin(t)
    dx=np.cos(psi)[:,None]*x-np.sin(psi)[:,None]*y;dy=np.sin(psi)[:,None]*x+np.cos(psi)[:,None]*y
    lines=np.stack((k[ix,0,None]+dx,k[ix,1,None]+dy),axis=-1)
    lc=LineCollection(lines,cmap=CMAP,norm=Normalize(-1,1),linewidths=.65);lc.set_array(norm[ix,2]);ax.add_collection(lc)
    if alpha==0:ax.plot(0,0,'ko',ms=4)
    else:
        for c in cpoints:
            if c['alpha']==alpha and c['gmax']==4.5:
                ax.plot(*c['kxy'],'o',ms=4,color=CMAP((c['normalized_stokes'][2]+1)/2),mec='black',mew=.4,zorder=5)
    ax.set(xlim=(-extent,extent),ylim=(-extent,extent),aspect='equal',xlabel=r'$k_x a/(2\pi)$',ylabel=r'$k_y a/(2\pi)$',title=rf'$\alpha={alpha:g}$')
    return lc

def main():
    plt.rcParams.update({'font.size':10,'axes.linewidth':.8})
    cp=json.loads((ROOT/'validation/cpoints.json').read_text())
    data=[np.load(ROOT/'data'/f'{tag}.npz') for tag in ['alpha0','alpha0p1','alpha1']]
    for a,tag,z in zip([0,.1,1],['alpha0','alpha0p1','alpha1'],data):
        fig,ax=plt.subplots(figsize=(5.5,4.7));lc=draw(ax,z,a,cp)
        fig.colorbar(lc,ax=ax,label=r'$S_3/S_0$ (red +, blue −)',shrink=.8);fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/f'{tag}_polarization.{ext}',dpi=300)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13.6,4.5))
    for ax,a,z,label in zip(axes,[0,.1,1],data,['(a)','(b)','(c)']):
        lc=draw(ax,z,a,cp);ax.text(.025,.96,label,transform=ax.transAxes,va='top')
    fig.subplots_adjust(left=.06,right=.91,bottom=.17,top=.9,wspace=.34)
    cax=fig.add_axes([.93,.22,.012,.6]);fig.colorbar(lc,cax=cax,label=r'$S_3/S_0$')
    for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/f'Fig2_polarization_comparison.{ext}',dpi=300)
    plt.close(fig)
if __name__=='__main__':main()
