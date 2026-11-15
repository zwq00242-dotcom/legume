"""Independent modal diagnostic, remote work e205b78. No 2-D map or source edits."""
from _paths import BASE, REPO, CACHE
import os
os.environ['MPLBACKEND']='Agg'
os.environ['MPLCONFIGDIR']=str(CACHE / 'matplotlib')
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys, json, csv, shutil
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
ROOT=BASE
SRC=REPO/'reproduction_results/ACS_Photonics_2023/polarization_map_FigS1/scripts'
sys.path.insert(0,str(SRC)); sys.path.insert(0,str(REPO))
from compute_modes import solve, STRUCT
from track_modes import snapshot,select
from extract_farfield import extract

def stok(e):
 x,y=e;s0=abs(x)**2+abs(y)**2
 if s0<1e-28:return [float('nan')]*4
 s1=(abs(x)**2-abs(y)**2)/s0;s2=2*np.real(x.conjugate()*y)/s0;s3=2*np.imag(x.conjugate()*y)/s0
 return [float(s1),float(s2),float(s3),float(np.degrees(.5*np.arctan2(s2,s1)))]

def axes(k,f):
 k=2*np.pi*np.asarray(k)+[1e-15,0];p=k/np.linalg.norm(k);s=np.array([p[1],-p[0]])
 return p,s,np.sqrt(1-np.dot(k,k)/(2*np.pi*f)**2)

def jones(raw,k,f):
 p,s,ct=axes(k,f);return raw['u_te']*s-raw['u_tm']*ct*p

def field(ik,m,z,typ='e'):
 return np.asarray(g.ft_field_xy(typ,ik,m,z))

def mirror(ik,m):
 F=field(ik,m,h/2);M=np.diag([1,-1,1]);G=g.gvec*np.array([[1],[-1]])
 perm=[np.argmin(np.linalg.norm(g.gvec-v[:,None],axis=0)) for v in G.T]
 if np.max(np.linalg.norm(g.gvec[:,perm]-G,axis=0))>1e-8:return [np.nan,np.nan]
 b=M@F[:,perm];c=np.vdot(F,b)/np.vdot(F,F)
 return [float(c.real),float(np.linalg.norm(b-c*F)/np.linalg.norm(F))]

def symgamma(m):
 F=field(0,m,h/2);t=np.pi/3;R=np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]])
 G=R[:2,:2].T@g.gvec;perm=[np.argmin(np.linalg.norm(g.gvec-v[:,None],axis=0)) for v in G.T]
 b=R@F[:,perm];c=np.vdot(F,b)/np.vdot(F,F)
 a=field(0,m,h*.3);b=field(0,m,h*.7)*np.array([[1],[1],[-1]])
 zc=np.vdot(a,b)/np.vdot(a,a)
 return dict(C6_real=float(c.real),C6_imag=float(c.imag),C6_residual=float(np.linalg.norm(R@F[:,perm]-c*F)/np.linalg.norm(F)),mirror_y=mirror(0,m),z_mirror=float(zc.real),z_residual=float(np.linalg.norm(b-zc*a)/np.linalg.norm(a)))

def volume(ik,m,n=32):
 # Independent background scattering-field reciprocity, following archived audit.
 k=np.asarray(points[ik]);om=2*np.pi*g.freqs[ik,m];gn=2*np.pi*np.linalg.norm(k)
 p=k/np.linalg.norm(k);q=np.array([p[1],-p[0]]);iz=np.argmin(np.linalg.norm(g.gvec,axis=0))
 xs,ys,chi=g._get_rad(np.array([gn]),om,'te',1);xp,yp,_=g._get_rad(np.array([gn]),om,'tm',1)
 t,w=leggauss(n);I=np.zeros(2,complex)
 for z,wt in zip((t+1)*h/2,w*h/2):
  P=field(ik,m,z,'d')[:,iz]-g.eps_array[1]*field(ik,m,z)[:,iz]
  ep=np.exp(1j*chi[1,0]*(z-h/2));em=1/ep
  As,Bs=ys[1,0]*ep,xs[1,0]*em;Ap,Bp=yp[1,0]*ep,xp[1,0]*em
  Es=om*(As+Bs)*np.r_[q,0]
  Ep=(-chi[1,0]*(Ap-Bp)*np.r_[p,0]-gn*(Ap+Bp)*np.array([0,0,1]))/(om*g.eps_array[1])
  I+=wt*np.array([Es@P,Ep@P])
 _,_,ct=axes(k,g.freqs[ik,m]);return I[0]*q+I[1]*ct*p,I

def writecsv(name,rows):
 p=ROOT/name
 if p.exists():
  backup=p.with_name(p.name+'.before_remote_discovery')
  if not backup.exists():shutil.copy2(p,backup)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

h=STRUCT['thickness_over_a'];angles=np.arange(12)*np.pi/6
points=np.vstack(([[0,0],[.005,0],[.01,0],[.02,0],[.04,0],[-.04,0]],.02*np.c_[np.cos(angles),np.sin(angles)]))
g=solve(points);print('solved',len(points),'points x 12 modes',flush=True)
sn=[snapshot(g,i) for i in range(len(points))]
raw={}
def get(i,m):
 if (i,m) not in raw:raw[i,m]=extract(g,i,m)
 return raw[i,m]
# Track four branches from q=.04 inward and around the circle, independent of Q/frequency ranking.
tracks={};radii=[];loop=[];summary={};table=[]
current1=field(4,1,h/2).ravel()
for m in range(12):
 if m==0:
  fi,_,_,c,gv=g.compute_rad_sp(4,[m]);r=None
 else:r=get(4,m)
 F=field(4,m,h/2).ravel();ov=abs(np.vdot(current1,F))/(np.linalg.norm(current1)*np.linalg.norm(F))
 gm,gscore,gmargin=select((sn[4][0],sn[4][1][:,m]),sn[0])
 row=dict(mode_index=m,frequency_normalized=float(g.freqs[4,m]),frequency_THz=float(g.freqs[4,m]*299792.458/724),wavelength_nm=float(724/g.freqs[4,m]),frequency_im=r['freq_im'] if r else 0,Q=float(g.freqs[4,m]/(2*r['freq_im'])) if r else float('inf'),abs_Ex=float(abs(r['Ex'])) if r else np.nan,abs_Ey=float(abs(r['Ey'])) if r else np.nan,Ex_Ey_ratio=float(abs(r['Ex'])/max(abs(r['Ey']),1e-300)) if r else np.nan)
 st=stok([r['Ex'],r['Ey']]) if r else [np.nan]*4
 row.update(dict(zip(['S1_S0','S2_S0','S3_S0','psi_deg'],st)))
 row.update(gamma_best_mode=gm,gamma_frequency=float(g.freqs[0,gm]),gamma_overlap=gscore,gamma_margin=gmargin,mirror_y_electric=mirror(4,m)[0],mirror_residual=mirror(4,m)[1],field_overlap_to_current= float(ov),eigenvector_overlap_to_current=float(abs(np.vdot(sn[4][1][:,1],sn[4][1][:,m]))),upper_G0_E_fourier_norm=float(np.linalg.norm(field(4,m,h+.1)[:,np.argmin(np.linalg.norm(g.gvec,axis=0))])),z_mirror_parity='even E under slab-midplane reflection (basis 0,3)',raw_upper_te=str(r['u_te']) if r else '',raw_upper_tm=str(r['u_tm']) if r else '',raw_lower_te=str(r['l_te']) if r else '',raw_lower_tm=str(r['l_tm']) if r else '',channels_u=json.dumps(r['channels_u'].tolist()) if r else '[]',channels_l=json.dumps(r['channels_l'].tolist()) if r else '[]',nopen_u=r['nopen_u'] if r else 0,nopen_l=r['nopen_l'] if r else 0)
 table.append(row)
writecsv('data/single_point_modes.csv',table)
for seed in [1,2,3,4]:
 state=(sn[4][0],sn[4][1][:,seed]);path=[]
 for ik in [4,3,2,1,0]:
  m,ov,margin=select(state,sn[ik]);state=(sn[ik][0],sn[ik][1][:,m]);path.append((ik,m,ov,margin))
  r=get(ik,m);radii.append(dict(branch_at_q04=seed,qx=float(points[ik,0]),mode=m,f=float(g.freqs[ik,m]),Q=float(g.freqs[ik,m]/(2*r['freq_im'])),S0=float(abs(r['Ex'])**2+abs(r['Ey'])**2),overlap=ov,margin=margin))
 gamma_m=path[-1][1]
 # independent radial link .04->.02, then continuous 12-point loop
 m,_,_=select((sn[4][0],sn[4][1][:,seed]),sn[6]);state=(sn[6][0],sn[6][1][:,m]);chosen=[];scores=[];margins=[]
 for ik in range(6,18):
  m,ov,margin=select(state,sn[ik]);chosen.append(m);scores.append(ov);margins.append(margin);state=(sn[ik][0],sn[ik][1][:,m])
 cm,co,ca=select(state,sn[6]);close=(cm==chosen[0]);tracks[seed]=chosen
 # pointwise direct anchoring from Gamma is ambiguous for degenerate pairs: compare as a diagnostic.
 agrees=[]
 for j,ik in enumerate(range(6,18)):
  m=chosen[j];opp=6+(j+6)%12;ro=get(opp,chosen[(j+6)%12]);r=get(ik,m);k=points[ik];f=g.freqs[ik,m]
  plus=jones(r,k,f);minus=jones(ro,k,f);strict,I=volume(ik,m)
  pred=np.array([r['u_te'],-r['u_tm']]);fit=np.vdot(pred,I)/np.vdot(pred,pred);res=np.linalg.norm(I-fit*pred)/np.linalg.norm(I)
  gm0=gamma_m;direct,_,_=select((sn[0][0],sn[0][1][:,gm0]),sn[ik]);agrees.append(direct==m)
  for method,e in [('same_k_current',plus),('minus_k_current',minus),('same_k_volume',strict),('minus_k_volume',volume(opp,chosen[(j+6)%12])[0])]:
   loop.append(dict(branch_at_q04=seed,phi_deg=j*30,qx=float(k[0]),qy=float(k[1]),mode=m,method=method,S1_S0=stok(e)[0],S2_S0=stok(e)[1],S3_S0=stok(e)[2],psi_deg=stok(e)[3],overlap=scores[j],margin=margins[j],volume_residual=float(res)))
 # independent single point negative-k pairing and quadrature check
 mminus,mo,mm=select((sn[4][0],sn[4][1][:,seed]),sn[5]);rr=get(4,seed);rm=get(5,mminus)
 E24,I24=volume(4,seed,24);E48,I48=volume(4,seed,48)
 ones=[]
 for method,e in [('same_k_current',jones(rr,points[4],g.freqs[4,seed])),('minus_k_current',jones(rm,points[4],g.freqs[4,seed])),('same_k_volume_24',E24),('same_k_volume_48',E48),('minus_k_volume',volume(5,mminus)[0])]:
  ones.append(dict(branch_at_q04=seed,method=method,**dict(zip(['S1_S0','S2_S0','S3_S0','psi_deg'],stok(e)))))
 summary[str(seed)]=dict(gamma_mode=gamma_m,gamma_symmetry=symgamma(gamma_m),min_loop_overlap=min(scores),min_loop_margin=min(margins),loop_modes=chosen,closed_mode=close,closure_overlap=co,gamma_direct_agreement=agrees,single_point=ones)
 for method in ['same_k_current','minus_k_current','same_k_volume','minus_k_volume']:
  rows=[r for r in loop if r['branch_at_q04']==seed and r['method']==method]
  z=np.array([r['S1_S0']+1j*r['S2_S0'] for r in rows]);diff=np.angle(np.roll(z,-1)/z)
  summary[str(seed)][method+'_charge']=float(diff.sum()/(4*np.pi))
  summary[str(seed)][method+'_max_double_axis_step_deg']=float(np.max(abs(np.degrees(diff))))
 print('branch',seed,'gamma',gamma_m,'charge',summary[str(seed)]['same_k_current_charge'],'mirror',summary[str(seed)]['gamma_symmetry']['mirror_y'],flush=True)
writecsv('data/polarization_loop12.csv',loop);writecsv('data/radial_candidate_modes.csv',radii)
writecsv('data/single_point_extraction_methods.csv',[r for s in summary.values() for r in s['single_point']])
(ROOT/'validation/candidate_modes.json').write_text(json.dumps(summary,indent=2))
# Each panel contains actual axes from the 12 solved points. No rotation or offset.
fig,axs=plt.subplots(4,4,figsize=(13,12),sharex=True,sharey=True)
methods=['same_k_current','minus_k_current','same_k_volume','minus_k_volume']
for row,seed in enumerate([1,2,3,4]):
 for col,method in enumerate(methods):
  ax=axs[row,col];rows=[r for r in loop if r['branch_at_q04']==seed and r['method']==method]
  ps=np.radians([r['psi_deg'] for r in rows]);v=.0035*np.c_[np.cos(ps),np.sin(ps)];k=points[6:18]
  ax.add_collection(LineCollection(np.stack((k-v,k+v),axis=1),color='navy',linewidth=1.7));ax.scatter(k[:,0],k[:,1],s=6,c='gray');ax.plot(0,0,'k.',ms=4)
  ax.set_aspect('equal');ax.set_xlim(-.027,.027);ax.set_ylim(-.027,.027)
  ax.set_title(f'm(q=.04)={seed}; '+method.replace('_',' ')+'\ncharge='+str(round(summary[str(seed)][method+'_charge'],3)),fontsize=8)
  if col==0:ax.set_ylabel('qy')
  if row==3:ax.set_xlabel('qx')
fig.suptitle('ACS: independently solved 12-point q=0.02 loops; no angle correction',fontsize=12)
fig.tight_layout();fig.savefig(ROOT/'figures/polarization_loop12_comparison.png',dpi=200);plt.close(fig)
# Gamma electric field near-field plots to document mirror/rotation character.
fig,axs=plt.subplots(2,2,figsize=(9,8))
for ax,seed in zip(axs.ravel(),[1,2,3,4]):
 m=summary[str(seed)]['gamma_mode'];fields,x,y=g.get_field_xy('e',0,m,h/2,Nx=40,Ny=40,component='xyz')
 ex,ey=fields['x'],fields['y'];phase=np.exp(-1j*np.angle(ex.ravel()[np.argmax(abs(ex))]));mag=abs(ex)**2+abs(ey)**2
 ax.imshow(mag,origin='lower',extent=[x[0],x[-1],y[0],y[-1]],cmap='magma')
 ax.quiver(x[::4],y[::4],np.real(ex*phase)[::4,::4],np.real(ey*phase)[::4,::4],color='white')
 ax.set_title(f'Gamma m={m}, branch q=.04 m={seed}');ax.set_aspect('equal')
fig.tight_layout();fig.savefig(ROOT/'figures/gamma_nearfields.png',dpi=160)
print('diagnostic outputs saved',flush=True)
