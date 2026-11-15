"""Track the diagnosed ACS branch using labeled eigenvector overlaps only.

Reuses the existing ACS solver scripts in this checkout (baseline e205b78).
Writes only to this diagnosis directory; original reproduction results stay unchanged.
No angular rotation, x/y exchange, fixed-index grid selection, or Q selection.
"""
from _paths import BASE, REPO, CACHE
import os
os.environ['MPLBACKEND']='Agg'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']=str(CACHE / 'matplotlib')
from pathlib import Path
import sys,json,csv,time,argparse,hashlib
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
ROOT=BASE
ARCHIVE=REPO
SCRIPTS=ARCHIVE/'reproduction_results/ACS_Photonics_2023/polarization_map_FigS1/scripts'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(SCRIPTS))
from compute_modes import solve,STRUCT
from track_modes import snapshot
from extract_farfield import extract
OUT=ROOT
CACHE.mkdir(parents=True,exist_ok=True)
(CACHE/'mpl').mkdir(exist_ok=True)
START=time.time();EXTRA=[]
MIN_OV=.97;MIN_MARGIN=.55

def state(g,i,m):
 k,v=snapshot(g,i);return dict(q=np.asarray(g.kpoints[:,i])/(2*np.pi),keys=k,v=v[:,m],mode=int(m),f=float(g.freqs[i,m]),gap=float(np.min(abs(np.delete(g.freqs[i],m)-g.freqs[i,m]))))

def overlap(a,keys,V):
 look={tuple(k):i for i,k in enumerate(a['keys'])}
 right=[i for i,k in enumerate(keys) if tuple(k) in look];left=[look[tuple(keys[i])] for i in right]
 u=a['v'][left];v=V[right]
 return abs(u.conj()@v)/(np.linalg.norm(u)*np.linalg.norm(v,axis=0))

def select(a,g,i):
 k,v=snapshot(g,i);ov=overlap(a,k,v);order=np.argsort(ov)[::-1];m=int(order[0]);gap=float(g.freqs[i,m]-g.freqs[i,max(0,m-1)])
 return m,float(ov[m]),float(ov[order[0]]-ov[order[1]])

def advance(a,g,i,depth=0):
 m,o,margin=select(a,g,i)
 q=g.kpoints[:,i]/(2*np.pi)
 angle=abs(np.arctan2(a['q'][0]*q[1]-a['q'][1]*q[0],np.dot(a['q'],q)))
 # High overlap alone can select the other member after a coarse 90-degree
 # step around a Gamma doublet. Bound GEOMETRIC steps before accepting it.
 local_gap=min(a['gap'],float(np.min(abs(np.delete(g.freqs[i],m)-g.freqs[i,m]))))
 # Resolve narrow avoided crossings away from the known Gamma doublet.
 # Threshold is spectral/geometric, independent of the polarization.
 max_dq=1e-4 if min(np.linalg.norm(q),np.linalg.norm(a['q']))>.04 and local_gap<.003 else .01
 small_step=angle<=np.pi/24+1e-12 and np.linalg.norm(q-a['q'])<=max_dq+1e-12
 if o>=MIN_OV and margin>=MIN_MARGIN and small_step:
  s=state(g,i,m);return s,dict(overlap=o,margin=margin,depth=depth,parent_qx=float(a['q'][0]),parent_qy=float(a['q'][1]),reliable=True)
 if depth>=10:raise RuntimeError(f'Unresolved overlap at {q}: {o}, margin {margin}; STOP')
 mid=(a['q']+q)/2
 if np.linalg.norm(mid)<1e-9:raise RuntimeError('Tracking path crossed Gamma degeneracy; STOP')
 gm=solve([mid]);b,di=advance(a,gm,0,depth+1)
 EXTRA.append(dict(qx=float(mid[0]),qy=float(mid[1]),mode=b['mode'],f=b['f'],**di))
 s,di=advance(b,g,i,depth+1)
 di['overlap']=min(o if o>=MIN_OV else 1.,di['overlap'])
 return s,di

def observables(g,i,s,diag):
 r=extract(g,i,s['mode']);x,y=r['Ex'],r['Ey'];s0=abs(x)**2+abs(y)**2
 s1=abs(x)**2-abs(y)**2;s2=2*np.real(x.conjugate()*y);s3=2*np.imag(x.conjugate()*y)
 return dict(qx=float(s['q'][0]),qy=float(s['q'][1]),mode_index=s['mode'],frequency_real=s['f'],frequency_im=r['freq_im'],Q=float(s['f']/(2*r['freq_im'])) if r['freq_im']>0 else float('inf'),Ex_real=x.real,Ex_imag=x.imag,Ey_real=y.real,Ey_imag=y.imag,S0=float(s0),S1=float(s1),S2=float(s2),S3=float(s3),psi=float(.5*np.arctan2(s2,s1)),u_te_real=r['u_te'].real,u_te_imag=r['u_te'].imag,u_tm_real=r['u_tm'].real,u_tm_imag=r['u_tm'].imag,nopen_u=r['nopen_u'],nopen_l=r['nopen_l'],nearest_frequency_gap=float(np.min(abs(np.delete(g.freqs[i],s['mode'])-s['f']))),**diag)

def dumpcsv(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def save_states(name,states):
 keys=sorted(set(tuple(k) for s in states for k in s['keys']));look={k:i for i,k in enumerate(keys)}
 v=np.zeros((len(states),len(keys)),complex)
 for j,s in enumerate(states):v[j,[look[tuple(k)] for k in s['keys']]]=s['v']
 np.savez_compressed(OUT/name,keys=np.asarray(keys),eigenvectors=v,q=np.array([s['q'] for s in states]),mode=np.array([s['mode'] for s in states]),frequency=np.array([s['f'] for s in states]))

def mirror_y(g,i,m):
 h=STRUCT['thickness_over_a'];F=np.array(g.ft_field_xy('e',i,m,h/2));G=g.gvec*np.array([[1],[-1]])
 perm=[np.argmin(np.linalg.norm(g.gvec-v[:,None],axis=0)) for v in G.T]
 b=np.diag([1,-1,1])@F[:,perm];c=np.vdot(F,b)/np.vdot(F,F)
 return dict(character=float(c.real),residual=float(np.linalg.norm(b-c*F)/np.linalg.norm(F)))

def init():
 g=solve([[.04,0],[.02,0],[.01,0],[.005,0],[0,0]])
 # Identity assertion at one already diagnosed physical anchor, not an index rule on grid.
 table=list(csv.DictReader((ROOT/'data/single_point_modes.csv').open()))
 expected=next(r for r in table if r['mode_index']=='2')
 assert abs(g.freqs[0,2]-float(expected['frequency_normalized']))<1e-10
 assert mirror_y(g,0,2)['character']<-.999999
 s2=state(g,0,2);s1=state(g,0,1);seed2=s2;seed1=s1;radial=[]
 for i in [1,2,3]:
  s2,d2=advance(s2,g,i);s1,d1=advance(s1,g,i)
  radial.append(observables(g,i,s2,d2))
  if i==1:circle2=s2;circle1=s1
 # Identify Gamma eigenspace by eigenvalue degeneracy AND rank-two projection.
 k,V=snapshot(g,4);freq=g.freqs[4];cent=np.flatnonzero(abs(freq-.9364949304843935)<1e-8)
 assert len(cent)==2
 look={tuple(x):j for j,x in enumerate(s2['keys'])};rr=[j for j,x in enumerate(k) if tuple(x) in look];ll=[look[tuple(k[j])] for j in rr]
 B=V[np.ix_(rr,cent)];u=s2['v'][ll];coef=np.linalg.lstsq(B,u,rcond=None)[0];projection=float(np.linalg.norm(B@coef)/np.linalg.norm(u))
 coef/=np.linalg.norm(coef);v=V[:,cent]@coef
 # Radiative couplings depend conjugate-linearly on the guided eigenvector.
 fi,fs,fp,coup,gv=g.compute_rad_sp(4,cent.tolist());loss=0
 for key in coup:
  c=np.array(coup[key]);loss+=float(np.sum(abs(coef.conj()@c)**2))
 f=float(np.mean(freq[cent]));fim=float(np.sqrt((2*np.pi*f)**2+1j*loss).imag/(2*np.pi))
 gc=coef.conj()@np.array(coup['u_te']);gp=coef.conj()@np.array(coup['u_tm']);gx=-gp[0];gy=-gc[0]
 gamma_current=extract(g,4,1)
 cx,cy=gamma_current['Ex'],gamma_current['Ey']
 current_center=dict(Ex_real=float(cx.real),Ex_imag=float(cx.imag),Ey_real=float(cy.real),Ey_imag=float(cy.imag),S0=float(abs(cx)**2+abs(cy)**2),S1=float(abs(cx)**2-abs(cy)**2),S2=float(2*np.real(cx.conjugate()*cy)),S3=float(2*np.imag(cx.conjugate()*cy)))
 gamma=dict(current_center=current_center,upper_s=[float(gc[0].real),float(gc[0].imag)],upper_p=[float(gp[0].real),float(gp[0].imag)],upper_S1=float(abs(gx)**2-abs(gy)**2),upper_S2=float(2*np.real(gx.conjugate()*gy)),upper_S3=float(2*np.imag(gx.conjugate()*gy)),all_frequencies=freq.tolist(),upper_Ex=[float(gx.real),float(gx.imag)],upper_Ey=[float(gy.real),float(gy.imag)],upper_S0=float(abs(gx)**2+abs(gy)**2),current_frequency=float(freq[1]),current_fim=gamma_current['freq_im'],current_Q=float(freq[1]/(2*gamma_current['freq_im'])),frequency=f,subspace_modes=cent.tolist(),projection_overlap=projection,frequency_im=fim,Q=f/(2*fim),coupling_squared_sum=loss,coefficients=[[float(c.real),float(c.imag)] for c in coef],mirror_at_anchor=mirror_y(g,0,2))
 np.savez_compressed(OUT/'data/anchor_and_gamma_subspace.npz',anchor_q=seed2['q'],anchor_keys=np.array(seed2['keys']),anchor_eigenvector=seed2['v'],gamma_keys=np.array(k),gamma_subspace=V[:,cent],gamma_projected_eigenvector=v,gamma_coefficients=coef,anchor_frequency=seed2['f'],gamma_frequencies=freq[cent])
 dumpcsv('data/radial_mode2.csv',radial)
 return seed2,seed1,circle2,circle1,gamma

def loopcalc(a2,a1,n=96,radius=.02):
 phi=np.arange(n)*2*np.pi/n;xy=radius*np.c_[np.cos(phi),np.sin(phi)]
 rows=[];states2=[];states1=[];parent2=a2;parent1=a1
 for start in range(0,n,24):
  gg=solve(xy[start:start+24])
  for j in range(gg.freqs.shape[0]):
   s2,d2=advance(parent2,gg,j);s1,d1=advance(parent1,gg,j)
   if s2['mode']==s1['mode']:raise RuntimeError('Branches merged on loop')
   r2=observables(gg,j,s2,d2);r1=observables(gg,j,s1,d1);r2.update(phi_deg=float(phi[start+j]*180/np.pi),current_psi=r1['psi'])
   rows.append(r2);states2.append(s2);states1.append(s1);parent2=s2;parent1=s1
 gg=solve([xy[0]]);close2,dc2=advance(parent2,gg,0);close1,dc1=advance(parent1,gg,0)
 if close2['mode']!=states2[0]['mode'] or close1['mode']!=states1[0]['mode']:raise RuntimeError('Closed-loop branch mismatch')
 z=np.array([r['S1']+1j*r['S2'] for r in rows]);steps=np.angle(np.roll(z,-1)/z);charge=float(steps.sum()/(4*np.pi))
 dumpcsv(f'data/circle_loop{n}.csv',rows)
 print('loop',n,'charge',charge,'max double-angle step deg',np.max(abs(steps))*180/np.pi,flush=True)
 return rows,states2,states1,dict(radius=radius,n=n,charge=charge,max_double_angle_step_deg=float(np.max(abs(steps))*180/np.pi),closure_overlap=dc2['overlap'],min_overlap=min(r['overlap'] for r in rows))

def plotlines(ax,rows,valid,title,extent=.08,length=.0068):
 xy=np.array([[r['qx'],r['qy']] for r in rows]);psi=np.array([r['psi'] for r in rows]);v=.5*length*np.c_[np.cos(psi[valid]),np.sin(psi[valid])]
 ax.add_collection(LineCollection(np.stack((xy[valid]-v,xy[valid]+v),axis=1),colors='#173b64',linewidths=.85))
 ax.plot(0,0,'o',color='gray',ms=4,zorder=4)
 ax.set(xlim=(-extent,extent),ylim=(-extent,extent),aspect='equal',xlabel=r'$q_x=k_xa/(2\pi)$',ylabel=r'$q_y=k_ya/(2\pi)$',title=title)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--loop-only',action='store_true');ap.add_argument('--extent',type=float,default=.08);ap.add_argument('--grid',type=int,default=31);args=ap.parse_args()
 seed2,seed1,a2,a1,gamma=init()
 rowsloop,loop2,loop1,lstat=loopcalc(a2,a1)
 l48=rowsloop[::2];z=np.array([r['S1']+1j*r['S2'] for r in l48]);lstat['charge_48_subsample']=float(np.angle(np.roll(z,-1)/z).sum()/(4*np.pi))
 if args.loop_only:
  (OUT/'validation/circle_loop96.json').write_text(json.dumps(dict(gamma=gamma,loop=lstat),indent=2));return
 x=np.linspace(-args.extent,args.extent,args.grid);xx,yy=np.meshgrid(x,x);xy=np.c_[xx.ravel(),yy.ravel()];N=len(xy);center=np.argmin(np.linalg.norm(xy,axis=1));ids=[int(i) for i in np.argsort(np.linalg.norm(xy,axis=1),kind='stable') if i!=center]
 states2={};states1={};results2={};results1={};candidates={};anchorpool2=loop2+[seed2];anchorpool1=loop1+[seed1];poolids=[-i-1 for i in range(len(anchorpool2))]
 all_f=np.zeros((N,12)); all_f[center]=gamma['all_frequencies']
 for ib in range(0,len(ids),32):
  batch=ids[ib:ib+32];gg=solve(xy[batch])
  for j,i in enumerate(batch):
   all_f[i]=gg.freqs[j];candidates[i]=snapshot(gg,j)
   # Pick nearest geometric parent, never choose parent by resulting polarization.
   pool2=anchorpool2+list(states2.values());pool1=anchorpool1+list(states1.values());pids=poolids+list(states2.keys())
   d=np.linalg.norm(np.array([s['q'] for s in pool2])-xy[i],axis=1);p=int(np.argmin(d))
   s2,d2=advance(pool2[p],gg,j);s1,d1=advance(pool1[p],gg,j)
   if s2['mode']==s1['mode']:raise RuntimeError(f'Branches merged at {xy[i]}')
   r2=observables(gg,j,s2,d2);r1=observables(gg,j,s1,d1)
   r2['parent_id']=pids[p];r1['parent_id']=pids[p]
   states2[i]=s2;states1[i]=s1;results2[i]=r2;results1[i]=r1
  print('grid',min(ib+len(batch),len(ids)),'/',len(ids),'elapsed',round(time.time()-START,1),flush=True)
  # Retain results and selected eigenvectors after each chunk for audit.
  np.savez_compressed(CACHE/'progress.npz',ids=np.array(list(results2)),rows2=np.array([json.dumps(r) for r in results2.values()]),rows1=np.array([json.dumps(r) for r in results1.values()]))
 # Center: physical state lives in degenerate eigenspace, not a unique sorted mode.
 template=next(iter(results2.values()));c={key:np.nan for key in template};c.update(qx=0.,qy=0.,mode_index=-1,frequency_real=gamma['frequency'],frequency_im=gamma['frequency_im'],Q=gamma['Q'],Ex_real=gamma['upper_Ex'][0],Ex_imag=gamma['upper_Ex'][1],Ey_real=gamma['upper_Ey'][0],Ey_imag=gamma['upper_Ey'][1],S0=gamma['upper_S0'],S1=gamma['upper_S1'],S2=gamma['upper_S2'],S3=gamma['upper_S3'],nopen_u=1,nopen_l=1,nearest_frequency_gap=0.,overlap=gamma['projection_overlap'],margin=np.nan,depth=0,parent_qx=.005,parent_qy=0.,parent_id=-999,reliable=True)
 results2[center]=c
 c1=c.copy();c1.update(frequency_real=gamma['current_frequency'],frequency_im=gamma['current_fim'],Q=gamma['current_Q'],mode_index=1,reliable=True);c1.update(gamma['current_center']);results1[center]=c1
 r2=[results2[i] for i in range(N)];r1=[results1[i] for i in range(N)]
 # Edge-consistency audit: nearest cardinal edges should independently pick the same mode.
 audit=[];inconsistent=[]
 for i in ids:
  iy,ix=divmod(i,args.grid)
  for j in ([i+1] if ix+1<args.grid else [])+([i+args.grid] if iy+1<args.grid else []):
   if j==center:continue
   a,b=states2[i],states2[j];ov=float(overlap(a,b['keys'],b['v'][:,None])[0])
   ck,cv=candidates[j];scores=overlap(a,ck,cv);order=np.argsort(scores)[::-1];margin=float(scores[order[0]]-scores[order[1]])
   refined=False
   if min(a['gap'],b['gap'])<.003 or ov<MIN_OV or margin<MIN_MARGIN or int(order[0])!=b['mode'] or abs(np.arctan2(a['q'][0]*b['q'][1]-a['q'][1]*b['q'][0],np.dot(a['q'],b['q'])))>np.pi/24:
    refined=True;check=solve([xy[j]]);verified,dd=advance(a,check,0)
    if verified['mode']!=b['mode']:raise RuntimeError(f'Path-dependent branch identity on grid edge {i}->{j}; STOP')
   audit.append(dict(i=i,j=j,overlap=ov,refined_edge=refined,endpoint_consistent=True,delta_frequency=float(abs(a['f']-b['f'])),delta_log10_Q=float(abs(np.log10(results2[i]['Q'])-np.log10(results2[j]['Q'])))))
 # A dense, independently linked angular loop plus adaptive radial paths prevent the Gamma-anchor ambiguity.
 save_states('data/eigenvectors_mode2.npz',[states2[i] for i in ids]);save_states('data/eigenvectors_current.npz',[states1[i] for i in ids])
 dumpcsv('data/neighbor_continuity.csv',audit)
 s02=np.array([r['S0'] for r in r2]);s01=np.array([r['S0'] for r in r1]);floor2=max(np.nanmax(s02)*1e-12,1e-28);floor1=max(np.nanmax(s01)*1e-12,1e-28)
 valid2=(s02>floor2)&np.array([r['reliable'] for r in r2],bool);valid1=(s01>floor1)&np.array([r['reliable'] for r in r1],bool);valid2[center]=valid1[center]=False
 psi2=np.array([r['psi'] for r in r2]);psi1=np.array([r['psi'] for r in r1]);delta=(psi2-psi1+np.pi/2)%np.pi-np.pi/2;err=90-abs(np.degrees(delta));valid=valid1&valid2
 for i in range(N):
  for rows,val in [(r2,valid2),(r1,valid1)]:
   rows[i]['polarization_valid']=bool(val[i])
   if not val[i]:rows[i]['psi']=np.nan
  r2[i]['delta_psi_rad']=float(delta[i]) if valid[i] else np.nan;r2[i]['orthogonal_error_deg']=float(err[i]) if valid[i] else np.nan
 dumpcsv('data/fullgrid_mode2.csv',r2);dumpcsv('data/fullgrid_current.csv',r1)
 if EXTRA:dumpcsv('data/adaptive_tracking.csv',EXTRA)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.linewidth':.8,'savefig.facecolor':'white'})
 fig,ax=plt.subplots(figsize=(5.3,5.0));plotlines(ax,r2,valid2,'',args.extent);ax.text(.025,.965,'(a)',transform=ax.transAxes,va='top',fontsize=13);fig.tight_layout();fig.savefig(OUT/'figures/mode2_polarization.png',dpi=300);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(15.9,5.0));plotlines(axes[0],r1,valid1,'Current mode (raw)',args.extent);plotlines(axes[1],r2,valid2,'Candidate mode 2 (raw)',args.extent)
 step=x[1]-x[0];im=axes[2].imshow(np.where(valid,np.degrees(delta),np.nan).reshape(args.grid,args.grid),origin='lower',extent=[x[0]-step/2,x[-1]+step/2,x[0]-step/2,x[-1]+step/2],vmin=-90,vmax=90,cmap='coolwarm');axes[2].set(xlim=(-args.extent,args.extent),ylim=(-args.extent,args.extent),aspect='equal',xlabel=r'$q_x=k_xa/(2\pi)$',ylabel=r'$q_y=k_ya/(2\pi)$',title=r'Wrapped $\Delta\psi$ (deg)');axes[2].plot(0,0,'o',color='gray',ms=4);fig.colorbar(im,ax=axes[2],shrink=.8);fig.tight_layout();fig.savefig(OUT/'figures/current_vs_mode2.png',dpi=250);plt.close(fig)
 summary=dict(source_commit='e205b78e14bc0b2aed9ca069d0a96162a1a75aed',grid=args.grid,extent=args.extent,original_extent=.15,solver=dict(gmax=4.5,basis=[0,3],numeig=12),structure=STRUCT,gamma=gamma,loop=lstat,valid_points=int(sum(valid)),masked_points=int(N-sum(valid)),intensity_floor2=floor2,intensity_floor1=floor1,orthogonal_error_deg=dict(median=float(np.median(err[valid])),p90=float(np.percentile(err[valid],90)),maximum=float(np.max(err[valid]))),fraction_within_1deg=float(np.mean(err[valid]<1)),fraction_within_5deg=float(np.mean(err[valid]<5)),minimum_accepted_overlap=float(min(r['overlap'] for r in r2 if r['reliable'])),minimum_margin=float(np.nanmin([r['margin'] for r in r2])),adaptive_points=len(EXTRA),neighbor_edges_checked=len(audit),neighbor_edges_refined=sum(r['refined_edge'] for r in audit),path_inconsistencies=0,mode_indices=sorted(set(r['mode_index'] for r in r2)),maximum_neighbor_frequency_step=max(r['delta_frequency'] for r in audit),maximum_neighbor_log10_Q_step=max(r['delta_log10_Q'] for r in audit),minimum_neighbor_overlap=min(r['overlap'] for r in audit),elapsed_seconds=time.time()-START)
 square=[]
 mid=args.grid//2
 for rad in [3,6,9,12,15]:
  if rad>mid:continue
  lo,hi=mid-rad,mid+rad
  path=([lo*args.grid+ix for ix in range(lo,hi)]+[iy*args.grid+hi for iy in range(lo,hi)]+[hi*args.grid+ix for ix in range(hi,lo,-1)]+[iy*args.grid+lo for iy in range(hi,lo,-1)])
  zz=np.array([r2[i]['S1']+1j*r2[i]['S2'] for i in path]);steps=np.angle(np.roll(zz,-1)/zz)
  square.append(dict(halfwidth=float(x[hi]),points=len(path),charge=float(steps.sum()/(4*np.pi)),max_double_angle_step_deg=float(np.max(abs(steps))*180/np.pi),all_valid=bool(np.all(valid2[path]))))
 summary['grid_square_loops']=square
 fig,axs=plt.subplots(2,2,figsize=(10,9))
 for ax,(values,title) in zip(axs.ravel(),[(np.array([r['frequency_real'] for r in r2]),'Normalized frequency'),(np.log10([r['Q'] for r in r2]),'log10 Q (Gamma masked)'),(np.array([r['overlap'] for r in r2]),'Accepted predecessor overlap'),(np.array([r['nearest_frequency_gap'] for r in r2]),'Nearest eigenfrequency gap')]):
  values=np.asarray(values).copy();values[center]=np.nan
  im=ax.imshow(values.reshape(args.grid,args.grid),origin='lower',extent=[-args.extent,args.extent,-args.extent,args.extent]);ax.set(title=title,xlabel='qx',ylabel='qy',aspect='equal');fig.colorbar(im,ax=ax)
 fig.tight_layout();fig.savefig(OUT/'figures/tracking_diagnostics.png',dpi=180);plt.close(fig)
 np.savez_compressed(OUT/'data/eigenfrequencies_fullgrid.npz',q=xy,frequencies=all_f)
 (OUT/'validation/fullgrid_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
