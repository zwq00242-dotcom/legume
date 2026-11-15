"""Verify the archived 31x31, extent=0.08 mode-2 results."""
from _paths import BASE, REPO, CACHE
from pathlib import Path
import csv,json,subprocess,hashlib
import numpy as np
R=BASE

def read(name):return list(csv.DictReader((R/name).open()))

def num(rows,key):return np.array([float(r[key]) for r in rows])

def main():
 s=json.loads((R/'validation/fullgrid_summary.json').read_text());a=read('data/fullgrid_mode2.csv');b=read('data/fullgrid_current.csv')
 assert len(a)==len(b)==31*31
 v=np.array([r['polarization_valid']=='True' for r in a]);v1=np.array([r['polarization_valid']=='True' for r in b]);valid=v&v1
 x=num(a,'Ex_real')+1j*num(a,'Ex_imag');y=num(a,'Ey_real')+1j*num(a,'Ey_imag')
 for field,z in [('S0',abs(x)**2+abs(y)**2),('S1',abs(x)**2-abs(y)**2),('S2',2*np.real(x.conj()*y)),('S3',2*np.imag(x.conj()*y))]:
  scale=np.maximum(abs(x)**2+abs(y)**2,1e-28)
  np.testing.assert_allclose(num(a,field)/scale,z/scale,rtol=0,atol=1e-13)
 psi=.5*np.arctan2(num(a,'S2'),num(a,'S1'));np.testing.assert_allclose(num(a,'psi')[v],psi[v],atol=1e-14)
 assert np.all([r['reliable']=='True' for r in a]);assert s['path_inconsistencies']==0
 assert sum(~v)>=1 and not v[480]
 assert all(r['endpoint_consistent']=='True' for r in read('data/neighbor_continuity.csv'))
 err=num(a,'orthogonal_error_deg')[valid]
 assert abs(np.median(err)-s['orthogonal_error_deg']['median'])<1e-10
 # Compare the current branch against pre-existing raw data at common grid points.
 archive=REPO
 old=np.load(archive/'reproduction_results/ACS_Photonics_2023/polarization_map_FigS1/data/grid31.npz')
 shared=[]
 for r in b:
  q=np.array([float(r['qx']),float(r['qy'])]);j=np.argmin(np.linalg.norm(old['kxy']-q,axis=1))
  if np.linalg.norm(old['kxy'][j]-q)<1e-10 and np.linalg.norm(q)>1e-10:
   d=.5*np.angle(np.exp(2j*(float(r['psi'])-old['psi'][j])))
   shared.append(dict(q=q.tolist(),frequency_difference=float(float(r['frequency_real'])-old['frequency'][j]),axis_difference_deg=float(d*180/np.pi)))
 assert shared and max(abs(r['frequency_difference']) for r in shared)<1e-10
 assert max(abs(r['axis_difference_deg']) for r in shared)<1e-6
 verify=dict(max_abs_S3_over_S0=float(max(abs(num(a,'S3')[v]/num(a,'S0')[v]))),minimum_linear_fraction=float(min(np.hypot(num(a,'S1')[v],num(a,'S2')[v])/num(a,'S0')[v])),rows=len(a),valid=int(sum(v)),raw_jones_stokes_consistent=True,psi_from_raw_stokes=True,neighbor_path_consistency=True,shared_current_branch_checks=shared)
 boundary=json.loads((R/'validation/outer_boundary_convergence.json').read_text())
 local=json.loads((R/'validation/local_continuity65.json').read_text())
 assert all(abs(boundary[str(n)]['charge'])<1e-10 for n in [120,240,480])
 verify['outer_boundary_convergence']=boundary;verify['local_continuity']=local
 (R/'validation/output_checks.json').write_text(json.dumps(verify,indent=2))
 print(json.dumps(verify,indent=2));print('Independent output checks passed.')

if __name__=='__main__':main()
