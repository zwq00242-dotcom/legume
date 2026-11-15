"""Refine only the outer square boundary; no additional 2-D map."""
from _paths import BASE, REPO, CACHE
import compute_mode2_fullmap as m
import numpy as np,json

def main():
 n=120;h=.08
 points=np.vstack((np.c_[np.linspace(-h,h,n,endpoint=False),np.full(n,-h)],np.c_[np.full(n,h),np.linspace(-h,h,n,endpoint=False)],np.c_[np.linspace(h,-h,n,endpoint=False),np.full(n,h)],np.c_[np.full(n,-h),np.linspace(h,-h,n,endpoint=False)]))
 z=np.load(m.OUT/'data/eigenvectors_mode2.npz');j=int(np.argmin(np.linalg.norm(z['q']-points[0],axis=1)))
 st=dict(q=z['q'][j],keys=z['keys'],v=z['eigenvectors'][j],mode=int(z['mode'][j]),f=float(z['frequency'][j]),gap=.01)
 rows=[];initialmode=st['mode']
 for ib in range(0,len(points),32):
  g=m.solve(points[ib:ib+32])
  for i in range(len(g.freqs)):
   st,d=m.advance(st,g,i);rows.append(m.observables(g,i,st,d))
  print('outer boundary',min(ib+32,len(points)),'/',len(points),flush=True)
 g=m.solve([points[0]]);end,di=m.advance(st,g,0);assert end['mode']==initialmode
 m.dumpcsv('data/outer_boundary480.csv',rows)
 Z=np.array([r['S1']+1j*r['S2'] for r in rows]);result={}
 for stride in [4,2,1]:
  zz=Z[::stride];steps=np.angle(np.roll(zz,-1)/zz)
  result[str(len(zz))]=dict(charge=float(steps.sum()/(4*np.pi)),max_double_angle_step_deg=float(np.max(abs(steps))*180/np.pi),minimum_linear_fraction=float(min(np.hypot(r['S1'],r['S2'])/r['S0'] for r in rows[::stride])))
 result['closure_overlap']=di['overlap'];result['minimum_accepted_overlap']=min(r['overlap'] for r in rows)
 (m.OUT/'validation/outer_boundary_convergence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
