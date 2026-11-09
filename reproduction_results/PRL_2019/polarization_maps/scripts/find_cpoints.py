"""Refine zeros of normalized S1,S2 by fresh eigenmode solves (no interpolation)."""
import json,sys
import numpy as np
from scipy.optimize import root
import compute_farfield as cf
from track_te2 import snapshot,select
from calculate_stokes import extract,stokes
ROOT=cf.ROOT


def evaluate(k,alpha,gmax=4.5):
    cf.ALPHA=alpha
    k=np.asarray(k,dtype=float);g=cf.solve(np.array([[0.,0.],k,-k]),gmax)
    am,_,_=cf.anchor(g);keys,v=snapshot(g,0);state=(keys,v[:,am])
    ms=[select(state,snapshot(g,j))[0] for j in [1,2]]
    r=extract(g,2,ms[1]);p=k/np.linalg.norm(k) if np.linalg.norm(k)>1e-14 else np.array([1.,0.]);s=np.array([p[1],-p[0]])
    co=np.sqrt(1-(np.linalg.norm(k)/g.freqs[1,ms[0]])**2)
    e=r['u_te']*s-r['u_tm']*co*p
    d=stokes(np.array([e[0]]),np.array([e[1]]),np.array([k]))
    return d['normalized_stokes'][0],float(d['S0'][0]),float(g.freqs[1,ms[0]])


def refine(k,alpha,gmax=4.5):
    history=[]
    def fun(x):
        s,i,f=evaluate(x,alpha,gmax);history.append([*x,*s,i,f]);return s[:2]
    sol=root(fun,k,tol=1e-9)
    s,i,f=evaluate(sol.x,alpha,gmax)
    if np.linalg.norm(s[:2])>1e-7 or i<1e-12:raise RuntimeError(f'Unresolved C point {alpha} {sol}')
    return dict(alpha=alpha,gmax=gmax,kxy=sol.x.tolist(),normalized_stokes=s.tolist(),S0=i,frequency=f,residual=float(np.linalg.norm(s[:2])),solver_success=bool(sol.success),evaluations=history)


def main():
    output=[]
    for alpha in [.1,1.]:
        z=np.load(ROOT/'data'/(cf.tag(alpha)+'.npz'))
        for sign in [-1,1]:
            ix=np.flatnonzero(z['valid'] & (np.sign(z['normalized_stokes'][:,2])==sign))
            seed=z['kxy'][ix[np.argmax(abs(z['normalized_stokes'][ix,2]))]]
            for gm in [4.5,5.5,6.5,8.5]:
                r=refine(seed,alpha,gm);output.append(r)
                print('C point',alpha,gm,r['kxy'],r['normalized_stokes'],flush=True)
    for a in [.1,1.]:
        for gm in [4.5,5.5,6.5,8.5]:
            pair=[r for r in output if r['alpha']==a and r['gmax']==gm]
            if len(pair)!=2 or pair[0]['normalized_stokes'][2]*pair[1]['normalized_stokes'][2]>-.99:
                raise RuntimeError('Two distinct opposite-handed C points were not resolved')
    (ROOT/'validation/cpoints.json').write_text(json.dumps(output,indent=2))
if __name__=='__main__':main()
