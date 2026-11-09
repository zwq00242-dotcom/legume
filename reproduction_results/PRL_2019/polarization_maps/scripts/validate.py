"""Geometry, BIC/C-point windings and reciprocity audits."""
import json
import numpy as np
from numpy.polynomial.legendre import leggauss
import compute_farfield as cf
from geometry import vertices,S0
from track_te2 import snapshot,select
from calculate_stokes import extract
ROOT=cf.ROOT


def audit(alpha,k,n=32):
    cf.ALPHA=alpha;k=np.array(k);g=cf.solve([[0,0],k,-k]);am,_,_=cf.anchor(g)
    keys,v=snapshot(g,0);state=(keys,v[:,am]);m=select(state,snapshot(g,1))[0];mm=select(state,snapshot(g,2))[0]
    r=extract(g,2,mm);om=2*np.pi*g.freqs[1,m];gn=2*np.pi*np.linalg.norm(k);p=k/np.linalg.norm(k);q=np.array([p[1],-p[0]])
    iz=np.argmin(np.linalg.norm(g.gvec,axis=0));xs,ys,chi=g._get_rad(np.array([gn]),om,'te',1);xp,yp,_=g._get_rad(np.array([gn]),om,'tm',1)
    t,w=leggauss(n);d=100/450;ints=np.zeros(2,complex)
    for z,wt in zip((t+1)*d/2,w*d/2):
        D=np.array(g.ft_field_xy('d',1,m,z))[:,iz];E=np.array(g.ft_field_xy('e',1,m,z))[:,iz];P=D-g.eps_array[1]*E
        ep=np.exp(1j*chi[1,0]*(z-d/2));As=ys[1,0]*ep;Bs=xs[1,0]/ep;Ap=yp[1,0]*ep;Bp=xp[1,0]/ep
        Es=om*(As+Bs)*np.r_[q,0];Ep=(-chi[1,0]*(Ap-Bp)*np.r_[p,0]-gn*(Ap+Bp)*np.array([0,0,1]))/(om*g.eps_array[1])
        ints+=wt*np.array([Es@P,Ep@P])
    pred=np.array([r['u_te'],-r['u_tm']]);scale=np.vdot(pred,ints)/np.vdot(pred,pred)
    return dict(alpha=alpha,kxy=k.tolist(),quadrature=n,relative_residual=float(np.linalg.norm(ints-scale*pred)/np.linalg.norm(ints)),integrals=[[x.real,x.imag] for x in ints],couplings_minus_k=[[x.real,x.imag] for x in pred])


def wind(z):return float(np.angle(np.roll(z,-1)/z).sum()/(4*np.pi))

def main():
    geom=[]
    for a in [0,.1,1]:
        v=vertices(a);cross=v[:,0]*np.roll(v[:,1],-1)-np.roll(v[:,0],-1)*v[:,1];area=.5*cross.sum();cent=np.sum((v+np.roll(v,-1,axis=0))*cross[:,None],axis=0)/(6*area)
        geom.append(dict(alpha=a,vertices_over_a=v.tolist(),area_over_a2=area,area_nm2=area*450**2,centroid_over_a=cent.tolist(),bounding_box_center=[0,0]))
        assert abs(area-S0)<1e-14
    audits=[audit(a,k,n) for a in [0,.1,1] for k in [[.025,.013],[-.017,.031]] for n in [24,48]]
    assert max(x['relative_residual'] for x in audits)<1e-7
    loops=[]
    cf.ALPHA=0.
    for gm in [4.5,6.5]:
        pts=[];spec=[]
        for radius in [.01,.03,.06]:
            for density in [32,64]:
                t=np.arange(density)*2*np.pi/density;start=len(pts);pts.extend(radius*np.column_stack((np.cos(t),np.sin(t))));spec.append((radius,density,start,len(pts)))
        d=cf.calculate(pts,gm,name=f'bic_loops_g{gm}')
        for r,n,a,b in spec:loops.append(dict(alpha=0,gmax=gm,radius=r,density=n,charge=wind(d['S1'][a:b]+1j*d['S2'][a:b]),Q_median=float(np.median(d['Q'][a:b]))))
    cp=json.loads((ROOT/'validation/cpoints.json').read_text())
    for alpha in [.1,1.]:
        cf.ALPHA=alpha;centers=[np.array(c['kxy']) for c in cp if c['alpha']==alpha and c['gmax']==4.5];pts=[];spec=[]
        for center in centers:
            for radius in [.0004,.0008]:
                t=np.arange(48)*2*np.pi/48;k=center+radius*np.column_stack((np.cos(t),np.sin(t)));start=len(pts);pts.extend(k);spec.append((center,radius,start,len(pts)))
        # Include exact reciprocal partners for every loop point.
        pts=np.array(pts);pts=np.vstack((pts,-pts));d=cf.calculate(pts,4.5,name=f'{cf.tag(alpha)}_cpoint_loops')
        for c,r,a,b in spec:loops.append(dict(alpha=alpha,gmax=4.5,center=c.tolist(),radius=r,density=48,charge=wind(d['S1'][a:b]+1j*d['S2'][a:b])))
    (ROOT/'validation/checks.json').write_text(json.dumps(dict(geometry=geom,reciprocity=audits,loops=loops),indent=2))
    print('validated',loops,flush=True)
if __name__=='__main__':main()
