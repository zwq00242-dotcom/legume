"""True closed-loop eigenmode computations and independent reciprocity audit."""
import json,csv
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from compute_modes import ROOT,calculate,solve,anchor


def winding(z):
    return float(np.angle(np.roll(z,-1)/z).sum()/(4*np.pi))


def reciprocal_audit(kxy,quadrature=48):
    """Lorentz reciprocity, independently integrated from volume polarization.

    Incoming test at -k is normalized to the *global outgoing electric axes*:
    s=(p_y,-p_x,0), p_out=(cos(theta)*p_x,cos(theta)*p_y,-sin(theta)).
    It is not read out of rad_coup. Compare only up to one common complex factor.
    """
    from extract_farfield import extract
    g=solve([[0,0],kxy]);m0,_,_=anchor(g)
    from track_modes import snapshot,select
    ak,av=snapshot(g,0);m,_,_=select((ak,av[:,m0]),snapshot(g,1))
    r=extract(g,1,m);om=2*np.pi*g.freqs[1,m];gn=2*np.pi*np.linalg.norm(kxy)
    p=kxy/np.linalg.norm(kxy);q=np.array([p[1],-p[0]])
    iz=np.argmin(np.linalg.norm(g.gvec,axis=0));xs,ys,chi=g._get_rad(np.array([gn]),om,'te',1);xp,yp,_=g._get_rad(np.array([gn]),om,'tm',1)
    d=g.d_array[0];t,w=leggauss(quadrature);integrals=np.zeros(2,complex)
    for z,weight in zip((t+1)*d/2,w*d/2):
        D=np.array(g.ft_field_xy('d',1,m,z))[:,iz]
        E=np.array(g.ft_field_xy('e',1,m,z))[:,iz]
        P=D-g.eps_array[1]*E
        ep=np.exp(1j*chi[1,0]*(z-d/2));em=1/ep
        As=ys[1,0]*ep;Bs=xs[1,0]*em;Ap=yp[1,0]*ep;Bp=xp[1,0]*em
        Es=om*(As+Bs)*np.r_[q,0]
        Ep=(-chi[1,0]*(Ap-Bp)*np.r_[p,0]-gn*(Ap+Bp)*np.array([0,0,1]))/(om*g.eps_array[1])
        integrals+=weight*np.array([Es@P,Ep@P])
    predicted=np.array([r['u_te'],-r['u_tm']])
    factor=np.vdot(predicted,integrals)/np.vdot(predicted,predicted)
    residual=np.linalg.norm(integrals-factor*predicted)/np.linalg.norm(integrals)
    return dict(kxy=kxy.tolist(),quadrature=quadrature,relative_residual=float(residual),factor=[factor.real,factor.imag],integrals=[[v.real,v.imag] for v in integrals],couplings=[[v.real,v.imag] for v in predicted])


def gamma_character():
    g=solve([[0,0]]);m,_,_=anchor(g)
    # H_z is an axial scalar under proper in-plane rotation.
    hz=np.asarray(g.ft_field_xy('h',0,m,50/724)[2]);theta=np.pi/3
    R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    rotated=R@g.gvec
    perm=[np.argmin(np.linalg.norm(g.gvec-v[:,None],axis=0)) for v in rotated.T]
    assert np.max(np.linalg.norm(g.gvec[:,perm]-rotated,axis=0))<1e-10
    character=np.vdot(hz,hz[perm])/np.vdot(hz,hz)
    residual=np.linalg.norm(hz[perm]-character*hz)/np.linalg.norm(hz)
    return dict(character_C6=[character.real,character.imag],relative_residual=float(residual),frequency=float(g.freqs[0,m]),sorted_band_1based=int(m+1))


def main():
    rows=[]
    cases=[(3.5,(0,3)),(4.5,(0,3)),(5.5,(0,3)),(6.5,(0,3)),(6.5,(0,3,4,7))]
    for gm,basis in cases:
        points=[[0.,0.]];spec=[]
        for radius in [.01,.02,.04,.08,.12]:
            for density in [24,48]:
                start=len(points);t=np.arange(density)*2*np.pi/density
                points.extend((radius*np.column_stack((np.cos(t),np.sin(t)))).tolist())
                spec.append((radius,density,start,len(points)))
        name='loops_g'+str(gm).replace('.','p')+('_expanded' if len(basis)>2 else '')
        d=calculate(points,gm,basis,name=name)
        for radius,density,a,b in spec:
            z=d['S1'][a:b]+1j*d['S2'][a:b]
            q=winding(z)
            # C6 covariance: compare actual points separated by 60 degrees.
            psi=d['psi'][a:b];step=density//6
            err=.5*np.angle(np.exp(2j*(np.roll(psi,-step)-psi-np.pi/3)))
            tr=.5*np.angle(np.exp(2j*(np.roll(psi,-density//2)-psi)))
            qrad=d['Q'][a:b]
            rows.append(dict(gmax=gm,basis=' '.join(map(str,basis)),radius=radius,density=density,charge=q,gamma_frequency=d['frequency'][0],mean_frequency=d['frequency'][a:b].mean(),Q_min=qrad.min(),Q_median=np.median(qrad),Q_max=qrad.max(),max_abs_S3_over_S0=np.max(abs(d['normalized_stokes'][a:b,2])),C6_axis_error_rad=np.max(abs(err)),TR_axis_error_rad=np.max(abs(tr)),min_tracking_overlap=d['overlap'][a:b].min(),min_tracking_margin=d['margin'][a:b].min()))
        print(name,'charges',[r['charge'] for r in rows[-10:]],flush=True)
    with (ROOT/'validation/convergence_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    audits=[reciprocal_audit(r*np.array([np.cos(t),np.sin(t)]),n) for r,t in [(.035,.23),(.06,.37),(.12,1.13)] for n in [24,48]]
    result=dict(gamma=gamma_character(),reciprocity_audits=audits)
    (ROOT/'validation/source_audit.json').write_text(json.dumps(result,indent=2))
    print('source audit',result,flush=True)
if __name__=='__main__':main()
