"""Independent real-k eigenmode/radiation calculations, with chunk checkpoints."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/workspace/setup-legume/matplotlib')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,time,sys,platform,subprocess
from pathlib import Path
import numpy as np
import legume
from track_modes import snapshot,select
from extract_farfield import extract,stokes

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'band_structure/highres/metadata.json'
STRUCT=json.loads(OLD.read_text())['structure']


def build():
    # Reuse the archived Fig 1(a) geometry and exact lattice convention.
    l=legume.Lattice(STRUCT['lattice'])
    p=legume.PhotCryst(l,eps_l=1,eps_u=1)
    p.add_layer(d=STRUCT['thickness_over_a'],eps_b=STRUCT['slab_eps'])
    p.add_shape(legume.Circle(eps=1,r=STRUCT['radius_over_a']))
    return l,p


def solve(kxy,gmax=4.5,basis=(0,3)):
    l,p=build();g=legume.GuidedModeExp(p,gmax=gmax,truncate_g='abs')
    g.run(kpoints=2*np.pi*np.asarray(kxy).T,gmode_inds=list(basis),numeig=12,
          compute_im=False,verbose=False,kz_symmetry=None)
    return g


def anchor(g):
    expected=json.loads(OLD.read_text())['high_symmetry']['target_normalized_frequencies']['Gamma']
    base=solve([[0,0]],4.5)
    mb=int(np.argmin(abs(base.freqs[0]-expected)))
    if abs(base.freqs[0,mb]-expected)>1e-8: raise RuntimeError('Archived Gamma anchor mismatch')
    keys,vec=snapshot(base,0)
    m,score,margin=select((keys,vec[:,mb]),snapshot(g,0))
    return m,score,margin


def calculate(kxy,gmax=4.5,basis=(0,3),name='grid31',chunk=48):
    kxy=np.asarray(kxy);n=len(kxy);dest=ROOT/'data'/name;dest.mkdir(parents=True,exist_ok=True)
    # Calculation order is radial, output order stays Cartesian input order.
    import hashlib
    spec=dict(gmax=gmax,basis=list(basis),chunk=chunk,kxy_sha256=hashlib.sha256(kxy.tobytes()).hexdigest(),n=n)
    sp=dest/'calculation_spec.json'
    if sp.exists() and json.loads(sp.read_text())!=spec:
        raise RuntimeError('Checkpoint parameters differ: choose a new output name')
    sp.write_text(json.dumps(spec,indent=2))
    order=np.argsort(np.linalg.norm(kxy,axis=1),kind='stable')
    states={};results={};frequencies={};vectors={};keys_all={};start=time.time()
    for ib in range(0,n,chunk):
        ids=order[ib:ib+chunk];checkpoint=dest/f'chunk_{ib//chunk:03d}.npz'
        if checkpoint.exists():
            z=np.load(checkpoint,allow_pickle=False)
            for j,i in enumerate(z['ids']):
                keys=[tuple(t) for t in z[f'keys_{j}']];v=z[f'vec_{j}']
                states[int(i)]=(keys,v);results[int(i)]=dict(zip(z['fields'],z['values'][j]))
                frequencies[int(i)]=z['frequencies'][j];vectors[int(i)]=v;keys_all[int(i)]=keys
            print(name,'resumed',len(states),'/',n,flush=True);continue
        g=solve(np.vstack(([0,0],kxy[ids])),gmax,basis)
        am,_,_=anchor(g);ak,av=snapshot(g,0);astate=(ak,av[:,am])
        chunk_res=[]
        for j,i in enumerate(ids):
            cur=snapshot(g,j+1)
            if np.linalg.norm(kxy[i])<1e-12:
                m,ov,margin=select(astate,cur);parent=-1
            elif states:
                eligible=np.array(list(states))
                parent=int(eligible[np.argmin(np.linalg.norm(kxy[eligible]-kxy[i],axis=1))])
                m,ov,margin=select(states[parent],cur)
            else:
                parent=-1;m,ov,margin=select(astate,cur)
            keys,v=cur;states[int(i)]=(keys,v[:,m]);vectors[int(i)]=v[:,m];keys_all[int(i)]=keys
            r=extract(g,j+1,m)
            # Channels are serialized explicitly; the present target must be zero order only.
            if r['nopen_u']!=1 or r['nopen_l']!=1: raise RuntimeError('Extra diffraction channel: extend storage/interpretation')
            r.pop('channels_u');r.pop('channels_l')
            gap=np.min(abs(np.delete(g.freqs[j+1],m)-g.freqs[j+1,m]))
            r.update(frequency=float(g.freqs[j+1,m]),mode=m,overlap=ov,margin=margin,parent=parent,gap=gap)
            results[int(i)]=r;frequencies[int(i)]=g.freqs[j+1];chunk_res.append(r)
        fields=list(chunk_res[0]);payload=dict(ids=ids,fields=np.array(fields),values=np.array([[r[f] for f in fields] for r in chunk_res]),frequencies=np.array([frequencies[int(i)] for i in ids]))
        for j,i in enumerate(ids):payload[f'keys_{j}']=np.array(keys_all[int(i)]);payload[f'vec_{j}']=vectors[int(i)]
        tmp=checkpoint.with_suffix('.tmp.npz');np.savez_compressed(tmp,**payload);tmp.replace(checkpoint)
        print(name,'computed',len(states),'/',n,'elapsed',round(time.time()-start,1),flush=True)
    fields=list(results[0]);out={f:np.array([results[i][f] for i in range(n)]) for f in fields}
    for f in ['frequency','freq_im','freq_im_s','freq_im_p','overlap','margin','gap','loss_sum']:out[f]=out[f].real
    for f in ['mode','parent','nopen_u','nopen_l']:out[f]=out[f].real.astype(int)
    out.update(kxy=kxy,all_frequencies=np.array([frequencies[i] for i in range(n)]),rad_gvec_u=np.zeros((n,1,2)),rad_gvec_l=np.zeros((n,1,2)),gmax=gmax,basis=np.array(basis))
    # Reconstruct from raw channels even when resuming an older checkpoint.
    kk=2*np.pi*kxy+np.array([1e-15,0]);pn=kk/np.linalg.norm(kk,axis=1)[:,None]
    sn=np.column_stack((pn[:,1],-pn[:,0]))
    ct=np.sqrt(1-np.sum(kk**2,axis=1)/(2*np.pi*out['frequency'])**2)
    ee=out['u_te'][:,None]*sn-out['u_tm'][:,None]*ct[:,None]*pn
    out['Ex'],out['Ey']=ee.T
    out.update(stokes(out['Ex'],out['Ey'],kxy))
    out['Q']=np.divide(out['frequency'],2*abs(out['freq_im']),out=np.full(n,np.inf),where=out['freq_im']!=0)
    # Compact selected eigenvectors in a common labeled basis for independent audit.
    keys=sorted(set(k for ks in keys_all.values() for k in ks));lookup={k:i for i,k in enumerate(keys)}
    v=np.zeros((n,len(keys)),complex)
    for i in range(n):v[i,[lookup[k] for k in keys_all[i]]]=vectors[i]
    out.update(selected_eigenvectors=v,basis_keys=np.array(keys))
    np.savez_compressed(ROOT/'data'/f'{name}.npz',**out)
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--grid',type=int,default=31);ap.add_argument('--gmax',type=float,default=4.5);a=ap.parse_args()
    x=np.linspace(-.15,.15,a.grid);xx,yy=np.meshgrid(x,x)
    d=calculate(np.column_stack((xx.ravel(),yy.ravel())),a.gmax,name=f'grid{a.grid}')
    l,_=build()
    import importlib.metadata as im
    meta=dict(structure=STRUCT,legume=legume.__version__,python=sys.version,platform=platform.platform(),packages={p:im.version(p) for p in ['numpy','scipy','matplotlib','autograd','pytest']},solver=dict(gmax=a.gmax,gmode_inds=[0,3],numeig=12,truncate_g='abs',kz_symmetry=None,gmode_compute='exact',compute_im=False,radiation='compute_rad_sp for selected target only'),grid=dict(n=a.grid,range=[-.15,.15],units='2pi/a',all_points_independently_solved=True),lattice=dict(a1=l.a1.tolist(),a2=l.a2.tolist(),b1=l.b1.tolist(),b2=l.b2.tolist()),anchor_source=str(OLD.relative_to(ROOT.parents[2])),source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),amplitude_units='Legume loss-normalized Jones amplitudes; common dimensional E factor omitted',mask_relative_floor=1e-12,checkpoint_chunks=48)
    (ROOT/'data/calculation_metadata.json').write_text(json.dumps(meta,indent=2))
    print('complete',a.grid,'valid',sum(d['valid']),'Qgamma',d['Q'][np.argmin(np.linalg.norm(d['kxy'],axis=1))],flush=True)
if __name__=='__main__':main()
