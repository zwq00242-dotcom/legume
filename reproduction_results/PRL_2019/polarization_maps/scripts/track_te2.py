"""Continue the existing Gamma-anchored branch using labeled GME basis rows."""
import numpy as np


def snapshot(gme, ik):
    keys=[]
    for b,m in enumerate(gme.gmode_include[ik]):
        for gi in np.asarray(gme.ind_modes[ik][b],int):
            keys.append((int(m),int(gme.inds1[gi]),int(gme.inds2[gi])))
    v=np.asarray(gme.eigvecs[ik])
    assert len(keys)==v.shape[0]
    return keys,v


def select(previous, current):
    pk,pv=previous;ck,cv=current
    lookup={k:i for i,k in enumerate(pk)}
    right=[i for i,k in enumerate(ck) if k in lookup]
    left=[lookup[ck[i]] for i in right]
    a=pv[left];b=cv[right]
    overlaps=abs(a.conj()@b)/(np.linalg.norm(a)*np.linalg.norm(b,axis=0))
    order=np.argsort(overlaps)[::-1]
    return int(order[0]),float(overlaps[order[0]]),float(overlaps[order[0]]-overlaps[order[1]])
