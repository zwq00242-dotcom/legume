"""Geometry reused verbatim in convention from 01_legume_Fig2_abc_TE_bands.ipynb."""
import numpy as np
import legume
S0=(247/450)**2

def vertices(alpha):
    L=np.sqrt(2*S0/(2-alpha));h=L;top=(1-alpha)*L
    if alpha==1:return np.array([[-L/2,-h/2],[L/2,-h/2],[0,h/2]])
    return np.array([[-L/2,-h/2],[L/2,-h/2],[top/2,h/2],[-top/2,h/2]])

def build(alpha):
    v=vertices(alpha);p=legume.PhotCryst(legume.Lattice('square'))
    p.add_layer(d=100/450,eps_b=2.02**2)
    p.add_shape(legume.Poly(eps=1,x_edges=v[:,0],y_edges=v[:,1]))
    area=.5*abs(np.sum(v[:,0]*np.roll(v[:,1],-1)-v[:,1]*np.roll(v[:,0],-1)))
    assert np.isclose(area,S0,atol=1e-14)
    assert np.min(np.linalg.norm(np.roll(v,-1,axis=0)-v,axis=1))>0
    return p
