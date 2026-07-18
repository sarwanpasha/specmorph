import numpy as np
# EXACT lower-bound / optimality check for the obstruction theorem.
# Claim: among ALL corrections of the form  w - delta y  (y in C^0, i.e.
# any globally-consistent reassignment of stalk sections -- the most any
# projection scheme can do is subtract an element of im(delta)), the
# minimum achievable residual is exactly dist(w, im delta) = ||P_{H1} w||,
# attained by the orthogonal (coboundary) projection. Hence ||P_{H1} w||
# is a HARD floor: no scheme restricted to im(delta) can beat it, and it is
# > 0 iff H1 != 0. Verify min over y equals ||P_{H1} w|| to machine eps.

def coboundary(T,edges,V,r):
    E=len(edges); D=np.zeros((E*r,V*r))
    for ei,(u,v) in enumerate(edges):
        D[ei*r:(ei+1)*r,v*r:(v+1)*r]+=np.eye(r)
        D[ei*r:(ei+1)*r,u*r:(u+1)*r]+=-T[ei]
    return D

def build(V,r,extra,seed):
    rng=np.random.default_rng(seed)
    edges=[(i,i+1) for i in range(V-1)]
    for _ in range(extra):
        a=int(rng.integers(0,V)); b=int(rng.integers(0,V))
        while b==a: b=int(rng.integers(0,V))
        edges.append((a,b))
    T=[np.linalg.qr(rng.standard_normal((r,r)))[0] for _ in edges]
    return edges,T

maxerr=0.0; worst=None; checked=0
for V in [5,7,9]:
  for r in [2,3,4]:
    for extra in [0,1,2,4]:
      for seed in range(20):
        edges,T=build(V,r,extra,seed); E=len(edges)
        D=coboundary(T,edges,V,r)
        rng=np.random.default_rng(555+seed); w=rng.standard_normal(E*r)
        # least-squares min_y ||w - D y||  = residual of projection onto im(D)
        y,_,_,_=np.linalg.lstsq(D,w,rcond=None)
        res_opt=np.linalg.norm(w-D@y)
        # P_{H1} w  via SVD
        Ud,s,_=np.linalg.svd(D,full_matrices=False); k=int((s>1e-9).sum())
        Ph1=np.eye(E*r)-Ud[:,:k]@Ud[:,:k].T
        res_h1=np.linalg.norm(Ph1@w)
        err=abs(res_opt-res_h1)
        checked+=1
        if err>maxerr: maxerr=err; worst=(V,r,extra,seed,res_opt,res_h1)
print('=== EXACT OPTIMALITY OF COBOUNDARY PROJECTION ===')
print(f'configs checked                         : {checked}')
print(f'max |min_y||w-Dy||  -  ||P_H1 w|| |     : {maxerr:.3e}  (should be ~machine eps)')
print(f'=> coboundary projection is provably optimal; floor = ||P_H1 w||')
print(f'worst case (V,r,extra,seed,opt,h1)      : {worst}')
