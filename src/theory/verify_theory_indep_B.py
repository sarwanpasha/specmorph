import numpy as np, json
rng = np.random.default_rng(778899)
res = {}

def opnorm(M): return np.linalg.norm(M, 2)

# ---------- THM 2: coherence bound  ||Pperp B Pread|| <= m R_max mu, B=sum Rj^T Rj ----------
f=0; n=0; tight_wins=0
for t in range(4000):
    N=int(rng.integers(10,40)); k=int(rng.integers(1,max(2,N//3))); m=int(rng.integers(1,8))
    Uread,_=np.linalg.qr(rng.standard_normal((N,k))); Uread=Uread[:,:k]
    Pread=Uread@Uread.T; Pperp=np.eye(N)-Pread
    de=int(rng.integers(1,5))
    Rs=[rng.standard_normal((de,N))*rng.uniform(0.1,2.0) for _ in range(m)]
    B=sum(R.T@R for R in Rs)
    coup=opnorm(Pperp@B@Pread)
    Rmax=max(opnorm(R) for R in Rs)
    mu=max(opnorm(R@Uread) for R in Rs)
    bound=m*Rmax*mu
    n+=1
    if coup>bound+1e-8: f+=1
    # strictly-tighter claim: when mu << ||B||/(m Rmax), m Rmax mu < ||B||
    if m*Rmax*mu < opnorm(B)-1e-9: tight_wins+=1
res['THM2_coherence']=(f,n,tight_wins)

# ---------- THM 3: Gram bound ||Beff|| <= sqrt(||Ga||) sqrt(||Gb||) + topology separation ----------
def gram_trial(N=60,k=5,m=6,mode='disjoint'):
    U,_=np.linalg.qr(rng.standard_normal((N,k))); Uread=U[:,:k]; Pread=Uread@Uread.T
    Vs,_=np.linalg.qr(rng.standard_normal((N,k))); Pperp=np.eye(N)-Vs@Vs.T
    rs=[]; base=rng.standard_normal(N); base/=np.linalg.norm(base)
    for j in range(m):
        r = base+0.03*rng.standard_normal(N) if mode=='aligned' else rng.standard_normal(N)
        r/=np.linalg.norm(r); rs.append(r)
    B=sum(np.outer(r,r) for r in rs)
    Beff=Pperp@B@Pread; nB=opnorm(Beff)
    a=np.array([Pperp@r for r in rs]); b=np.array([Pread@r for r in rs])
    Ga=a@a.T; Gb=b@b.T
    bound=np.sqrt(opnorm(Ga))*np.sqrt(opnorm(Gb))
    mus=[opnorm(np.outer(Pperp@r,Pread@r)) for r in rs]
    return nB,bound,max(mus)
for mode in ['disjoint','aligned']:
    V=np.array([gram_trial(mode=mode) for _ in range(500)])
    holds=(V[:,0]<=V[:,1]+1e-9).mean()*100
    res[f'THM3_gram_{mode}']=(float(V[:,0].mean()),float(V[:,1].mean()),float(V[:,2].mean()),float(holds))

# ---------- PROP tightness: two-sided  ||Beff||/(2g) <= sinTheta <= 2||Beff||/g ----------
fL=0; fU=0; n=0
for t in range(5000):
    gamma=float(rng.uniform(0.5,4.0))
    # exact 2x2 construction from the proof
    alpha=rng.uniform(0.05,0.99); beta=np.sqrt(1-alpha**2)
    tmax=gamma/2; tscale=rng.uniform(0.05,1.0)*tmax
    # L0 = diag(0, gamma), B = t (alpha e0 + beta e1)(...)^T
    v=np.array([alpha,beta]); B=tscale*np.outer(v,v)
    L0=np.diag([0.0,gamma]); L=L0+B
    w,Vv=np.linalg.eigh(L); idx=np.argsort(w)
    u_new=Vv[:,idx[0]]  # read-out = lowest eig
    u_old=np.array([1.0,0.0])
    st=abs(np.sqrt(max(0,1-(u_old@u_new)**2)))
    Beff=abs(tscale*alpha*beta)  # ||Pperp B Pread|| for this 2x2
    lo=Beff/(2*gamma); hi=2*Beff/gamma
    n+=1
    if st < lo-1e-8: fL+=1
    if st > hi+1e-8: fU+=1
res['PROP_tightness_twosided']=(fL,fU,n)

# ---------- COR operational: projector-distance identity ||Pread-Pnew||=||sinTheta|| ----------
fid=0; n=0
for t in range(3000):
    N=int(rng.integers(6,30)); k=int(rng.integers(1,max(2,N//3)))
    A,_=np.linalg.qr(rng.standard_normal((N,N))); Ua=A[:,:k]
    Bq,_=np.linalg.qr(rng.standard_normal((N,N))); Ub=Bq[:,:k]
    Pa=Ua@Ua.T; Pb=Ub@Ub.T
    pd=opnorm(Pa-Pb)
    M=(np.eye(N)-Pa)@Ub; st=np.linalg.svd(M,compute_uv=False).max()
    n+=1
    if abs(pd-st)>1e-8: fid+=1
res['COR_op_projdist_identity']=(fid,n)

open('RESULT_theory_indep_partB.txt','w').write(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
