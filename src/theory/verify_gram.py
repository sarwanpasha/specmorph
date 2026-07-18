import numpy as np
rng=np.random.default_rng(0)
# Verify topology-dependent coherence: ||B_eff|| where B_eff = P_perp (sum_j R_j^T R_j) P_read
# Claim: ||B_eff|| <= ||sqrt(G)||_op related quantity; disjoint supports -> sqrt(sum mu_j^2),
# overlapping/aligned -> up to m*mu (constructive). Show a single scalar max_j mu_j cannot predict this.
def run(n=40,k=4,m=5,mode='disjoint',trials=400):
    ratios=[]
    for _ in range(trials):
        A=rng.standard_normal((n,k)); U,_=np.linalg.qr(A); Uread=U[:,:k]; Pread=Uread@Uread.T
        Vs,_=np.linalg.qr(rng.standard_normal((n,k))); Pperp=np.eye(n)-Vs@Vs.T
        # build m restriction maps; control overlap of their action on Uread
        Rs=[]
        base=rng.standard_normal((1,n))
        for j in range(m):
            if mode=='aligned':
                # all R_j produce nearly the SAME row-direction on Uread (overlap)
                R=(base+0.05*rng.standard_normal((1,n)))
            else:
                # disjoint: independent directions
                R=rng.standard_normal((1,n))
            Rs.append(R)
        B=sum(R.T@R for R in Rs)
        Beff=np.linalg.norm(Pperp@B@Pread,2)
        mus=[np.linalg.norm(Pperp@(R.T)@ (R@Pread),2) for R in Rs]  # per-stalk block norm
        maxmu=max(mus); sqrtsum=np.sqrt(sum(mu**2 for mu in mus)); summu=sum(mus)
        ratios.append((Beff,maxmu,sqrtsum,summu))
    A=np.array(ratios)
    return A.mean(0)
for mode in ['disjoint','aligned']:
    Beff,maxmu,sqrtsum,summu=run(mode=mode)
    print(f'{mode:9} mean||Beff||={Beff:.3f}  max_j mu_j={maxmu:.3f}  sqrt(sum mu^2)={sqrtsum:.3f}  sum mu={summu:.3f}  Beff/maxmu={Beff/maxmu:.3f}')
