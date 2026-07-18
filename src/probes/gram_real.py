import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, random, torchvision, argparse
ap=argparse.ArgumentParser()
ap.add_argument('--ntasks',type=int,default=8)
ap.add_argument('--epochs',type=int,default=2)
ap.add_argument('--seeds',type=int,default=5)
ap.add_argument('--r',type=int,default=12)   # per-task restriction-map rank
args=ap.parse_args()
dev='cuda' if torch.cuda.is_available() else 'cpu'
DIN,FEAT,NCLS=784,256,10

# ---- data ----
tr=torchvision.datasets.MNIST('.',train=True,download=True)
te=torchvision.datasets.MNIST('.',train=False,download=True)
TRX=(tr.data.float().view(-1,784)/255.0); TRY=tr.targets.clone()
TEX=(te.data.float().view(-1,784)/255.0); TEY=te.targets.clone()
print("loaded",TRX.shape,TEX.shape,flush=True)

class MLP(nn.Module):
    def __init__(s):
        super().__init__()
        s.f1=nn.Linear(DIN,FEAT); s.f2=nn.Linear(FEAT,FEAT); s.head=nn.Linear(FEAT,NCLS)
    def feat(s,x):
        h=F.relu(s.f1(x)); h=F.relu(s.f2(h)); return h
    def forward(s,x): return s.head(s.feat(x))

def batches(X,Y,bs,sh=True):
    n=X.shape[0]; idx=torch.randperm(n) if sh else torch.arange(n)
    for i in range(0,n,bs):
        j=idx[i:i+bs]; yield X[j],Y[j]

def feat_basis(net, perm, r, cap=3000):
    # top-r principal subspace of centered features under this permutation -> restriction map R (r x FEAT)
    net.eval(); feats=[]; n=0
    with torch.no_grad():
        for x,_ in batches(TRX,TRY,1000,False):
            feats.append(net.feat(x[:,perm].to(dev)).cpu()); n+=x.shape[0]
            if n>cap: break
    M=torch.cat(feats,0); M=M-M.mean(0,keepdim=True); M=torch.nan_to_num(M)
    U,S,Vt=torch.linalg.svd(M,full_matrices=False)
    return Vt[:r]   # (r, FEAT) orthonormal rows

def run_seed(seed):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    perms=[torch.randperm(DIN) for _ in range(args.ntasks)]
    net=MLP().to(dev); opt=torch.optim.SGD(net.parameters(),lr=0.1,momentum=0.9)
    Rmaps=[]  # restriction map per task, built AFTER training that task
    growth=[] # (k, ||B_eff||, max_mu, gram_bound)
    for t in range(args.ntasks):
        perm=perms[t]
        for ep in range(args.epochs):
            net.train()
            for x,y in batches(TRX,TRY,128):
                x=x[:,perm].to(dev); y=y.to(dev)
                opt.zero_grad(); loss=F.cross_entropy(net.head(net.feat(x)),y); loss.backward(); opt.step()
        # restriction map for this task, measured on the shared trained backbone
        R=feat_basis(net,perm,args.r).to(dev)     # (r,FEAT)
        Rmaps.append(R)
        # read-out reference = subspace retained from task 0
        Uread=Rmaps[0].T                          # (FEAT, r)
        # task-interaction graph: all pairs (i<j) sharing the read-out (complete graph on tasks so far)
        k=len(Rmaps)
        if k<2:
            growth.append((k,0.0,0.0,0.0)); continue
        # B_eff = sum over edges R_i^T S_ij R_j ; use S_ij=I (worst-case unit coupling on the shared support)
        Beff=torch.zeros(FEAT,FEAT,device=dev)
        Ga=torch.zeros(FEAT,FEAT,device=dev); Gb=torch.zeros(FEAT,FEAT,device=dev)
        mus=[]
        for i in range(k):
            for j in range(i+1,k):
                Ri,Rj=Rmaps[i],Rmaps[j]                      # (r,FEAT)
                Beff=Beff + Ri.T@Rj                          # (FEAT,FEAT)
                Ga=Ga + Ri.T@Ri; Gb=Gb + Rj.T@Rj
                # per-pair coherence: overlap of task i's map with the read-out subspace
                mu_ij=torch.linalg.norm(Ri@Uread,2).item()
                mus.append(mu_ij)
        nb=torch.linalg.norm(Beff,2).item()
        gbound=(torch.linalg.norm(Ga,2).item()**0.5)*(torch.linalg.norm(Gb,2).item()**0.5)
        growth.append((k,nb,max(mus),gbound))
    return growth

allg={}
for s in range(args.seeds):
    g=run_seed(s)
    for (k,nb,mm,gb) in g:
        allg.setdefault(k,[]).append((nb,mm,gb))
    print("seed",s,"done",flush=True)

print("\n=== REAL-DATA THEOREM 3 (permuted-MNIST, shared head, %d seeds) ==="%args.seeds)
print("k    ||B_eff||        max_mu           gram_bound      boundHolds")
import statistics as st
ks=sorted(allg)
for k in ks:
    nbs=[a[0] for a in allg[k]]; mms=[a[1] for a in allg[k]]; gbs=[a[2] for a in allg[k]]
    holds=all(a[0]<=a[2]+1e-4 for a in allg[k])
    print("%d  %.4f+-%.4f  %.4f+-%.4f  %.4f+-%.4f  %s"%(
        k, st.mean(nbs), (st.pstdev(nbs)), st.mean(mms), st.pstdev(mms), st.mean(gbs), st.pstdev(gbs), str(holds)))
