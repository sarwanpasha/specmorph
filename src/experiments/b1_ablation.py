import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as TT
torch.set_num_threads(4)
dev='cuda' if torch.cuda.is_available() else 'cpu'
NTASK=15; EPOCHS=3; CAP=40; ENERGY=0.90
DATA=__import__('os').environ['PROJ']+'/data'

class MLP(nn.Module):
    def __init__(self):
        super().__init__(); self.f1=nn.Linear(784,256); self.f2=nn.Linear(256,256); self.head=nn.Linear(256,10)
    def feat(self,x): return F.relu(self.f2(F.relu(self.f1(x))))
    def forward(self,x): return self.head(self.feat(x))

def batches(X,Y,bs=128,shuffle=True):
    idx=torch.randperm(len(X)) if shuffle else torch.arange(len(X))
    for i in range(0,len(X),bs): j=idx[i:i+bs]; yield X[j],Y[j]

def feat_basis(net,perm,energy,cap):
    net.eval(); H=[]
    with torch.no_grad():
        for x,y in batches(Xtr[:,perm],Ytr,256,False):
            H.append(net.feat(x.to(dev)).cpu())
    H=torch.cat(H).t()  # d x n
    U,S,_=torch.linalg.svd(H,full_matrices=False)
    e=torch.cumsum(S**2,0)/torch.sum(S**2); k=int((e<energy).sum())+1; k=min(k,cap)
    return U[:,:k].contiguous()

def subspace_overlap(Bi,Bj): return torch.linalg.svdvals(Bi.t()@Bj)[0].item()

def train_task(net,perm,epochs):
    net.train(); opt=torch.optim.SGD(net.parameters(),lr=0.05,momentum=0.9)
    for ep in range(epochs):
        for x,y in batches(Xtr[:,perm],Ytr):
            opt.zero_grad(); loss=F.cross_entropy(net(x.to(dev)),y.to(dev)); loss.backward(); opt.step()

def b1_from_thr(W, thr):
    n=W.shape[0]; E=[(i,j) for i in range(n) for j in range(i+1,n) if W[i,j]>=thr]
    parent=list(range(n))
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    for i,j in E:
        ri,rj=find(i),find(j)
        if ri!=rj: parent[ri]=rj
    comps=len(set(find(k) for k in range(n)))
    return len(E)-n+comps, len(E), comps

if __name__=='__main__':
    seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
    torch.manual_seed(seed); np.random.seed(seed); rng=np.random.RandomState(seed)
    tf=TT.Compose([TT.ToTensor(), TT.Lambda(lambda z: z.view(-1))])
    tr=datasets.MNIST(DATA,train=True,download=False,transform=tf)
    global Xtr,Ytr
    Xtr=torch.stack([tr[i][0] for i in range(len(tr))]); Ytr=torch.tensor([tr[i][1] for i in range(len(tr))])
    perms=[torch.arange(784)]+[torch.tensor(rng.permutation(784)) for _ in range(NTASK-1)]
    net=MLP().to(dev); tb=[]
    for t in range(NTASK):
        train_task(net,perms[t],EPOCHS); tb.append(feat_basis(net,perms[t],ENERGY,CAP))
        print('trained task',t,'basis',tb[-1].shape[1],flush=True)
    n=NTASK; W=np.zeros((n,n)); ov=[]
    for i in range(n):
        for j in range(i+1,n):
            o=subspace_overlap(tb[i],tb[j]); W[i,j]=W[j,i]=o; ov.append(o)
    ov=np.array(ov)
    print(f'OVERLAP_STATS min {ov.min():.3f} med {np.median(ov):.3f} max {ov.max():.3f} mean {ov.mean():.3f}',flush=True)
    print('THRESHOLD ABLATION (seed %d):'%seed,flush=True)
    for pct in [30,40,50,60,70,80,90]:
        thr=np.percentile(ov,pct); b1,ne,comps=b1_from_thr(W,thr)
        print(f'  pct{pct} thr {thr:.3f} edges {ne} comps {comps} b1 {b1}',flush=True)
    for at in [0.3,0.4,0.5,0.6,0.7,0.8]:
        b1,ne,comps=b1_from_thr(W,at)
        print(f'  abs {at:.2f} edges {ne} comps {comps} b1 {b1}',flush=True)
