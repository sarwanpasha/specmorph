import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as TT
torch.set_num_threads(4)
dev='cuda' if torch.cuda.is_available() else 'cpu'
NTASK=10; EPOCHS=3; CAP=40; ENERGY=0.90
DATA=__import__('os').environ['PROJ']+'/data'

class MLP(nn.Module):
    def __init__(self):
        super().__init__(); self.f1=nn.Linear(784,256); self.f2=nn.Linear(256,256); self.head=nn.Linear(256,10)
    def feat(self,x): return F.relu(self.f2(F.relu(self.f1(x))))
    def forward(self,x): return self.head(self.feat(x))

def batches(X,Y,bs=128,shuffle=True):
    idx=torch.randperm(len(X)) if shuffle else torch.arange(len(X))
    for i in range(0,len(X),bs): j=idx[i:i+bs]; yield X[j],Y[j]

def feat_basis(net,perm,cap):
    net.eval(); H=[]
    with torch.no_grad():
        for x,y in batches(Xtr[:,perm],Ytr,256,False):
            H.append(net.feat(x.to(dev)).cpu())
    H=torch.cat(H).t()
    U,S,_=torch.linalg.svd(H,full_matrices=False)
    e=torch.cumsum(S**2,0)/torch.sum(S**2); k=int((e<ENERGY).sum())+1; k=min(k,cap)
    return U[:,:k].contiguous()

def train_task(net,perm,epochs):
    net.train(); opt=torch.optim.SGD(net.parameters(),lr=0.05,momentum=0.9)
    for ep in range(epochs):
        for x,y in batches(Xtr[:,perm],Ytr):
            opt.zero_grad(); F.cross_entropy(net(x.to(dev)),y.to(dev)).backward(); opt.step()

def acc(net,perm,Xte,Yte):
    net.eval(); c=0;t=0
    with torch.no_grad():
        for x,y in batches(Xte[:,perm],Yte,512,False):
            p=net(x.to(dev)).argmax(1).cpu(); c+=(p==y).sum().item(); t+=len(y)
    return c/t

def sin_theta(Bi,Bj):
    # largest principal angle sine between column spaces
    s=torch.linalg.svdvals(Bi.t()@Bj)  # cosines
    cmin=float(s.min().clamp(max=1.0))
    return float(np.sqrt(max(0.0,1.0-cmin**2)))

def coherence(Bi,Bj):
    return torch.linalg.svdvals(Bi.t()@Bj)[0].item()

if __name__=='__main__':
    seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
    torch.manual_seed(seed); np.random.seed(seed); rng=np.random.RandomState(seed)
    tf=TT.Compose([TT.ToTensor(), TT.Lambda(lambda z: z.view(-1))])
    tr=datasets.MNIST(DATA,train=True,download=False,transform=tf)
    te=datasets.MNIST(DATA,train=False,download=False,transform=tf)
    global Xtr,Ytr
    Xtr=torch.stack([tr[i][0] for i in range(len(tr))]); Ytr=torch.tensor([tr[i][1] for i in range(len(tr))])
    Xte=torch.stack([te[i][0] for i in range(len(te))]); Yte=torch.tensor([te[i][1] for i in range(len(te))])
    perms=[torch.arange(784)]+[torch.tensor(rng.permutation(784)) for _ in range(NTASK-1)]
    net=MLP().to(dev)
    bases=[]; acc_after=[[None]*NTASK for _ in range(NTASK)]
    for t in range(NTASK):
        train_task(net,perms[t],EPOCHS)
        bases.append(feat_basis(net,perms[t],CAP))
        for s in range(t+1): acc_after[t][s]=acc(net,perms[s],Xte,Yte)
    # per past task s: coherence with task s+1 basis at learn time; sinTheta of its subspace after training vs final; drop
    mus=[]; sins=[]; drops=[]
    for s in range(NTASK-1):
        mu=coherence(bases[s],bases[s+1])  # coupling coherence to next task
        st=sin_theta(bases[s],bases[NTASK-1])  # rotation of subspace basis s vs final task subspace
        drop=acc_after[s][s]-acc_after[NTASK-1][s]  # forgetting on task s
        mus.append(mu); sins.append(st); drops.append(drop)
        print(f'task{s} mu {mu:.3f} sinTheta {st:.3f} drop {drop:.3f}',flush=True)
    mus=np.array(mus); sins=np.array(sins); drops=np.array(drops)
    def pear(a,b):
        a=a-a.mean(); b=b-b.mean(); d=np.sqrt((a*a).sum()*(b*b).sum()); return float((a*b).sum()/d) if d>0 else 0.0
    print(f'CORR seed{seed} pearson(mu,drop) {pear(mus,drops):.3f} pearson(sinTheta,drop) {pear(sins,drops):.3f} pearson(mu,sinTheta) {pear(mus,sins):.3f}',flush=True)
