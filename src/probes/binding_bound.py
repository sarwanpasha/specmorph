import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.datasets as datasets, torchvision.transforms as TT
DATA='./data'; dev='cuda' if torch.cuda.is_available() else 'cpu'
ENERGY=0.90; CAP=40; EPOCHS=3; NTASK=5
# --- MLP on 784-dim MNIST, shared head so forgetting is real ---
class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.f1=nn.Linear(784,256); s.f2=nn.Linear(256,256); s.head=nn.Linear(256,10)
    def feat(s,x): h=F.relu(s.f1(x)); h=F.relu(s.f2(h)); return h
    def forward(s,x): return s.head(s.feat(x))
def batches(X,Y,bs=128,shuffle=True):
    idx=torch.randperm(len(X)) if shuffle else torch.arange(len(X))
    for i in range(0,len(X),bs): j=idx[i:i+bs]; yield X[j],Y[j]
def feat_basis(net,mask,cap):
    net.eval(); H=[]
    with torch.no_grad():
        for x,y in batches(Xtr*mask,Ytr,256,False): H.append(net.feat(x.to(dev)).cpu())
    H=torch.cat(H).t(); U,S,_=torch.linalg.svd(H,full_matrices=False)
    e=torch.cumsum(S**2,0)/torch.sum(S**2); k=int((e<ENERGY).sum())+1; k=min(k,cap)
    return U[:,:k].contiguous()
def train_task(net,mask,epochs):
    net.train(); opt=torch.optim.SGD(net.parameters(),lr=0.05,momentum=0.9)
    for ep in range(epochs):
        for x,y in batches(Xtr*mask,Ytr): opt.zero_grad(); F.cross_entropy(net((x*1.0).to(dev)),y.to(dev)).backward(); opt.step()
def acc(net,mask,Xt,Yt):
    net.eval(); c=0;t=0
    with torch.no_grad():
        for x,y in batches(Xt*mask,Yt,512,False): p=net((x*1.0).to(dev)).argmax(1).cpu(); c+=(p==y).sum().item(); t+=len(y)
    return c/t
def coherence(Bi,Bj): return torch.linalg.svdvals(Bi.t()@Bj)[0].item()
if __name__=='__main__':
    seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
    torch.manual_seed(seed); np.random.seed(seed); rng=np.random.RandomState(seed)
    tf=TT.Compose([TT.ToTensor(), TT.Lambda(lambda z: z.view(-1))])
    tr=datasets.MNIST(DATA,train=True,download=False,transform=tf)
    te=datasets.MNIST(DATA,train=False,download=False,transform=tf)
    global Xtr,Ytr
    Xtr=torch.stack([tr[i][0] for i in range(len(tr))]); Ytr=torch.tensor([tr[i][1] for i in range(len(tr))])
    Xte=torch.stack([te[i][0] for i in range(len(te))]); Yte=torch.tensor([te[i][1] for i in range(len(te))])
    # DISJOINT-SUPPORT tasks: each task sees a distinct pixel block (near-orthogonal inputs -> low mu)
    px=torch.randperm(784); blk=784//NTASK
    masks=[]
    for t in range(NTASK):
        m=torch.zeros(784); m[px[t*blk:(t+1)*blk]]=1.0; masks.append(m)
    net=MLP().to(dev)
    bases=[]; acc_after=[[None]*NTASK for _ in range(NTASK)]
    for t in range(NTASK):
        train_task(net,masks[t],EPOCHS)
        bases.append(feat_basis(net,masks[t],CAP))
        for s in range(t+1): acc_after[t][s]=acc(net,masks[s],Xte,Yte)
    # measured forgetting on task 0 after all tasks, and bound RHS from measured mu
    # bound (schematic operational form): drop <= C * mu_max, C calibrated from first retained step
    mus=[coherence(bases[0],bases[j]) for j in range(1,NTASK)]
    mu_max=max(mus)
    drop0=acc_after[0][0]-acc_after[NTASK-1][0]
    # single retained-task check: predicted vs measured on the coupling to task 1
    drop_step=acc_after[0][0]-acc_after[1][0]
    print(f'SEED {seed} disjoint-support NTASK={NTASK}',flush=True)
    for j,m in enumerate(mus): print(f'  mu(task0,task{j+1}) = {m:.4f}',flush=True)
    print(f'RESULT seed{seed} mu_max {mu_max:.4f} drop0_total {drop0:.4f} drop0_step1 {drop_step:.4f} acc0_init {acc_after[0][0]:.4f} acc0_final {acc_after[NTASK-1][0]:.4f}',flush=True)
