import argparse,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F,collections
from torchvision import datasets,transforms
p=argparse.ArgumentParser()
p.add_argument('--data',default='./data'); p.add_argument('--epochs',type=int,default=3)
p.add_argument('--ntask',type=int,default=15); p.add_argument('--seeds',type=int,default=3)
p.add_argument('--cap',type=int,default=40); p.add_argument('--energy',type=float,default=0.90)
p.add_argument('--tau',type=float,default=-1.0)
args=p.parse_args(); dev='cuda' if torch.cuda.is_available() else 'cpu'
tf=transforms.Compose([transforms.ToTensor()])
tr=datasets.MNIST(args.data,train=True,download=False,transform=tf)
te=datasets.MNIST(args.data,train=False,download=False,transform=tf)
Xtr=tr.data.float().view(-1,784)/255.; Ytr=tr.targets
Xte=te.data.float().view(-1,784)/255.; Yte=te.targets
print('data',Xtr.shape,flush=True)
class MLP(nn.Module):
    def __init__(self):
        super().__init__(); self.f1=nn.Linear(784,256); self.f2=nn.Linear(256,256); self.head=nn.Linear(256,10)
    def feat(self,x): return F.relu(self.f2(F.relu(self.f1(x))))
    def forward(self,x): return self.head(self.feat(x))
def batches(X,Y,bs=128,shuffle=True):
    idx=torch.randperm(len(X)) if shuffle else torch.arange(len(X))
    for i in range(0,len(X),bs):
        j=idx[i:i+bs]; yield X[j].to(dev),Y[j].to(dev)
def feat_basis(net,perm,energy,cap):
    net.eval(); fs=[]; n=0
    with torch.no_grad():
        for x,_ in batches(Xtr[:,perm],Ytr,256,False):
            fs.append(net.feat(x).cpu()); n+=x.shape[0]
            if n>3000: break
    M=torch.cat(fs,0); M=M-M.mean(0,keepdim=True)
    U,S,Vt=torch.linalg.svd(M,full_matrices=False)
    e=torch.cumsum(S**2,0)/(S**2).sum(); r=min(int((e<energy).sum().item())+1,cap)
    return Vt[:r].to(dev)
def ev(net,perm):
    net.eval(); c=t=0
    with torch.no_grad():
        for x,y in batches(Xte[:,perm],Yte,256,False):
            pr=net(x).argmax(1); c+=(pr==y).sum().item(); t+=y.numel()
    return c/t
def subspace_overlap(Bi,Bj): return torch.linalg.svdvals(Bi@Bj.T)[0].item()
def ortho(rows):
    if not rows: return None
    Q,_=torch.linalg.qr(torch.cat(rows,0).T); return Q.T.contiguous()
def proj_out(g,B):
    if B is None: return g
    return g-(g@B.T)@B
def measure_graph(bases,tau):
    n=len(bases); ov=[]; W=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            o=subspace_overlap(bases[i],bases[j]); W[i,j]=W[j,i]=o; ov.append(o)
    ov=np.array(ov); thr=np.median(ov) if tau<0 else tau
    E=[(i,j) for i in range(n) for j in range(i+1,n) if W[i,j]>=thr]
    parent=list(range(n))
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    for i,j in E:
        ri,rj=find(i),find(j)
        if ri!=rj: parent[ri]=rj
    comps=len(set(find(k) for k in range(n))); b1=len(E)-n+comps
    return E,W,thr,b1,comps,ov

def train_task(net,perm,epochs,Bprot):
    opt=torch.optim.SGD(net.parameters(),lr=0.1,momentum=0.9); ce=nn.CrossEntropyLoss(); net.train()
    for ep in range(epochs):
        for x,y in batches(Xtr[:,perm],Ytr):
            opt.zero_grad(); ce(net(x),y).backward()
            if Bprot is not None:
                g=net.f2.weight.grad; P=Bprot.T@Bprot; net.f2.weight.grad[:]=g-P@g
                if net.head.weight.grad is not None: net.head.weight.grad[:]=proj_out(net.head.weight.grad,Bprot)
            opt.step()

def run(method,seed,perms,tau):
    torch.manual_seed(seed); np.random.seed(seed)
    net=MLP().to(dev); tb=[None]*args.ntask; acc_after=[0.]*args.ntask
    for t in range(args.ntask):
        perm=perms[t]
        if method=='FINETUNE': Bprot=None
        elif method=='GPM': Bprot=ortho([tb[s] for s in range(t) if tb[s] is not None])
        else:
            prelim=feat_basis(net,perm,args.energy,args.cap); nbrs=[]
            for s in range(t):
                if tb[s] is None: continue
                if subspace_overlap(prelim,tb[s])>=tau: nbrs.append(tb[s])
            Bprot=ortho(nbrs)
        train_task(net,perm,args.epochs,Bprot)
        tb[t]=feat_basis(net,perm,args.energy,args.cap); acc_after[t]=ev(net,perm)
    finals=[ev(net,perms[t]) for t in range(args.ntask)]
    avg=sum(finals)/len(finals)
    fg=sum(max(0.,acc_after[t]-finals[t]) for t in range(args.ntask-1))/(args.ntask-1)
    E,W,thr,b1,comps,ov=measure_graph([b for b in tb if b is not None],tau if tau>0 else -1.0)
    return avg,fg,b1,len(E),comps

if __name__=='__main__':
    rng0=np.random.RandomState(999)
    perms=[torch.arange(784)]+[torch.tensor(rng0.permutation(784)) for _ in range(args.ntask-1)]
    tau=args.tau
    if tau<0:
        torch.manual_seed(0); np.random.seed(0); net0=MLP().to(dev); tb0=[]
        for t in range(args.ntask):
            train_task(net0,perms[t],args.epochs,None); tb0.append(feat_basis(net0,perms[t],args.energy,args.cap))
        E,W,thr,b1,comps,ov=measure_graph(tb0,-1.0); tau=float(thr)
        print(f'ADAPTIVE tau={tau:.4f} dryrun |E|={len(E)} b1={b1} comps={comps} ov[min/med/max]={ov.min():.3f}/{np.median(ov):.3f}/{ov.max():.3f}',flush=True)
    agg=collections.defaultdict(lambda:{'avg':[],'fg':[],'b1':[]})
    for seed in range(args.seeds):
        for m in ['FINETUNE','GPM','COBOUND']:
            t0=time.time(); avg,fg,b1,ne,comps=run(m,seed,perms,tau)
            agg[m]['avg'].append(avg); agg[m]['fg'].append(fg); agg[m]['b1'].append(b1)
            print(f'seed{seed} {m:9} avg={avg:.4f} forget={fg:.4f} b1={b1} |E|={ne} [{time.time()-t0:.0f}s]',flush=True)
    print('==== SUMMARY realgraph pmnist (ntask=%d epochs=%d seeds=%d tau=%.3f) ===='%(args.ntask,args.epochs,args.seeds,tau),flush=True)
    for m in ['FINETUNE','GPM','COBOUND']:
        a=np.array(agg[m]['avg']); f=np.array(agg[m]['fg']); b=np.array(agg[m]['b1'])
        print(f'{m:10} avg={a.mean():.4f}+-{a.std():.3f} forget={f.mean():.4f}+-{f.std():.3f} b1={b.mean():.1f}',flush=True)
