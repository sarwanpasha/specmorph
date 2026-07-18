import argparse,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F,collections
from torchvision import datasets,transforms
p=argparse.ArgumentParser()
p.add_argument('--data',default='./data'); p.add_argument('--epochs',type=int,default=3)
p.add_argument('--ntask',type=int,default=10); p.add_argument('--seeds',type=int,default=3)
p.add_argument('--cap',type=int,default=60); p.add_argument('--energy',type=float,default=0.95)
args=p.parse_args(); dev='cuda' if torch.cuda.is_available() else 'cpu'
tf=transforms.Compose([transforms.ToTensor()])
tr=datasets.MNIST(args.data,train=True,download=True,transform=tf)
te=datasets.MNIST(args.data,train=False,download=True,transform=tf)
Xtr=tr.data.float().view(-1,784)/255.; Ytr=tr.targets
Xte=te.data.float().view(-1,784)/255.; Yte=te.targets
print('data',Xtr.shape,Xte.shape,flush=True)

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
    net.eval(); feats=[]; n=0
    with torch.no_grad():
        for x,_ in batches(Xtr[:,perm],Ytr,256,False):
            feats.append(net.feat(x).cpu()); n+=x.shape[0]
            if n>3000: break
    M=torch.cat(feats,0); M=M-M.mean(0,keepdim=True)
    U,S,Vt=torch.linalg.svd(M,full_matrices=False)
    e=torch.cumsum(S**2,0)/(S**2).sum(); r=min(int((e<energy).sum().item())+1,cap)
    return Vt[:r].to(dev)

def ev(net,perm):
    net.eval(); c=t=0
    with torch.no_grad():
        for x,y in batches(Xte[:,perm],Yte,256,False):
            p=net(x).argmax(1); c+=(p==y).sum().item(); t+=y.numel()
    return c/t

def ortho(rows):
    if not rows: return None
    Q,_=torch.linalg.qr(torch.cat(rows,0).T); return Q.T.contiguous()

def proj_grad(g,B):
    if B is None: return g
    return g - g@(B.T@B)

def run(method,seed,perms,edges):
    torch.manual_seed(seed); np.random.seed(seed)
    net=MLP().to(dev); opt=torch.optim.SGD(net.parameters(),lr=0.1,momentum=0.9)
    ce=nn.CrossEntropyLoss(); tb=[None]*args.ntask; acc_after=[0.]*args.ntask
    nbr=[[] for _ in range(args.ntask)]
    for a,b in edges: nbr[a].append(b); nbr[b].append(a)
    for t in range(args.ntask):
        perm=perms[t]
        if method=='FINETUNE': B=None
        elif method=='GPM': B=ortho([tb[s] for s in range(t) if tb[s] is not None])
        else: B=ortho([tb[s] for s in nbr[t] if s<t and tb[s] is not None])
        net.train()
        for ep in range(args.epochs):
            for x,y in batches(Xtr[:,perm],Ytr):
                opt.zero_grad(); ce(net(x),y).backward()
                if B is not None:
                    # project the feature-producing layer (f2) output-rows onto complement of protected subspace
                    g=net.f2.weight.grad  # (256_out, 256_in): rows live in the 256-dim feature space
                    P=B.T@B               # (256x256) projector onto protected span (B rows are basis vectors in R^256)
                    net.f2.weight.grad[:]=g - P@g
                    if net.head.weight.grad is not None:
                        net.head.weight.grad[:]=proj_grad(net.head.weight.grad,B)
                opt.step()
        tb[t]=feat_basis(net,perm,args.energy,args.cap); acc_after[t]=ev(net,perm)
    finals=[ev(net,perms[t]) for t in range(args.ntask)]
    avg=sum(finals)/len(finals)
    fg=sum(max(0.,acc_after[t]-finals[t]) for t in range(args.ntask-1))/(args.ntask-1)
    return avg,fg

def cyclic_graph(n):
    E=[(i,(i+1)%n) for i in range(n)]
    if n>=6: E+=[(0,n//2),(1,n//2+1)]
    return E

if __name__=='__main__':
    E=cyclic_graph(args.ntask); print('CYCGRAPH',E,flush=True)
    agg=collections.defaultdict(lambda:{'avg':[],'fg':[]})
    for seed in range(args.seeds):
        rng=np.random.RandomState(1000+seed)
        perms=[torch.arange(784)]+[torch.tensor(rng.permutation(784)) for _ in range(args.ntask-1)]
        for m in ['FINETUNE','GPM','COBOUND']:
            t0=time.time(); avg,fg=run(m,seed,perms,E)
            agg[m]['avg'].append(avg); agg[m]['fg'].append(fg)
            print(f'seed{seed} {m:9} avg={avg:.4f} forget={fg:.4f} [{time.time()-t0:.0f}s]',flush=True)
    print('==== SUMMARY pmnist (ntask=%d,epochs=%d,seeds=%d) ===='%(args.ntask,args.epochs,args.seeds),flush=True)
    for m in ['FINETUNE','GPM','COBOUND']:
        a=np.array(agg[m]['avg']); f=np.array(agg[m]['fg'])
        print(f'{m:10} avg={a.mean():.4f}+-{a.std():.3f} forget={f.mean():.4f}+-{f.std():.3f}',flush=True)
