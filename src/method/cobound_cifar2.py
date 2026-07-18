import argparse, time, math, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18

p=argparse.ArgumentParser()
p.add_argument('--data', default='./data')
p.add_argument('--epochs', type=int, default=5)
p.add_argument('--ntask', type=int, default=10)
p.add_argument('--seeds', type=int, default=3)
p.add_argument('--energy', type=float, default=0.95)
p.add_argument('--cap', type=int, default=60)
p.add_argument('--tag', default='cobound')
args=p.parse_args()
dev='cuda' if torch.cuda.is_available() else 'cpu'
PER=100//args.ntask

tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071,0.4865,0.4409),(0.2673,0.2564,0.2762))])
tr=datasets.CIFAR100(args.data,train=True,download=True,transform=None)
te=datasets.CIFAR100(args.data,train=False,download=True,transform=None)
def pack(ds):
    X=torch.stack([tf(img) for img,_ in ds]); Y=torch.tensor([y for _,y in ds]); return X,Y
Xtr,Ytr=pack(tr); Xte,Yte=pack(te)
print('data',Xtr.shape,Xte.shape,flush=True)

def task_idx(Y,t):
    lo,hi=t*PER,(t+1)*PER; return ((Y>=lo)&(Y<hi)).nonzero(as_tuple=True)[0]

class Net(nn.Module):
    def __init__(self):
        super().__init__(); m=resnet18(weights=None)
        m.conv1=nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool=nn.Identity(); m.fc=nn.Identity()
        self.body=m
    def forward(self,x): return F.relu(self.body(x))

def batches(X,Y,idx,bs=128,shuffle=True):
    idx=idx[torch.randperm(len(idx))] if shuffle else idx
    for i in range(0,len(idx),bs):
        j=idx[i:i+bs]; yield X[j].to(dev),Y[j].to(dev)

def evaluate_ti(net,head,Xs,Ys,idx,cls):
    net.eval(); correct=total=0; cls_t=torch.tensor(cls,device=dev)
    with torch.no_grad():
        for x,y in batches(Xs,Ys,idx,256,False):
            logits=head(net(x))[:,cls_t]; pred=cls_t[logits.argmax(1)]
            correct+=(pred==y).sum().item(); total+=y.numel()
    return correct/total

def feat_basis(net,idx,energy,cap):
    net.eval(); feats=[]; n=0
    with torch.no_grad():
        for x,_ in batches(Xtr,Ytr,idx,256,False):
            feats.append(net(x).cpu()); n+=x.shape[0]
            if n>2000: break
    M=torch.cat(feats,0); M=M-M.mean(0,keepdim=True); M=torch.nan_to_num(M)
    U,S,Vt=torch.linalg.svd(M,full_matrices=False)
    e=torch.cumsum(S**2,0)/(S**2).sum(); r=min(int((e<energy).sum().item())+1,cap)
    return Vt[:r].to(dev)   # r x D rows = orthonormal basis of task feature subspace

def ortho(rows):
    # rows: list of (k_i x D) tensors -> orthonormal basis (m x D) of their span
    if len(rows)==0: return None
    A=torch.cat(rows,0)
    Q,_=torch.linalg.qr(A.T)  # D x m
    return Q.T.contiguous()   # m x D orthonormal

def cobound_basis(nbr_bases):
    # Coboundary-consistent (im delta) directions are SHARED across >=2 neighbor
    # stalks; directions private to one neighbor are H^1 candidates (cycle-
    # inconsistent) that projection cannot correct (Thm 4), so we do not spend
    # plasticity protecting them. Keep the shared subspace only.
    if len(nbr_bases)==0: return None
    if len(nbr_bases)==1: return nbr_bases[0]
    U=ortho(nbr_bases)
    if U is None: return None
    score=torch.zeros(U.shape[0], device=U.device)
    for Bk in nbr_bases:
        P=Bk.T@Bk
        score+= (U@P*U).sum(1)
    keep = score>1.5
    if keep.sum().item()==0: return None
    return ortho([U[keep]])

def proj_grad(g, basis):
    # remove component of grad-on-feature-weights along span(basis rows); basis: m x D or None
    if basis is None: return g
    # g shape (out, D); project each row's D-part
    P=basis.T@basis  # D x D projector onto span
    return g - g@P

def run(method, seed, cyc_edges):
    torch.manual_seed(seed); np.random.seed(seed)
    net=Net().to(dev); head=nn.Linear(512,100).to(dev)
    opt=torch.optim.SGD(list(net.parameters())+list(head.parameters()),lr=0.05,momentum=0.9,weight_decay=5e-4)
    ce=nn.CrossEntropyLoss()
    task_basis=[None]*args.ntask   # per-task feature basis (rows x 512)
    acc_after=[0.0]*args.ntask
    # neighbor sets from cyclic task-interaction graph
    nbr=[[] for _ in range(args.ntask)]
    for a,b in cyc_edges: nbr[a].append(b); nbr[b].append(a)
    for t in range(args.ntask):
        idx=task_idx(Ytr,t); cls=list(range(t*PER,(t+1)*PER))
        # build projection basis for this task's updates
        if method=='FINETUNE':
            B=None
        elif method=='GPM':
            past=[task_basis[s] for s in range(t) if task_basis[s] is not None]
            B=ortho(past)
        elif method=='COBOUND':
            # only protect directions coupled along graph edges incident to already-seen neighbors
            past=[task_basis[s] for s in nbr[t] if s<t and task_basis[s] is not None]
            B=ortho(past)
        elif method=='COBOUND2':
            # delta-aware: protect only the coboundary-consistent (shared) part
            past=[task_basis[s] for s in nbr[t] if s<t and task_basis[s] is not None]
            B=cobound_basis(past)
        net.train()
        for ep in range(args.epochs):
            for x,y in batches(Xtr,Ytr,idx):
                opt.zero_grad(); loss=ce(head(net(x)),y); loss.backward()
                if B is not None and head.weight.grad is not None:
                    head.weight.grad[:]=proj_grad(head.weight.grad,B)
                opt.step()
        task_basis[t]=feat_basis(net,idx,args.energy,args.cap)
        acc_after[t]=evaluate_ti(net,head,Xte,Yte,task_idx(Yte,t),cls)
    # final accuracies + forgetting
    finals=[]
    for t in range(args.ntask):
        cls=list(range(t*PER,(t+1)*PER)); finals.append(evaluate_ti(net,head,Xte,Yte,task_idx(Yte,t),cls))
    avg=sum(finals)/len(finals)
    forget=sum(max(0.0,acc_after[t]-finals[t]) for t in range(args.ntask-1))/(args.ntask-1)
    return avg,forget,finals

def cyclic_graph(n):
    # engineered sparse cyclic task-interaction graph: a ring 0-1-2-...-n-1-0
    # plus two chords to create multiple independent cycles (nontrivial H1)
    E=[(i,(i+1)%n) for i in range(n)]
    if n>=6:
        E+=[(0,n//2),(1,n//2+1)]
    return E

if __name__=='__main__':
    E=cyclic_graph(args.ntask)
    print('CYCGRAPH edges',E,flush=True)
    import collections
    agg=collections.defaultdict(lambda:{'avg':[],'fg':[]})
    for seed in range(args.seeds):
        for m in ['FINETUNE','GPM','COBOUND','COBOUND2']:
            t0=time.time(); avg,fg,finals=run(m,seed,E)
            agg[m]['avg'].append(avg); agg[m]['fg'].append(fg)
            print(f'seed{seed} {m:9} avg={avg:.4f} forget={fg:.4f} [{time.time()-t0:.0f}s]',flush=True)
    print('==== SUMMARY (ntask=%d, epochs=%d, seeds=%d) ===='%(args.ntask,args.epochs,args.seeds),flush=True)
    print(f'{"method":10}{"avg_acc":>12}{"forget":>12}',flush=True)
    for m in ['FINETUNE','GPM','COBOUND','COBOUND2']:
        a=np.array(agg[m]['avg']); f=np.array(agg[m]['fg'])
        print(f'{m:10}{a.mean():>8.4f}+-{a.std():.3f}{f.mean():>8.4f}+-{f.std():.3f}',flush=True)
