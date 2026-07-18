import argparse, time, random
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18

p = argparse.ArgumentParser()
p.add_argument('--data', default='./data')
p.add_argument('--epochs', type=int, default=5)
p.add_argument('--lam', type=float, default=0.2)
p.add_argument('--seeds', type=int, default=3)
p.add_argument('--ntask', type=int, default=10)
args = p.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
NCLS = 100
PER = NCLS // args.ntask
FEAT = 512  # resnet18 penultimate dim

MEAN = torch.tensor([0.5071,0.4865,0.4409]).view(1,3,1,1)
STD  = torch.tensor([0.2673,0.2564,0.2762]).view(1,3,1,1)

def load(split):
    d = torch.load(f'{args.data}/cifar100_{split}.pt')
    X = d['X'].permute(0,3,1,2).float().div(255.0)
    X = (X - MEAN) / STD
    Y = d['Y'].long()
    return X, Y

Xtr, Ytr = load('train'); Xte, Yte = load('test')

def task_idx(Y, t):
    lo, hi = t*PER, (t+1)*PER
    return ((Y>=lo)&(Y<hi)).nonzero(as_tuple=True)[0]

class ResNetCIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False)  # CIFAR stem
        m.maxpool = nn.Identity()
        m.fc = nn.Identity()
        self.body = m
    def forward(self,x):
        return F.relu(self.body(x))  # (B,512)

def batches(X, Y, idx, bs=128, shuffle=True):
    idx = idx[torch.randperm(len(idx))] if shuffle else idx
    for i in range(0, len(idx), bs):
        j = idx[i:i+bs]
        yield X[j].to(dev), Y[j].to(dev)

def evaluate_ti(net, head, Xs, Ys, idx, cls):
    net.eval(); correct=total=0
    cls_t = torch.tensor(cls, device=dev)
    with torch.no_grad():
        for x,y in batches(Xs,Ys,idx,256,False):
            logits = head(net(x))[:, cls_t]
            pred = cls_t[logits.argmax(1)]
            correct += (pred==y).sum().item(); total += y.numel()
    return correct/total

def feature_basis(net, idx, energy=0.95, cap=80):
    net.eval(); feats=[]; n=0
    with torch.no_grad():
        for x,_ in batches(Xtr,Ytr,idx,256,False):
            feats.append(net(x).cpu()); n+=x.shape[0]
            if n>2000: break
    M=torch.cat(feats,0); M=M-M.mean(0,keepdim=True); M=torch.nan_to_num(M)
    U,S,Vt=torch.linalg.svd(M,full_matrices=False)
    e=torch.cumsum(S**2,0)/(S**2).sum()
    r=min(int((e<energy).sum().item())+1,cap)
    return Vt[:r].to(dev)

def run(method, seed):
    torch.manual_seed(seed); random.seed(seed)
    net=ResNetCIFAR().to(dev); head=nn.Linear(FEAT,NCLS).to(dev)
    opt=torch.optim.SGD(list(net.parameters())+list(head.parameters()),lr=0.05,momentum=0.9,weight_decay=5e-4)
    tr_idx=[task_idx(Ytr,t) for t in range(args.ntask)]
    te_idx=[task_idx(Yte,t) for t in range(args.ntask)]
    cls_of=[list(range(t*PER,(t+1)*PER)) for t in range(args.ntask)]
    acc_after=[]; Pfeat=None; mu_log=[]; gmem=[]
    for t in range(args.ntask):
        cls=cls_of[t]
        for ep in range(args.epochs):
            net.train()
            for x,y in batches(Xtr,Ytr,tr_idx[t]):
                opt.zero_grad()
                z=net(x); loss=F.cross_entropy(head(z),y)
                if method=='OURS' and Pfeat is not None:
                    Wc=head.weight[cls]
                    coh=(Wc @ Pfeat.t())
                    loss=loss + args.lam*(coh.pow(2).sum()/Wc.shape[0])
                loss.backward()
                if method in ('GPM','OURS') and Pfeat is not None:
                    g=head.weight.grad
                    head.weight.grad=g-((g@Pfeat.t())@Pfeat)
                if method=='OGD' and len(gmem)>0:
                    gv=torch.cat([head.weight.grad.flatten(),head.bias.grad.flatten()])
                    for gm in gmem: gv=gv-(gv@gm)/(gm@gm+1e-12)*gm
                    n1=head.weight.numel()
                    head.weight.grad=gv[:n1].view_as(head.weight)
                    head.bias.grad=gv[n1:].view_as(head.bias)
                torch.nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),5.0)
                opt.step()
        acc_after.append(evaluate_ti(net,head,Xte,Yte,te_idx[t],cls))
        if method in ('GPM','OURS'):
            B=feature_basis(net,tr_idx[t])
            Pfeat=B if Pfeat is None else torch.cat([Pfeat,B],0)
            q,_=torch.linalg.qr(Pfeat.t()); Pfeat=q.t()[:min(Pfeat.shape[0],FEAT)]
            with torch.no_grad():
                Wc=head.weight[cls]
                mu=(Wc@Pfeat.t()).norm(dim=1).max().item()/(Wc.norm(dim=1).max().item()+1e-12)
            mu_log.append(mu)
        if method=='OGD':
            net.eval()
            for x,y in batches(Xtr,Ytr,tr_idx[t],256,False):
                head.zero_grad(); net.zero_grad()
                F.cross_entropy(head(net(x)),y).backward()
                gmem.append(torch.cat([head.weight.grad.flatten(),head.bias.grad.flatten()]).detach().clone()); break
    final=[evaluate_ti(net,head,Xte,Yte,te_idx[t],cls_of[t]) for t in range(args.ntask)]
    avg=sum(final)/len(final)
    forget=sum(max(0.0,acc_after[t]-final[t]) for t in range(args.ntask-1))/(args.ntask-1)
    mu=sum(mu_log)/len(mu_log) if mu_log else float('nan')
    aa=sum(acc_after)/len(acc_after)
    return avg,forget,mu,aa

print(f'CIFAR100-ResNet18 seeds={args.seeds} lam={args.lam} epochs={args.epochs} ntask={args.ntask} dev={dev}')
import numpy as np
agg={m:{'avg':[],'fg':[],'mu':[],'aa':[]} for m in ['FINETUNE','OGD','GPM','OURS']}
for s in range(args.seeds):
    for m in ['FINETUNE','OGD','GPM','OURS']:
        t0=time.time(); avg,fg,mu,aa=run(m,s)
        agg[m]['avg'].append(avg); agg[m]['fg'].append(fg); agg[m]['mu'].append(mu); agg[m]['aa'].append(aa)
        print(f'seed{s} {m:9} avg={avg:.4f} forget={fg:.4f} mu={mu:.4f} accAfter={aa:.4f} [{time.time()-t0:.0f}s]',flush=True)
print('='*72)
print(f'{"method":9}{"avg_acc":>16}{"forget":>16}{"mu":>16}{"accAfter":>10}')
for m in ['FINETUNE','OGD','GPM','OURS']:
    a=np.array(agg[m]['avg']); f=np.array(agg[m]['fg']); u=np.array(agg[m]['mu']); aa=np.array(agg[m]['aa'])
    mus='---' if np.isnan(u).all() else f'{u.mean():.3f}+-{u.std():.3f}'
    print(f'{m:9}{a.mean():.4f}+-{a.std():.4f}   {f.mean():.4f}+-{f.std():.4f}   {mus:>14}   {aa.mean():.3f}')
print('='*72)
print('Done.')
