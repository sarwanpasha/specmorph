import argparse, time, random
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18

p = argparse.ArgumentParser()
p.add_argument('--data', default='./data')
p.add_argument('--epochs', type=int, default=5)
p.add_argument('--seeds', type=int, default=3)
p.add_argument('--ntask', type=int, default=10)
p.add_argument('--lams', default='0.2,1.0,5.0')  # sweep of penalty weights for OURS(feat)
args = p.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
NCLS = 100; PER = NCLS // args.ntask; FEAT = 512
LAMS = [float(x) for x in args.lams.split(',')]

MEAN = torch.tensor([0.5071,0.4865,0.4409]).view(1,3,1,1)
STD  = torch.tensor([0.2673,0.2564,0.2762]).view(1,3,1,1)
def load(split):
    d = torch.load(f'{args.data}/cifar100_{split}.pt')
    X = d['X'].permute(0,3,1,2).float().div(255.0); X=(X-MEAN)/STD
    return X, d['Y'].long()
Xtr,Ytr = load('train'); Xte,Yte = load('test')
def task_idx(Y,t):
    lo,hi=t*PER,(t+1)*PER; return ((Y>=lo)&(Y<hi)).nonzero(as_tuple=True)[0]

class ResNetCIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        m=resnet18(weights=None); m.conv1=nn.Conv2d(3,64,3,1,1,bias=False)
        m.maxpool=nn.Identity(); m.fc=nn.Identity(); self.body=m
    def forward(self,x): return F.relu(self.body(x))

def batches(X,Y,idx,bs=128,shuffle=True):
    idx = idx[torch.randperm(len(idx))] if shuffle else idx
    for i in range(0,len(idx),bs):
        j=idx[i:i+bs]; yield X[j].to(dev),Y[j].to(dev)

def evaluate_ti(net,head,Xs,Ys,idx,cls):
    net.eval(); c=t=0; ct=torch.tensor(cls,device=dev)
    with torch.no_grad():
        for x,y in batches(Xs,Ys,idx,256,False):
            lg=head(net(x))[:,ct]; pred=ct[lg.argmax(1)]; c+=(pred==y).sum().item(); t+=y.numel()
    return c/t

def feature_basis(net,idx,energy=0.95,cap=80):
    net.eval(); fs=[]; n=0
    with torch.no_grad():
        for x,_ in batches(Xtr,Ytr,idx,256,False):
            fs.append(net(x).cpu()); n+=x.shape[0]
            if n>2000: break
    M=torch.cat(fs,0); M=M-M.mean(0,keepdim=True); M=torch.nan_to_num(M)
    _,_,Vt=torch.linalg.svd(M,full_matrices=False)
    e=torch.cumsum(_.new_zeros(1),0) if False else None
    U,S,Vt=torch.linalg.svd(M,full_matrices=False)
    e=torch.cumsum(S**2,0)/(S**2).sum(); r=min(int((e<energy).sum().item())+1,cap)
    return Vt[:r].to(dev)

# feature-coherence measured uniformly: mean over batch of ||Pfeat z_hat||
def measure_mu(net,idx,Pfeat):
    net.eval(); zs=[]; n=0
    with torch.no_grad():
        for x,_ in batches(Xtr,Ytr,idx,256,False):
            zs.append(net(x)); n+=x.shape[0]
            if n>1000: break
    Z=torch.cat(zs,0); Z=Z/(Z.norm(dim=1,keepdim=True)+1e-9)
    return (Pfeat@Z.t()).norm(dim=0).mean().item()

def run(method,seed,lam=0.0):
    torch.manual_seed(seed); random.seed(seed)
    net=ResNetCIFAR().to(dev); head=nn.Linear(FEAT,NCLS).to(dev)
    opt=torch.optim.SGD(list(net.parameters())+list(head.parameters()),lr=0.05,momentum=0.9,weight_decay=5e-4)
    tr=[task_idx(Ytr,t) for t in range(args.ntask)]; te=[task_idx(Yte,t) for t in range(args.ntask)]
    cof=[list(range(t*PER,(t+1)*PER)) for t in range(args.ntask)]
    acc_after=[]; Pfeat=None; mu_log=[]; gmem=[]
    for t in range(args.ntask):
        cls=cof[t]
        for ep in range(args.epochs):
            net.train()
            for x,y in batches(Xtr,Ytr,tr[t]):
                opt.zero_grad(); z=net(x); loss=F.cross_entropy(head(z),y)
                if method=='OURSFEAT' and Pfeat is not None:
                    zc=z/(z.norm(dim=1,keepdim=True)+1e-6)
                    align=(Pfeat@zc.t())            # (r,B) feature-subspace coherence
                    loss=loss+lam*align.pow(2).sum(0).mean()
                loss.backward()
                if method in ('GPM','OURSFEAT') and Pfeat is not None:
                    g=head.weight.grad; head.weight.grad=g-((g@Pfeat.t())@Pfeat)
                torch.nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),5.0)
                opt.step()
        acc_after.append(evaluate_ti(net,head,Xte,Yte,te[t],cls))
        if method in ('GPM','OURSFEAT'):
            B=feature_basis(net,tr[t]); Pfeat=B if Pfeat is None else torch.cat([Pfeat,B],0)
            q,_=torch.linalg.qr(Pfeat.t()); Pfeat=q.t()[:min(Pfeat.shape[0],FEAT)]
            mu_log.append(measure_mu(net,tr[t],Pfeat))
    final=[evaluate_ti(net,head,Xte,Yte,te[t],cof[t]) for t in range(args.ntask)]
    avg=sum(final)/len(final)
    forget=sum(max(0.0,acc_after[t]-final[t]) for t in range(args.ntask-1))/(args.ntask-1)
    mu=sum(mu_log)/len(mu_log) if mu_log else float('nan')
    return avg,forget,mu

import numpy as np
print(f'DIAG CIFAR100-ResNet18 feat-penalty seeds={args.seeds} epochs={args.epochs} ntask={args.ntask} lams={LAMS} dev={dev}')
configs=[('FINETUNE',0.0),('GPM',0.0)]+[('OURSFEAT',l) for l in LAMS]
agg={}
for name,lam in configs:
    key=f'{name}' if name!='OURSFEAT' else f'OURSFEAT(lam={lam})'
    A=[];Fg=[];Mu=[]
    for s in range(args.seeds):
        t0=time.time(); a,f,mu=run(name,s,lam); A.append(a);Fg.append(f);Mu.append(mu)
        print(f'{key:18} seed{s} avg={a:.4f} forget={f:.4f} mu={mu:.4f} [{time.time()-t0:.0f}s]',flush=True)
    agg[key]=(np.array(A),np.array(Fg),np.array(Mu))
print('='*72)
print(f'{"config":20}{"avg_acc":>16}{"forget":>16}{"mu":>16}')
for key,(A,Fg,Mu) in agg.items():
    mus='---' if np.isnan(Mu).all() else f'{Mu.mean():.3f}+-{Mu.std():.3f}'
    print(f'{key:20}{A.mean():.4f}+-{A.std():.4f}   {Fg.mean():.4f}+-{Fg.std():.4f}   {mus:>14}')
print('='*72); print('Done.')
