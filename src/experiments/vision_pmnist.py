import argparse, time, random, math
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
p=argparse.ArgumentParser()
p.add_argument('--data', default='./data')
p.add_argument('--epochs', type=int, default=2)
p.add_argument('--lam', type=float, default=0.2)
p.add_argument('--seeds', type=int, default=5)
p.add_argument('--ntasks', type=int, default=5)
args=p.parse_args()
dev='cuda' if torch.cuda.is_available() else 'cpu'
tf=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
train_full=datasets.MNIST(args.data, train=True, download=False, transform=tf)
test_full=datasets.MNIST(args.data, train=False, download=False, transform=tf)
# preload all images into flat tensors once
def stack(ds):
    xs=torch.stack([ds[i][0].view(-1) for i in range(len(ds))])
    ys=torch.tensor([ds[i][1] for i in range(len(ds))])
    return xs, ys
print('loading MNIST into memory...', flush=True)
TRX,TRY=stack(train_full); TEX,TEY=stack(test_full)
print('loaded', TRX.shape, TEX.shape, flush=True)
FEAT=256; NCLS=10; DIN=784
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1=nn.Linear(DIN,FEAT); self.f2=nn.Linear(FEAT,FEAT); self.head=nn.Linear(FEAT,NCLS)
    def feat(self,x):
        h=F.relu(self.f1(x)); h=F.relu(self.f2(h)); return h
    def forward(self,x):
        return self.head(self.feat(x))
def batches(X,Y,bs,shuffle=True):
    n=X.shape[0]; idx=torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0,n,bs):
        j=idx[i:i+bs]; yield X[j], Y[j]
def evaluate(net, perm):
    net.eval(); correct=0; tot=0
    with torch.no_grad():
        for x,y in batches(TEX,TEY,1000,False):
            x=x[:,perm].to(dev); y=y.to(dev)
            pred=net(x).argmax(1)
            correct+=(pred==y).sum().item(); tot+=y.numel()
    return correct/tot
def feat_basis(net, perm, energy=0.97, cap=100):
    net.eval(); feats=[]; n=0
    with torch.no_grad():
        for x,_ in batches(TRX,TRY,1000,False):
            feats.append(net.feat(x[:,perm].to(dev)).cpu()); n+=x.shape[0]
            if n>3000: break
    M=torch.cat(feats,0); M=M-M.mean(0,keepdim=True); M=torch.nan_to_num(M)
    U,S,Vt=torch.linalg.svd(M,full_matrices=False)
    e=torch.cumsum(S**2,0)/(S**2).sum()
    r=min(int((e<energy).sum().item())+1,cap)
    return Vt[:r].to(dev)
def run(method, seed):
    torch.manual_seed(seed); random.seed(seed)
    perms=[torch.randperm(DIN) for _ in range(args.ntasks)]
    net=MLP().to(dev)
    opt=torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    acc_after=[]; P=None; mu_log=[]; gmem=[]
    for t in range(args.ntasks):
        perm=perms[t]
        for ep in range(args.epochs):
            net.train()
            for x,y in batches(TRX,TRY,128):
                x=x[:,perm].to(dev); y=y.to(dev)
                opt.zero_grad()
                h=net.feat(x); logits=net.head(h); loss=F.cross_entropy(logits,y)
                if method=='OURS' and P is not None:
                    hc=h/(h.norm(dim=1,keepdim=True)+1e-6)
                    loss=loss+args.lam*(P@hc.t()).pow(2).sum(0).mean()
                loss.backward()
                if method in ('GPM','OURS') and P is not None:
                    g=net.f2.weight.grad
                    net.f2.weight.grad=g-(P.t()@(P@g))
                if method=='OGD' and len(gmem)>0:
                    g=net.head.weight.grad.flatten()
                    for gm in gmem: g=g-(g@gm)/(gm@gm+1e-12)*gm
                    net.head.weight.grad=g.view_as(net.head.weight.grad)
                torch.nn.utils.clip_grad_norm_(net.parameters(),5.0)
                opt.step()
        acc_after.append(evaluate(net,perm))
        if method in ('GPM','OURS'):
            B=feat_basis(net,perm)
            P=B if P is None else torch.cat([P,B],0)
            q,_=torch.linalg.qr(P.t()); P=q.t()[:min(P.shape[0],FEAT)]
            with torch.no_grad():
                hs=[]; n=0
                for x,_ in batches(TRX,TRY,1000,False):
                    hs.append(net.feat(x[:,perm].to(dev))); n+=x.shape[0]
                    if n>2000: break
                H=torch.cat(hs,0); H=H/(H.norm(dim=1,keepdim=True)+1e-9)
                mu_log.append((P@H.t()).norm(dim=0).mean().item())
        if method=='OGD':
            net.eval()
            for x,y in batches(TRX,TRY,1000,False):
                x=x[:,perm].to(dev); y=y.to(dev)
                net.zero_grad(); F.cross_entropy(net(x),y).backward()
                gmem.append(net.head.weight.grad.flatten().detach().clone()); break
    final=[evaluate(net,perms[t]) for t in range(args.ntasks)]
    avg=sum(final)/len(final)
    forget=sum(max(0.0,acc_after[t]-final[t]) for t in range(args.ntasks-1))/(args.ntasks-1)
    mu=sum(mu_log)/len(mu_log) if mu_log else float('nan')
    return avg,forget,mu,sum(acc_after)/len(acc_after)
def mstd(v):
    m=sum(v)/len(v); s=math.sqrt(sum((x-m)**2 for x in v)/len(v)) if len(v)>1 else 0.0
    return m,s
print('PMNIST', 'seeds=%d'%args.seeds,'ntasks=%d'%args.ntasks,'lam=%.2f'%args.lam,'epochs=%d'%args.epochs,'dev='+dev, flush=True)
ME=['FINETUNE','OGD','GPM','OURS']
agg={m:{'avg':[],'f':[],'mu':[],'aa':[]} for m in ME}
for s in range(args.seeds):
    for m in ME:
        t0=time.time(); a,f,mu,aa=run(m,s)
        agg[m]['avg'].append(a); agg[m]['f'].append(f); agg[m]['aa'].append(aa)
        if not math.isnan(mu): agg[m]['mu'].append(mu)
        print(f'seed{s} {m:9} avg={a:.4f} forget={f:.4f} mu={mu:.4f} accAfter={aa:.4f} [{time.time()-t0:.0f}s]', flush=True)
print('='*72)
print(f'{"method":9}{"avg_acc":>18}{"forget":>18}{"mu":>14}')
for m in ME:
    am,asd=mstd(agg[m]['avg']); fm,fsd=mstd(agg[m]['f'])
    mus=(f'{mstd(agg[m]["mu"])[0]:.3f}+-{mstd(agg[m]["mu"])[1]:.3f}') if agg[m]['mu'] else '---'
    print(f'{m:9}{am:.4f}+-{asd:.4f}   {fm:.4f}+-{fsd:.4f}   {mus:>14}')
print('='*72); print('Done.')
