import argparse, time, random
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

p = argparse.ArgumentParser()
p.add_argument('--data', default='./data')
p.add_argument('--epochs', type=int, default=3)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--lam', type=float, default=0.5)
args = p.parse_args()

torch.manual_seed(args.seed); random.seed(args.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={dev} torch={torch.__version__} lam={args.lam} epochs={args.epochs} seed={args.seed}')

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_full = datasets.MNIST(args.data, train=True, download=False, transform=tf)
test_full  = datasets.MNIST(args.data, train=False, download=False, transform=tf)

TASKS = [(0,1),(2,3),(4,5),(6,7),(8,9)]
def subset(ds, classes):
    idx = [i for i,(_,y) in enumerate(ds) if y in classes]
    return torch.utils.data.Subset(ds, idx)
train_sets = [subset(train_full, c) for c in TASKS]
test_sets  = [subset(test_full, c) for c in TASKS]

class Backbone(nn.Module):
    def __init__(self, feat=128):
        super().__init__()
        self.c1 = nn.Conv2d(1,32,3,padding=1); self.c2 = nn.Conv2d(32,64,3,padding=1)
        self.fc = nn.Linear(64*7*7, feat)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.c1(x)),2)
        x = F.max_pool2d(F.relu(self.c2(x)),2)
        return F.relu(self.fc(x.flatten(1)))

FEAT=128; NCLS=10
def loader(ds, bs=128, shuffle=True):
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=2)

def evaluate_ti(net, head, ds, classes):
    net.eval(); correct=total=0
    cls=torch.tensor(classes, device=dev)
    with torch.no_grad():
        for x,y in loader(ds,256,False):
            x,y=x.to(dev),y.to(dev)
            logits = head(net(x))[:, cls]
            pred = cls[logits.argmax(1)]
            correct += (pred==y).sum().item(); total += y.numel()
    return correct/total

def eval_all_ti(net, head, upto):
    return [evaluate_ti(net, head, test_sets[t], TASKS[t]) for t in range(upto+1)]

def feature_basis(net, ds, energy=0.95, cap=40):
    net.eval(); feats=[]; n=0
    with torch.no_grad():
        for x,_ in loader(ds,256,False):
            feats.append(net(x.to(dev)).cpu()); n+=x.shape[0]
            if n>2000: break
    M=torch.cat(feats,0); M=M-M.mean(0,keepdim=True)
    M=torch.nan_to_num(M)
    U,S,Vt=torch.linalg.svd(M,full_matrices=False)
    e=torch.cumsum(S**2,0)/(S**2).sum()
    r=min(int((e<energy).sum().item())+1,cap)
    return Vt[:r].to(dev)

def run(method):
    torch.manual_seed(args.seed)
    net=Backbone(FEAT).to(dev); head=nn.Linear(FEAT,NCLS).to(dev)
    opt=torch.optim.SGD(list(net.parameters())+list(head.parameters()),lr=0.05,momentum=0.9)
    acc_after=[]; Pfeat=None; mu_log=[]; gmem=[]
    for t in range(len(TASKS)):
        ds_tr=train_sets[t]; cls=TASKS[t]
        for ep in range(args.epochs):
            net.train()
            for x,y in loader(ds_tr):
                x,y=x.to(dev),y.to(dev)
                opt.zero_grad()
                z=net(x); loss=F.cross_entropy(head(z),y)
                if method=='OURS' and Pfeat is not None:
                    zc = z / (z.norm(dim=1,keepdim=True)+1e-6)
                    align=(Pfeat @ zc.t())
                    loss=loss + args.lam*align.pow(2).sum(0).mean()
                loss.backward()
                if method in ('GPM','OURS') and Pfeat is not None:
                    g=net.fc.weight.grad
                    proj=(Pfeat.t()@(Pfeat@g))
                    net.fc.weight.grad=g-proj
                if method=='OGD' and len(gmem)>0:
                    g=net.fc.weight.grad.flatten()
                    for gm in gmem: g=g-(g@gm)/(gm@gm+1e-12)*gm
                    net.fc.weight.grad=g.view_as(net.fc.weight.grad)
                torch.nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),5.0)
                opt.step()
        acc_after.append(evaluate_ti(net,head,test_sets[t],cls))
        if method in ('GPM','OURS'):
            B=feature_basis(net,ds_tr)
            Pfeat=B if Pfeat is None else torch.cat([Pfeat,B],0)
            q,_=torch.linalg.qr(Pfeat.t()); Pfeat=q.t()[:min(Pfeat.shape[0],FEAT)]
            with torch.no_grad():
                zs=[]; n=0
                for x,_ in loader(ds_tr,256,False):
                    zs.append(net(x.to(dev))); n+=x.shape[0]
                    if n>1000: break
                Z=torch.cat(zs,0); Z=Z/(Z.norm(dim=1,keepdim=True)+1e-9)
                mu=(Pfeat@Z.t()).norm(dim=0).mean().item()
                mu_log.append(mu)
        if method=='OGD':
            net.eval()
            for x,y in loader(ds_tr,256,False):
                x,y=x.to(dev),y.to(dev)
                net.zero_grad(); head.zero_grad()
                F.cross_entropy(head(net(x)),y).backward()
                gmem.append(net.fc.weight.grad.flatten().detach().clone()); break
    final=eval_all_ti(net,head,len(TASKS)-1)
    avg=sum(final)/len(final)
    forget=sum(max(0.0,acc_after[t]-final[t]) for t in range(len(TASKS)-1))/(len(TASKS)-1)
    mu=sum(mu_log)/len(mu_log) if mu_log else float('nan')
    return avg,forget,mu,final

print('='*72)
print(f'{"method":9}{"avg_acc":>9}{"forget":>9}{"mu_coh":>9}   per-task-acc')
res={}
for m in ['FINETUNE','OGD','GPM','OURS']:
    t0=time.time(); a,f,mu,fin=run(m); res[m]=(a,f,mu,fin)
    print(f'{m:9}{a:9.4f}{f:9.4f}{mu:9.4f}   {[round(x,3) for x in fin]}  [{time.time()-t0:.0f}s]')
print('='*72)
print(f'acc OURS {res["OURS"][0]:.4f} GPM {res["GPM"][0]:.4f} OGD {res["OGD"][0]:.4f} FT {res["FINETUNE"][0]:.4f}')
print(f'forget OURS {res["OURS"][1]:.4f} GPM {res["GPM"][1]:.4f} FT {res["FINETUNE"][1]:.4f}')
print(f'mu OURS {res["OURS"][2]:.4f} GPM {res["GPM"][2]:.4f} reduced={res["OURS"][2]<res["GPM"][2]}')
print('Done.')
