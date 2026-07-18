import argparse, time, math, random
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

p = argparse.ArgumentParser()
p.add_argument('--data', default='./data')
p.add_argument('--epochs', type=int, default=3)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--lam', type=float, default=1.0)  # mu-coherence penalty weight
args = p.parse_args()

torch.manual_seed(args.seed); random.seed(args.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={dev} torch={torch.__version__}')

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_full = datasets.MNIST(args.data, train=True, download=True, transform=tf)
test_full  = datasets.MNIST(args.data, train=False, download=True, transform=tf)

# 5 tasks: {0,1},{2,3},{4,5},{6,7},{8,9}
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
        x = x.flatten(1)
        return F.relu(self.fc(x))

FEAT=128; NCLS=10

def loader(ds, bs=128, shuffle=True):
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=2)

def evaluate(net, head, ds):
    net.eval()
    correct=total=0
    with torch.no_grad():
        for x,y in loader(ds, 256, False):
            x,y = x.to(dev), y.to(dev)
            logits = head(net(x))
            correct += (logits.argmax(1)==y).sum().item(); total += y.numel()
    return correct/total

def eval_all(net, head, upto):
    return [evaluate(net, head, test_sets[t]) for t in range(upto+1)]

# ---- feature-subspace projector shared by GPM and OURS ----
def feature_basis(net, ds, energy=0.97, cap=60):
    net.eval(); feats=[]
    with torch.no_grad():
        for x,_ in loader(ds,256,False):
            feats.append(net(x.to(dev)).cpu())
            if sum(f.shape[0] for f in feats) > 2000: break
    F_ = torch.cat(feats,0)                 # (N, FEAT)
    F_ = F_ - F_.mean(0, keepdim=True)
    U,S,_ = torch.linalg.svd(F_, full_matrices=False)
    e = torch.cumsum(S**2,0)/ (S**2).sum()
    r = int((e<energy).sum().item())+1; r=min(r,cap)
    _,_,Vt = torch.linalg.svd(F_, full_matrices=False)
    return Vt[:r].to(dev)                    # (r, FEAT) rows span old feature subspace

def run(method):
    torch.manual_seed(args.seed)
    net = Backbone(FEAT).to(dev)
    head = nn.Linear(FEAT, NCLS).to(dev)
    opt = torch.optim.SGD(list(net.parameters())+list(head.parameters()), lr=0.05, momentum=0.9)
    acc_after = []          # acc on each task right after learning it
    Pfeat = None            # (r,FEAT) old feature subspace basis (GPM/OURS)
    mu_log = []
    for t,(ds_tr) in enumerate(train_sets):
        cls = TASKS[t]
        for ep in range(args.epochs):
            net.train()
            for x,y in loader(ds_tr):
                x,y = x.to(dev), y.to(dev)
                opt.zero_grad()
                z = net(x)
                logits = head(z)
                loss = F.cross_entropy(logits, y)
                if method=='OURS' and Pfeat is not None:
                    # mu-coherence penalty: penalize how strongly the NEW readout rows
                    # (for current classes) align with the OLD maintained feature subspace.
                    Wc = head.weight[list(cls)]            # (2,FEAT) new readout rows
                    coh = (Wc @ Pfeat.t())                 # projection coeffs onto old subspace
                    loss = loss + args.lam * (coh.pow(2).sum() / Wc.shape[0])
                loss.backward()
                if method in ('GPM','OURS') and Pfeat is not None:
                    # project the readout-weight gradient orthogonal to old feature subspace
                    g = head.weight.grad                    # (NCLS,FEAT)
                    proj = (g @ Pfeat.t()) @ Pfeat          # component in old subspace
                    head.weight.grad = g - proj
                if method=='OGD' and len(g_mem)>0:
                    gv = torch.cat([head.weight.grad.flatten(), head.bias.grad.flatten()])
                    for gm in g_mem:
                        gv = gv - (gv@gm)/(gm@gm+1e-12)*gm
                    n1=head.weight.numel()
                    head.weight.grad = gv[:n1].view_as(head.weight)
                    head.bias.grad   = gv[n1:].view_as(head.bias)
                opt.step()
        acc_after.append(evaluate(net, head, test_sets[t]))
        # consolidate: update old feature subspace and OGD memory
        if method in ('GPM','OURS'):
            Bnew = feature_basis(net, ds_tr)
            Pfeat = Bnew if Pfeat is None else torch.cat([Pfeat,Bnew],0)
            # keep it a proper orthonormal-ish basis via QR on rows
            q,_ = torch.linalg.qr(Pfeat.t()); Pfeat = q.t()[:min(Pfeat.shape[0],FEAT)]
            # measure mu-coherence for this task's readout vs maintained subspace
            with torch.no_grad():
                Wc = head.weight[list(cls)]
                mu = (Wc @ Pfeat.t()).norm(dim=1).max().item() / (Wc.norm(dim=1).max().item()+1e-12)
                mu_log.append(mu)
        if method=='OGD':
            net.eval()
            for x,y in loader(ds_tr,256,False):
                x,y=x.to(dev),y.to(dev)
                head.zero_grad(); net.zero_grad()
                F.cross_entropy(head(net(x)),y).backward()
                gv=torch.cat([head.weight.grad.flatten(),head.bias.grad.flatten()]).detach()
                g_mem.append(gv); break
    final = eval_all(net, head, len(TASKS)-1)
    avg_acc = sum(final)/len(final)
    forget = sum(max(0.0, acc_after[t]-final[t]) for t in range(len(TASKS)-1))/(len(TASKS)-1)
    mu_mean = sum(mu_log)/len(mu_log) if mu_log else float('nan')
    return avg_acc, forget, mu_mean, final

g_mem = []
print('='*70)
print(f'{"method":8} {"avg_acc":>9} {"forgetting":>11} {"mu_coh":>9}')
results={}
for m in ['FINETUNE','OGD','GPM','OURS']:
    globals()['g_mem'] = []
    t0=time.time()
    aa,fg,mu,final = run(m)
    results[m]=(aa,fg,mu,final)
    print(f'{m:8} {aa:9.4f} {fg:11.4f} {mu:9.4f}   [{time.time()-t0:.0f}s]  per-task={[round(x,3) for x in final]}')
print('='*70)
# Theorem-2 check: OURS penalizes mu; verify OURS mu < GPM mu AND OURS forgetting <= GPM
mo=results['OURS'][2]; mg=results['GPM'][2]
print(f'mu(OURS)={mo:.4f}  mu(GPM)={mg:.4f}  -> coherence reduced: {mo < mg}')
print(f'forget(OURS)={results["OURS"][1]:.4f} <= forget(GPM)={results["GPM"][1]:.4f}: {results["OURS"][1] <= results["GPM"][1]+1e-4}')
print(f'acc(OURS)={results["OURS"][0]:.4f} >= acc(FINETUNE)={results["FINETUNE"][0]:.4f}: {results["OURS"][0] >= results["FINETUNE"][0]}')
print('Done.')
