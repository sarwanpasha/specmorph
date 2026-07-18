import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, time
from torchvision import datasets, transforms
torch.manual_seed(0); np.random.seed(0)
dev='cuda' if torch.cuda.is_available() else 'cpu'
tf=transforms.Compose([transforms.ToTensor()])
tr=datasets.MNIST('./data',train=True,download=False,transform=tf)
te=datasets.MNIST('./data',train=False,download=False,transform=tf)
Xtr=tr.data.float().view(-1,784)/255.0; Ytr=tr.targets
Xte=te.data.float().view(-1,784)/255.0; Yte=te.targets
Xtr=Xtr.to(dev); Ytr=Ytr.to(dev); Xte=Xte.to(dev); Yte=Yte.to(dev)
def mlp():
    return nn.Sequential(nn.Linear(784,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,10)).to(dev)
def perm(seed):
    g=np.random.RandomState(seed); return torch.tensor(g.permutation(784),device=dev)
# AGGRESSIVE drift: fresh full permutation every micro-step (fast, strong shift)
# initial distribution = identity permutation; measure retention on it as stream drifts far
NSTEP=40; BATCH=256; STEPS_PER=6
P0=torch.arange(784,device=dev)  # initial dist = identity
def evalret(net,P):
    net.eval()
    with torch.no_grad():
        idx=torch.randperm(Xte.size(0),device=dev)[:2000]
        xb=Xte[idx][:,P]; yb=Yte[idx]
        acc=(net(xb).argmax(1)==yb).float().mean().item()
    net.train(); return acc
def run(method,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net=mlp(); opt=torch.optim.SGD(net.parameters(),lr=0.05)
    feats=[]  # COBonline: EMA of activation covariance top-dirs (no task boundary)
    ema=None; lam=0.0
    curve=[]
    for t in range(NSTEP):
        # aggressive: totally new permutation each step -> strong drift
        Pt=perm(1000+t*97+seed)
        for s in range(STEPS_PER):
            bi=torch.randint(0,Xtr.size(0),(BATCH,),device=dev)
            xb=Xtr[bi][:,Pt]; yb=Ytr[bi]
            opt.zero_grad()
            h1=F.relu(net[0](xb))
            out=net(xb)
            loss=F.cross_entropy(out,yb)
            if method=='COBonline' and ema is not None:
                # coherence penalty: discourage moving in stored principal input dirs
                # penalize alignment of first-layer weight update directions with ema basis
                W=net[0].weight  # 256x784
                proj=W@ema  # 256xk
                loss=loss+0.05*(proj**2).mean()
            loss.backward()
            if method=='GPMoracle':
                # oracle: needs boundary snapshot; approximate by projecting grad out of ema
                if ema is not None:
                    g=net[0].weight.grad
                    g-= (g@ema)@ema.t()
                    net[0].weight.grad.copy_(g)
            opt.step()
            # update EMA covariance basis from current batch activations (task-free)
            if method in ('COBonline','GPMoracle'):
                with torch.no_grad():
                    xc=xb-xb.mean(0,keepdim=True)
                    cov=xc.t()@xc/xb.size(0)
                    U,S,_=torch.linalg.svd(cov)
                    newb=U[:,:20]
                    ema=newb if ema is None else newb
        curve.append(round(evalret(net,P0),3))
    return curve
methods=['Naive','GPMoracle','COBonline']
for seed in range(3):
    for m in methods:
        c=run(m,seed)
        print(f"RESULT seed={seed} method={m} final_retain={c[-1]} min_retain={min(c)} curve={c}",flush=True)
print("ALLDONE2",flush=True)
