import argparse,time,math,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from torchvision import datasets,transforms
p=argparse.ArgumentParser()
p.add_argument('--ntasks',type=int,default=10)
p.add_argument('--epochs',type=int,default=1)
p.add_argument('--seeds',type=int,default=3)
p.add_argument('--cap',type=int,default=100)
p.add_argument('--energy',type=float,default=0.97)
args=p.parse_args()
dev='cuda' if torch.cuda.is_available() else 'cpu'

tf=transforms.Compose([transforms.ToTensor()])
tr=datasets.MNIST('./data',train=True,download=True,transform=None)
te=datasets.MNIST('./data',train=False,download=True,transform=None)
def stack(ds):
    X=(ds.data.float()/255.).view(-1,784); Y=ds.targets.clone(); return X,Y
TRX,TRY=stack(tr); TEX,TEY=stack(te)

class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.f1=nn.Linear(784,256); self.f2=nn.Linear(256,256); self.head=nn.Linear(256,10)
    def feat(self,x): return F.relu(self.f2(F.relu(self.f1(x))))
    def forward(self,x): return self.head(self.feat(x))

def batches(X,Y,bs,shuffle=True):
    idx=torch.randperm(len(X)) if shuffle else torch.arange(len(X))
    for i in range(0,len(X),bs):
        j=idx[i:i+bs]; yield X[j],Y[j]

def evaluate(net,perm):
    net.eval(); c=t=0
    with torch.no_grad():
        for x,y in batches(TEX,TEY,1000,False):
            x=x[:,perm].to(dev); y=y.to(dev)
            c+=(net(x).argmax(1)==y).sum().item(); t+=y.numel()
    return c/t

def feat_basis(net,perm,energy,cap):
    net.eval(); hs=[]; n=0
    with torch.no_grad():
        for x,_ in batches(TRX,TRY,256,False):
            hs.append(net.feat(x[:,perm].to(dev)).cpu()); n+=x.shape[0]
            if n>3000: break
    M=torch.cat(hs,0); M=M-M.mean(0,keepdim=True)
    U,S,Vt=torch.linalg.svd(M,full_matrices=False)
    e=torch.cumsum(S**2,0)/(S**2).sum(); r=min(int((e<energy).sum().item())+1,cap)
    return Vt[:r].to(dev)          # r x 256 orthonormal (task feature subspace)

def run(method,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net=Net().to(dev); opt=torch.optim.SGD(net.parameters(),lr=0.05,momentum=0.9)
    perms=[torch.randperm(784) for _ in range(args.ntasks)]
    P=None                          # maintained (protected) subspace, r x 256
    acc_after=[]
    mu_common=[]; mu_frame=[]       # two measurements
    Pref=None
    for t in range(args.ntasks):
        perm=perms[t]
        net.train()
        for ep in range(args.epochs):
            for x,y in batches(TRX,TRY,128):
                x=x[:,perm].to(dev); y=y.to(dev)
                opt.zero_grad(); F.cross_entropy(net(x),y).backward()
                if method in ('GPM',) and P is not None:
                    g=net.f2.weight.grad
                    net.f2.weight.grad=g-(P.t()@(P@g))     # project update off protected subspace
                opt.step()
        acc_after.append(evaluate(net,perm))
        Bnew=feat_basis(net,perm,args.energy,args.cap)     # this task's feature subspace
        if Pref is None: Pref=Bnew.clone()
        # (A) OLD common-reference mu: alignment of NEW task subspace with fixed task-0 basis
        mu_common.append(float((Pref@Bnew.t()).norm(dim=0).mean().item()))
        # (B) THEORY-FRAME mu: coherence of NEW task subspace with the MAINTAINED subspace P
        # (the quantity the projection mechanism actually drives). Undefined for t=0 (P empty).
        if P is not None:
            # ||P_maintained applied to new-task directions||: max singular value of P@Bnew^T
            mu_frame.append(float(torch.linalg.matrix_norm(P@Bnew.t(),ord=2).item()))
        # update maintained subspace (GPM/OURS grow it; here we track it for measurement for all)
        P = Bnew if P is None else torch.linalg.qr(torch.cat([P,Bnew],0).t())[0].t()[:min(P.shape[0]+Bnew.shape[0],args.cap)]
    final=[evaluate(net,perms[t]) for t in range(args.ntasks)]
    forget=sum(max(0.0,acc_after[t]-final[t]) for t in range(args.ntasks-1))/(args.ntasks-1)
    return sum(final)/len(final), forget, float(np.mean(mu_common)), float(np.mean(mu_frame))

for m in ['FINETUNE','GPM']:
    A=[];Fg=[];MC=[];MF=[]
    for s in range(args.seeds):
        a,f,mc,mf=run(m,s); A.append(a);Fg.append(f);MC.append(mc);MF.append(mf)
        print(f'{m} seed{s} acc {a:.4f} forget {f:.4f} mu_common {mc:.4f} mu_frame {mf:.4f}',flush=True)
    print(f'RESULT {m} acc {np.mean(A):.4f} forget {np.mean(Fg):.4f} mu_common {np.mean(MC):.4f}+-{np.std(MC):.4f} mu_frame {np.mean(MF):.4f}+-{np.std(MF):.4f}',flush=True)
