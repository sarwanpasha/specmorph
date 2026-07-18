import os,sys,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
dev='cuda' if torch.cuda.is_available() else 'cpu'
PROJ=os.environ['PROJ']
from torchvision import datasets,transforms
tr=transforms.Compose([transforms.ToTensor()])
ds=datasets.MNIST(PROJ+'/data',train=True,download=False,transform=tr)
dt=datasets.MNIST(PROJ+'/data',train=False,download=False,transform=tr)
Xtr=(ds.data.float().view(-1,784)/255.0).to(dev); Ytr=ds.targets.clone().to(dev)
Xte=(dt.data.float().view(-1,784)/255.0).to(dev); Yte=dt.targets.clone().to(dev)
class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.f1=nn.Linear(784,256,bias=False); s.f2=nn.Linear(256,256,bias=False); s.head=nn.Linear(256,10,bias=False)
    def feat(s,x): return torch.relu(s.f2(torch.relu(s.f1(x))))
    def forward(s,x): return s.head(s.feat(x))
def ev(net,perm):
    net.eval()
    with torch.no_grad(): p=net(Xte[:,perm]).argmax(1); return float((p==Yte).float().mean())
# Build a TASK-FREE drifting stream: permutation drifts by a few swaps every micro-step, NO discrete boundaries.
def make_stream(nsteps,seed):
    rng=np.random.RandomState(seed); base=np.arange(784); perms=[]
    cur=base.copy()
    for k in range(nsteps):
        # smooth drift: swap a small fraction of positions each step
        for _ in range(6):
            i,j=rng.randint(784),rng.randint(784); cur[i],cur[j]=cur[j],cur[i]
        perms.append(torch.tensor(cur.copy(),device=dev))
    return perms
def feat_basis(net,perm,cap=40):
    with torch.no_grad():
        idx=torch.randperm(Xtr.shape[0],device=dev)[:1024]; H=net.feat(Xtr[idx][:,perm])
    U,S,_=torch.linalg.svd(H.t(),full_matrices=False)
    k=int((torch.cumsum(S**2,0)/S.pow(2).sum()<0.99).sum())+1; return U[:,:min(k,cap)].contiguous()
def ortho(mats):
    M=torch.cat(mats,1); Q,_=torch.linalg.qr(M); return Q
def batch(perm,bs=128):
    idx=torch.randperm(Xtr.shape[0],device=dev)[:bs]; return Xtr[idx][:,perm],Ytr[idx]
def run(method,nsteps,perms,perm_eval0,lam=1.0,boundary_every=0,seed=0):
    torch.manual_seed(seed); np.random.seed(seed); net=MLP().to(dev)
    opt=torch.optim.SGD(net.parameters(),lr=0.1,momentum=0.9); ce=nn.CrossEntropyLoss()
    tb=[]; retain=[]  # retention on the INITIAL distribution over time
    Bprot=None
    for k in range(nsteps):
        perm=perms[k]; net.train()
        # GPM variants only snapshot a basis when a boundary is DECLARED
        if method=='GPMoracle' and boundary_every>0 and k>0 and k%boundary_every==0:
            tb.append(feat_basis(net,perms[k-1])); Bprot=ortho(tb)
        # GPMnobound: no boundaries exist -> never snapshots -> Bprot stays None (degenerates to naive)
        for _ in range(4):
            x,y=batch(perm); opt.zero_grad(); out=net(x); loss=ce(out,y)
            if method=='COBonline' and k>0:
                Wr=net.head.weight; Ur,_=torch.linalg.qr(Wr.t())
                fb=net.feat(x); fb=fb-fb.mean(0,keepdim=True)
                Uf,_=torch.linalg.qr(fb.t()[:,:min(fb.shape[0],64)])
                coh=torch.linalg.svdvals(Uf.t()@Ur)[0]; loss=loss+lam*coh
            loss.backward()
            if method in ('GPMoracle',) and Bprot is not None:
                g=net.f2.weight.grad; net.f2.weight.grad[:]=g-(Bprot@(Bprot.t()@g.t())).t()
            opt.step()
        if k%5==0: retain.append(ev(net,perm_eval0))
    final_retain=ev(net,perm_eval0)
    return final_retain,retain
if __name__=='__main__':
    nsteps=60
    for seed in range(3):
        perms=make_stream(nsteps,seed); perm0=perms[0]
        for method in ['Naive','GPMnobound','GPMoracle','COBonline']:
            be=10 if method=='GPMoracle' else 0
            fr,curve=run(method,nsteps,perms,perm0,lam=1.0,boundary_every=be,seed=seed)
            print(f'RESULT seed={seed} method={method} boundary_every={be} final_retain_on_init={fr:.4f} retain_curve={[round(c,3) for c in curve]}',flush=True)
    print('ALLDONE',flush=True)
