import os,sys,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
dev='cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark=True
# permuted-MNIST from torchvision-cached tensors if present, else synthetic-but-structured fallback
PROJ=os.environ['PROJ']
from torchvision import datasets,transforms
tr=transforms.Compose([transforms.ToTensor()])
ds=datasets.MNIST(PROJ+'/data',train=True,download=False,transform=tr)
dt=datasets.MNIST(PROJ+'/data',train=False,download=False,transform=tr)
Xtr=ds.data.float().view(-1,784)/255.0; Ytr=ds.targets.clone()
Xte=dt.data.float().view(-1,784)/255.0; Yte=dt.targets.clone()
Xtr=Xtr.to(dev);Ytr=Ytr.to(dev);Xte=Xte.to(dev);Yte=Yte.to(dev)
class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.f1=nn.Linear(784,256,bias=False); s.f2=nn.Linear(256,256,bias=False); s.head=nn.Linear(256,10,bias=False)
    def feat(s,x): return torch.relu(s.f2(torch.relu(s.f1(x))))
    def forward(s,x): return s.head(s.feat(x))
def ortho(mats):
    if len(mats)==0: return None
    M=torch.cat(mats,1); Q,_=torch.linalg.qr(M); return Q
def feat_basis(net,perm,cap=40):
    with torch.no_grad():
        idx=torch.randperm(Xtr.shape[0],device=dev)[:1024]; H=net.feat(Xtr[idx][:,perm])
    U,S,_=torch.linalg.svd(H.t(),full_matrices=False)
    k=int((torch.cumsum(S**2,0)/S.pow(2).sum()<0.99).sum())+1; return U[:,:min(k,cap)].contiguous()
def batches(X,Y,bs=128):
    idx=torch.randperm(X.shape[0],device=dev)
    for i in range(0,len(idx),bs): j=idx[i:i+bs]; yield X[j],Y[j]
def ev(net,perm):
    net.eval()
    with torch.no_grad(): p=net(Xte[:,perm]).argmax(1); return float((p==Yte).float().mean())
def run(method,ntask,perms,lam=1.0,epochs=1,seed=0):
    torch.manual_seed(seed); np.random.seed(seed); net=MLP().to(dev)
    tb=[]; acc_after=[]; per_task_time=[]; stored_floats=[]
    for t in range(ntask):
        perm=perms[t]; t0=time.time()
        Bprot=None
        if method=='GPM':
            Bprot=ortho([tb[s] for s in range(t) if tb[s] is not None])
        opt=torch.optim.SGD(net.parameters(),lr=0.1,momentum=0.9); ce=nn.CrossEntropyLoss(); net.train()
        for ep in range(epochs):
            for x,y in batches(Xtr[:,perm],Ytr):
                opt.zero_grad(); out=net(x); loss=ce(out,y)
                if method=='COBpen' and t>0:
                    # differentiable coherence penalty: align current feat-subspace away from prior read-out rows
                    Wr=net.head.weight  # (10,256) current read-out
                    Ur,_=torch.linalg.qr(Wr.t())  # basis of read-out row space, computed on the fly, NO growing store
                    fb=net.feat(x); fb=fb-fb.mean(0,keepdim=True)
                    # coherence = top singular value of alignment between new-feature dirs and read-out basis
                    Uf,_=torch.linalg.qr(fb.t()[:, :min(fb.shape[0],64)])
                    coh=torch.linalg.svdvals(Uf.t()@Ur)[0]
                    loss=loss+lam*coh
                loss.backward()
                if method=='GPM' and Bprot is not None:
                    g=net.f2.weight.grad; net.f2.weight.grad[:]=g-(Bprot@(Bprot.t()@g.t())).t()
                opt.step()
        tb.append(feat_basis(net,perm))
        acc_after.append(ev(net,perm)); per_task_time.append(time.time()-t0)
        if method=='GPM':
            Bp=ortho([tb[s] for s in range(t+1)]); stored_floats.append(Bp.numel())
        else:
            stored_floats.append(0)  # penalty variant stores no cross-task basis
    finals=[ev(net,perms[t]) for t in range(ntask)]
    avg=float(np.mean(finals)); fg=float(np.mean([max(0.,acc_after[t]-finals[t]) for t in range(ntask-1)]))
    return avg,fg,stored_floats,per_task_time
if __name__=='__main__':
    ntask=int(sys.argv[1]) if len(sys.argv)>1 else 20
    rng=np.random.RandomState(999); perms=[torch.arange(784,device=dev)]+[torch.tensor(rng.permutation(784),device=dev) for _ in range(ntask-1)]
    for method in ['GPM','COBpen']:
        for seed in range(3):
            avg,fg,sf,pt=run(method,ntask,perms,lam=1.0,epochs=1,seed=seed)
            print(f'RESULT method={method} seed={seed} ntask={ntask} avg={avg:.4f} forget={fg:.4f} stored_floats_final={sf[-1]} stored_bytes_final={sf[-1]*4} time_task0={pt[0]:.3f} time_taskLast={pt[-1]:.3f} total_time={sum(pt):.2f}',flush=True)
            print(f'  stored_curve={sf}',flush=True)
            print(f'  time_curve={[round(x,3) for x in pt]}',flush=True)
    print('ALLDONE',flush=True)
