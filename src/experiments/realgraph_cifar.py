import argparse,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F,collections
from torchvision import datasets,transforms
from torchvision.models import resnet18
p=argparse.ArgumentParser()
p.add_argument('--data',default='./data'); p.add_argument('--epochs',type=int,default=5)
p.add_argument('--ntask',type=int,default=20); p.add_argument('--seeds',type=int,default=3)
p.add_argument('--energy',type=float,default=0.90); p.add_argument('--cap',type=int,default=40)
p.add_argument('--tau',type=float,default=-1.0)  # overlap threshold; -1 => adaptive (median)
args=p.parse_args(); dev='cuda' if torch.cuda.is_available() else 'cpu'
PER=100//args.ntask
tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071,0.4865,0.4409),(0.2673,0.2564,0.2762))])
tr=datasets.CIFAR100(args.data,train=True,download=False,transform=None)
te=datasets.CIFAR100(args.data,train=False,download=False,transform=None)
def pack(ds):
    X=torch.stack([tf(i) for i,_ in ds]); Y=torch.tensor([y for _,y in ds]); return X,Y
Xtr,Ytr=pack(tr); Xte,Yte=pack(te)
print('data',Xtr.shape,Xte.shape,'ntask',args.ntask,'PER',PER,flush=True)

def task_idx(Y,t):
    lo,hi=t*PER,(t+1)*PER; return ((Y>=lo)&(Y<hi)).nonzero(as_tuple=True)[0]
class Net(nn.Module):
    def __init__(self):
        super().__init__(); m=resnet18(weights=None)
        m.conv1=nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool=nn.Identity(); m.fc=nn.Identity(); self.body=m
    def forward(self,x): return F.relu(self.body(x))
def batches(X,Y,idx,bs=128,shuffle=True):
    idx=idx[torch.randperm(len(idx))] if shuffle else idx
    for i in range(0,len(idx),bs):
        j=idx[i:i+bs]; yield X[j].to(dev),Y[j].to(dev)
def evaluate_ti(net,head,Xs,Ys,idx,cls):
    net.eval(); c=t=0; ct=torch.tensor(cls,device=dev)
    with torch.no_grad():
        for x,y in batches(Xs,Ys,idx,256,False):
            lg=head(net(x))[:,ct]; pr=ct[lg.argmax(1)]; c+=(pr==y).sum().item(); t+=y.numel()
    return c/t
def feat_basis(net,idx,energy,cap):
    net.eval(); fs=[]; n=0
    with torch.no_grad():
        for x,_ in batches(Xtr,Ytr,idx,256,False):
            fs.append(net(x).cpu()); n+=x.shape[0]
            if n>2000: break
    M=torch.cat(fs,0); M=M-M.mean(0,keepdim=True); M=torch.nan_to_num(M)
    U,S,Vt=torch.linalg.svd(M,full_matrices=False)
    e=torch.cumsum(S**2,0)/(S**2).sum(); r=min(int((e<energy).sum().item())+1,cap)
    return Vt[:r].to(dev)  # r x 512 orthonormal rows

def subspace_overlap(Bi,Bj):
    # top principal cosine between rowspaces of Bi (ri x512), Bj (rj x512), both orthonormal rows
    M=Bi@Bj.T   # ri x rj
    return torch.linalg.svdvals(M)[0].item()  # largest principal cosine in [0,1]

def measure_graph(bases, tau):
    n=len(bases); W=np.zeros((n,n))
    ov=[]
    for i in range(n):
        for j in range(i+1,n):
            o=subspace_overlap(bases[i],bases[j]); W[i,j]=W[j,i]=o; ov.append(o)
    ov=np.array(ov)
    thr=np.median(ov) if tau<0 else tau
    E=[(i,j) for i in range(n) for j in range(i+1,n) if W[i,j]>=thr]
    # b1 = |E| - |V| + (#connected components)
    parent=list(range(n))
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    for i,j in E:
        ri,rj=find(i),find(j)
        if ri!=rj: parent[ri]=rj
    comps=len(set(find(k) for k in range(n)))
    b1=len(E)-n+comps
    return E,W,thr,b1,comps,ov

def ortho(rows):
    if not rows: return None
    Q,_=torch.linalg.qr(torch.cat(rows,0).T); return Q.T.contiguous()

def proj_out(g,B):
    if B is None: return g
    return g - (g@B.T)@B   # remove component in rowspace(B); B: m x512 orthonormal

def train_task(net,head,idx,epochs,Bprot):
    opt=torch.optim.SGD(list(net.parameters())+list(head.parameters()),lr=0.05,momentum=0.9,weight_decay=5e-4)
    ce=nn.CrossEntropyLoss(); net.train()
    for ep in range(epochs):
        for x,y in batches(Xtr,Ytr,idx):
            opt.zero_grad(); ce(head(net(x)),y).backward()
            if Bprot is not None and head.weight.grad is not None:
                head.weight.grad[:]=proj_out(head.weight.grad,Bprot)
            opt.step()

def train_task_trgp(net,head,idx,epochs,Bfull,Btrust,scale):
    opt=torch.optim.SGD(list(net.parameters())+list(head.parameters()),lr=0.05,momentum=0.9,weight_decay=5e-4)
    ce=nn.CrossEntropyLoss(); net.train()
    for ep in range(epochs):
        for x,y in batches(Xtr,Ytr,idx):
            opt.zero_grad(); ce(head(net(x)),y).backward()
            g=head.weight.grad
            if g is not None and Bfull is not None:
                gp=g-(g@Bfull.T)@Bfull
                if Btrust is not None: gp=gp+scale*((g@Btrust.T)@Btrust)
                head.weight.grad[:]=gp
            opt.step()


def run(method,seed,tau):
    torch.manual_seed(seed); np.random.seed(seed)
    net=Net().to(dev); head=nn.Linear(512,100).to(dev)
    tb=[None]*args.ntask; acc_after=[0.]*args.ntask
    # first pass to get a reference overlap scale: we build graph online using past bases
    edges_seen=[]
    for t in range(args.ntask):
        idx=task_idx(Ytr,t); cls=list(range(t*PER,(t+1)*PER))
        if method=='FINETUNE':
            Bprot=None
        elif method=='GPM':
            Bprot=ortho([tb[s] for s in range(t) if tb[s] is not None])
        elif method=='COBOUND':
            # measure overlap of a PRELIMINARY basis for task t vs each past task; protect only adjacent
            # get prelim basis by a quick 1-epoch warm probe is costly; instead use current net feats on task t
            prelim=feat_basis(net,idx,args.energy,args.cap)
            nbrs=[]
            for s in range(t):
                if tb[s] is None: continue
                o=subspace_overlap(prelim,tb[s])
                if o>=tau: nbrs.append(tb[s]); edges_seen.append((s,t,o))
            Bprot=ortho(nbrs)
        Btrust=None
        if method==chr(84)+chr(82)+chr(71)+chr(80):
            Bfull=ortho([tb[s] for s in range(t) if tb[s] is not None])
            prelim=feat_basis(net,idx,args.energy,args.cap); tr=[]
            for s in range(t):
                if tb[s] is None: continue
                if subspace_overlap(prelim,tb[s])>=tau: tr.append(tb[s])
            Btrust=ortho(tr); Bprot=Bfull
        if method==chr(84)+chr(82)+chr(71)+chr(80):
            train_task_trgp(net,head,idx,args.epochs,Bprot,Btrust,0.5)
        else:
            train_task(net,head,idx,args.epochs,Bprot)
        tb[t]=feat_basis(net,idx,args.energy,args.cap)
        acc_after[t]=evaluate_ti(net,head,Xte,Yte,task_idx(Yte,t),cls)
    finals=[evaluate_ti(net,head,Xte,Yte,task_idx(Yte,t),list(range(t*PER,(t+1)*PER))) for t in range(args.ntask)]
    avg=sum(finals)/len(finals)
    fg=sum(max(0.,acc_after[t]-finals[t]) for t in range(args.ntask-1))/(args.ntask-1)
    # measure final full graph from learned bases (post-hoc, for b1 report)
    E,W,thr,b1,comps,ov=measure_graph([b for b in tb if b is not None], tau if tau>0 else -1.0)
    return avg,fg,b1,thr,len(E),comps,edges_seen

if __name__=='__main__':
    tau=args.tau
    # If adaptive: first do a dry finetune run to learn bases, measure median overlap, set tau=median
    if tau<0:
        torch.manual_seed(0); np.random.seed(0)
        net0=Net().to(dev); head0=nn.Linear(512,100).to(dev); tb0=[]
        for t in range(args.ntask):
            idx=task_idx(Ytr,t); train_task(net0,head0,idx,args.epochs,None); tb0.append(feat_basis(net0,idx,args.energy,args.cap))
        E,W,thr,b1,comps,ov=measure_graph(tb0,-1.0)
        tau=float(thr)
        print(f'ADAPTIVE tau(median overlap)={tau:.4f} | dry-run graph: |E|={len(E)} b1={b1} comps={comps} ov[min/med/max]={ov.min():.3f}/{np.median(ov):.3f}/{ov.max():.3f}',flush=True)
    agg=collections.defaultdict(lambda:{'avg':[],'fg':[],'b1':[]})
    for seed in range(args.seeds):
        for m in ['FINETUNE','GPM','TRGP','COBOUND']:
            t0=time.time(); avg,fg,b1,thr,ne,comps,es=run(m,seed,tau)
            agg[m]['avg'].append(avg); agg[m]['fg'].append(fg); agg[m]['b1'].append(b1)
            print(f'seed{seed} {m:9} avg={avg:.4f} forget={fg:.4f} b1={b1} |E|={ne} comps={comps} [{time.time()-t0:.0f}s]',flush=True)
    print('==== SUMMARY realgraph CIFAR (ntask=%d epochs=%d seeds=%d tau=%.3f) ===='%(args.ntask,args.epochs,args.seeds,tau),flush=True)
    for m in ['FINETUNE','GPM','TRGP','COBOUND']:
        a=np.array(agg[m]['avg']); f=np.array(agg[m]['fg']); b=np.array(agg[m]['b1'])
        print(f'{m:10} avg={a.mean():.4f}+-{a.std():.3f} forget={f.mean():.4f}+-{f.std():.3f} b1={b.mean():.1f}',flush=True)
