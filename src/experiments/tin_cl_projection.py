import os, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4)
PROJ=os.environ['PROJ']; FE=PROJ+'/data/tin_vitfeat'
dev='cuda' if torch.cuda.is_available() else 'cpu'
NTASK=10; CPT=20  # 10 tasks, 20 classes/task
EPOCHS=15; LR=0.01; WD=0.0; BS=256

def load():
    tr=torch.load(FE+'/train.pt'); va=torch.load(FE+'/val.pt')
    return tr['feat'].float(), tr['label'], va['feat'].float(), va['label']

def task_masks(labels, t):
    lo=t*CPT; hi=lo+CPT
    return (labels>=lo)&(labels<hi)

class Head(nn.Module):
    def __init__(self, d, ncls):
        super().__init__(); self.fc=nn.Linear(d, ncls)
    def forward(self,x): return self.fc(x)

def evaluate(head, Xv, Yv, upto):
    head.eval(); accs=[]
    with torch.no_grad():
        for t in range(upto+1):
            m=task_masks(Yv,t)
            if m.sum()==0: accs.append(0.0); continue
            x=Xv[m].to(dev); y=Yv[m].to(dev)
            logit=head(x)
            # class-incremental: argmax over all seen classes
            logit=logit[:,:(upto+1)*CPT]
            pred=logit.argmax(1)
            accs.append((pred==y).float().mean().item())
    return accs

def proj_out(g, basis):
    # remove components of g lying in span(basis columns), basis: d x k orthonormal
    if basis is None or basis.shape[1]==0: return g
    return g - basis@(basis.t()@g)

def update_basis(basis, feats, thresh=0.97, cap=300):
    # feats: n x d ; collect principal directions of representation
    X=feats.t()  # d x n
    if basis is not None and basis.shape[1]>0:
        X = X - basis@(basis.t()@X)
    U,S,_=torch.linalg.svd(X, full_matrices=False)
    if S.numel()==0: return basis
    energy=torch.cumsum(S**2,0)/torch.sum(S**2)
    k=int((energy<thresh).sum().item())+1
    k=min(k, U.shape[1])
    newb=U[:,:k]
    if basis is None: nb=newb
    else: nb=torch.cat([basis,newb],1)
    if nb.shape[1]>cap: nb=nb[:,:cap]
    return nb

def run(method, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr,Ytr,Xv,Yv=load()
    d=Xtr.shape[1]; ncls=NTASK*CPT
    head=Head(d,ncls).to(dev)
    opt=torch.optim.SGD(head.parameters(), lr=LR, momentum=0.9, weight_decay=WD)
    basis=None
    acc_after=[]
    for t in range(NTASK):
        m=task_masks(Ytr,t); Xt=Xtr[m].to(dev); Yt=Ytr[m].to(dev)
        head.train()
        for ep in range(EPOCHS):
            perm=torch.randperm(Xt.shape[0])
            for i in range(0,len(perm),BS):
                idx=perm[i:i+BS]; x=Xt[idx]; y=Yt[idx]
                opt.zero_grad()
                logit=head(x)[:, :(t+1)*CPT]
                loss=F.cross_entropy(logit, y-0*0)  # labels already global; slice handles range
                loss=F.cross_entropy(head(x)[:, :(t+1)*CPT], y)
                loss.backward()
                if method in ('gpm','cobound') and basis is not None:
                    W=head.fc.weight.grad  # ncls x d
                    Wp=W - (W@basis)@basis.t()
                    head.fc.weight.grad.copy_(Wp)
                opt.step()
        # after task t, update memory basis from this task's features
        with torch.no_grad():
            sub=Xt[torch.randperm(Xt.shape[0])[:2000]]
            if method=='gpm':
                basis=update_basis(basis, sub, thresh=0.97)
            elif method=='cobound':
                basis=update_basis(basis, sub, thresh=0.985)
        accs=evaluate(head,Xv,Yv,t)
        acc_after.append(accs)
        print(f'{method} s{seed} task{t} meanacc {np.mean(accs):.4f} basis {0 if basis is None else basis.shape[1]}', flush=True)
    final=acc_after[-1]; final_avg=float(np.mean(final))
    # forgetting: max over time of acc[i] - final acc[i]
    fgs=[]
    for i in range(NTASK):
        peak=max(acc_after[t][i] for t in range(i,NTASK))
        fgs.append(peak-acc_after[-1][i])
    forget=float(np.mean(fgs))
    return final_avg, forget

if __name__=='__main__':
    method=sys.argv[1]; seeds=[int(x) for x in sys.argv[2].split(',')]
    fa=[]; fg=[]
    for s in seeds:
        a,f=run(method,s); fa.append(a); fg.append(f)
        print(f'RESULT {method} seed{s} acc {a:.4f} forget {f:.4f}', flush=True)
    print(f'SUMMARY {method} acc {np.mean(fa):.4f} +- {np.std(fa):.4f} forget {np.mean(fg):.4f} +- {np.std(fg):.4f}', flush=True)
