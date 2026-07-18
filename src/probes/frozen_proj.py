import os, sys, numpy as np, torch, torch.nn.functional as F
PROJ=os.environ['PROJ']; dev='cuda' if torch.cuda.is_available() else 'cpu'
D=torch.load(PROJ+'/data/tin_vitfeat/train.pt'); Dv=torch.load(PROJ+'/data/tin_vitfeat/val.pt')
Xtr,Ytr=D['feat'].float(),D['label'].long(); Xte,Yte=Dv['feat'].float(),Dv['label'].long()
mu_=Xtr.mean(0,keepdim=True); sd_=Xtr.std(0,keepdim=True)+1e-6
Xtr=(Xtr-mu_)/sd_; Xte=(Xte-mu_)/sd_
NTASK=10; CPT=20; d=Xtr.shape[1]; NC=NTASK*CPT
def tmask(Y,t): lo,hi=t*CPT,(t+1)*CPT; return (Y>=lo)&(Y<hi)
def basis_of(rows):
    U,S,_=torch.linalg.svd(rows.t(),full_matrices=False)
    k=int((torch.cumsum(S**2,0)/S.pow(2).sum()<0.99).sum())+1; return U[:,:k]
def sin_theta(Bi,Bj):
    s=torch.linalg.svdvals(Bi.t()@Bj).clamp(max=1.0); return float(torch.sqrt((1-s.min()**2).clamp(min=0)))
def coherence(Bi,Bj): return float(torch.linalg.svdvals(Bi.t()@Bj)[0])
if __name__=='__main__':
    seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
    torch.manual_seed(seed); np.random.seed(seed)
    W=torch.zeros(NC,d,device=dev,requires_grad=True); b=torch.zeros(NC,device=dev,requires_grad=True)
    def acc0():
        mv=tmask(Yte,0); Xi=Xte[mv].to(dev); Yi=Yte[mv].to(dev)
        with torch.no_grad(): p=(Xi@W.t()+b).argmax(1)
        return float((p==Yi).float().mean())
    # GPM-style input feature memory: project later-task gradients off task0 feature subspace
    Mem=None
    acc0_curve=[]; bases={}
    for t in range(NTASK):
        mtr=tmask(Ytr,t); Xi=Xtr[mtr].to(dev); Yi=Ytr[mtr].to(dev)
        opt=torch.optim.SGD([W,b],lr=0.05)
        for ep in range(15):
            for i in range(0,len(Xi),512):
                opt.zero_grad(); F.cross_entropy(Xi[i:i+512]@W.t()+b, Yi[i:i+512]).backward()
                if Mem is not None:
                    with torch.no_grad(): W.grad-= (W.grad@Mem)@Mem.t()
                opt.step()
        bases[t]=basis_of(W.detach()[t*CPT:(t+1)*CPT].cpu())
        acc0_curve.append(acc0())
        # accumulate feature memory from this task's inputs (frozen features)
        with torch.no_grad():
            Uf=basis_of(Xi.cpu()).to(dev)
            Mem=Uf if Mem is None else torch.linalg.qr(torch.cat([Mem,Uf],1))[0][:,:min(d, Mem.shape[1]+Uf.shape[1])]
    mus=[coherence(bases[0],bases[j]) for j in range(1,NTASK)]
    sins=[sin_theta(bases[0],bases[j]) for j in range(1,NTASK)]
    mu_max=max(mus); sin_agg=float(np.sqrt(np.mean(np.square(sins))))
    acc0_init=acc0_curve[0]; acc0_final=acc0_curve[-1]; drop0=acc0_init-acc0_final
    rel_drop=drop0/max(acc0_init,1e-6); RHS=sin_agg*mu_max
    print(f'SEED {seed} frozen ViT-B/16 GPM-projected NTASK={NTASK}',flush=True)
    print(f'  acc0 curve '+' '.join(f'{a:.3f}' for a in acc0_curve),flush=True)
    for j in range(NTASK-1): print(f'  task0-task{j+1}: mu {mus[j]:.4f} sin {sins[j]:.4f}',flush=True)
    print(f'RESULT seed{seed} acc0_init {acc0_init:.4f} acc0_final {acc0_final:.4f} drop0 {drop0:.4f} rel_drop {rel_drop:.4f} mu_max {mu_max:.4f} sin_agg {sin_agg:.4f} RHS {RHS:.4f} holds {int(rel_drop<=RHS)}',flush=True)
