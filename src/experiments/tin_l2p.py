import os, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import timm
from PIL import Image
PROJ=os.environ['PROJ']; ROOT=PROJ+'/data/tiny-imagenet-200'
dev='cuda'
NTASK=10; CPT=20; NCLS=NTASK*CPT
EPOCHS=5; LR=0.03; BS=128
POOL=10; PLEN=5; TOPN=4  # L2P prompt pool params

# ---- data: use precomputed features? No, L2P needs raw images through ViT.
# We cache preprocessed tensors on the fly per task to keep it fast.
import torchvision.transforms as TT
tf=TT.Compose([TT.Resize((224,224), interpolation=TT.InterpolationMode.BICUBIC), TT.ToTensor(), TT.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

def build_index():
    wnids=sorted(os.listdir(os.path.join(ROOT,'train')))
    w2i={w:i for i,w in enumerate(wnids)}
    tr=[]
    for w in wnids:
        d=os.path.join(ROOT,'train',w,'images')
        for fn in os.listdir(d): tr.append((os.path.join(d,fn), w2i[w]))
    va=[]
    for line in open(os.path.join(ROOT,'val','val_annotations.txt')):
        p=line.strip().split('\t'); va.append((os.path.join(ROOT,'val','images',p[0]), w2i[p[1]]))
    return tr, va

class ImgDS(torch.utils.data.Dataset):
    def __init__(self, items): self.items=items
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        p,l=self.items[i]; return tf(Image.open(p).convert('RGB')), l

class L2P(nn.Module):
    def __init__(self, vit, d=768):
        super().__init__(); self.vit=vit
        for p in self.vit.parameters(): p.requires_grad=False
        self.prompt=nn.Parameter(torch.randn(POOL,PLEN,d)*0.02)
        self.keys=nn.Parameter(torch.randn(POOL,d)*0.02)
        self.head=nn.Linear(d,NCLS)
    @torch.no_grad()
    def query(self,x):
        f=self.vit.forward_features(x)  # B, 1+N, D
        return f[:,0]  # cls token as query
    def forward(self,x):
        q=self.query(x)  # B,D
        kn=F.normalize(self.keys,dim=1); qn=F.normalize(q,dim=1)
        sim=qn@kn.t()  # B,POOL
        idx=sim.topk(TOPN,dim=1).indices  # B,TOPN
        sel=self.prompt[idx]  # B,TOPN,PLEN,D
        B=x.shape[0]; sel=sel.reshape(B,TOPN*PLEN,self.prompt.shape[2])
        # manual token assembly
        v=self.vit
        pe=v.patch_embed(x)  # B,N,D
        cls=v.cls_token.expand(B,-1,-1)
        tok=torch.cat([cls,pe],1)
        tok=tok+v.pos_embed[:,:tok.shape[1]]
        tok=torch.cat([tok[:, :1], sel, tok[:, 1:]], 1)  # prompts after cls
        tok=v.pos_drop(tok)
        tok=v.blocks(tok); tok=v.norm(tok)
        feat=tok[:,0]
        # pull-loss to align keys with query
        pull=(1.0 - (qn*F.normalize(self.keys[idx].mean(1),dim=1)).sum(1)).mean()
        return self.head(feat), pull

def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    vit=timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=0).eval().to(dev)
    model=L2P(vit).to(dev)
    tr,va=build_index()
    def taskitems(items,t):
        lo=t*CPT; hi=lo+CPT; return [it for it in items if lo<=it[1]<hi]
    params=[model.prompt, model.keys]+list(model.head.parameters())
    opt=torch.optim.Adam(params, lr=LR)
    acc_after=[]
    for t in range(NTASK):
        ds=ImgDS(taskitems(tr,t)); dl=torch.utils.data.DataLoader(ds,batch_size=BS,shuffle=True,num_workers=6,pin_memory=True)
        model.train()
        for ep in range(EPOCHS):
            for x,y in dl:
                x=x.to(dev,non_blocking=True); y=y.to(dev,non_blocking=True)
                opt.zero_grad()
                logit,pull=model(x)
                logit=logit[:, :(t+1)*CPT]
                loss=F.cross_entropy(logit,y)+0.5*pull
                loss.backward(); opt.step()
        # eval
        accs=[]
        model.eval()
        with torch.no_grad():
            for tt in range(t+1):
                dv=ImgDS(taskitems(va,tt)); dvl=torch.utils.data.DataLoader(dv,batch_size=256,num_workers=6)
                cor=0; tot=0
                for x,y in dvl:
                    x=x.to(dev); y=y.to(dev)
                    lo,_=model(x); lo=lo[:, :(t+1)*CPT]
                    cor+=(lo.argmax(1)==y).sum().item(); tot+=len(y)
                accs.append(cor/max(tot,1))
        acc_after.append(accs)
        print(f'l2p s{seed} task{t} meanacc {np.mean(accs):.4f}', flush=True)
    final_avg=float(np.mean(acc_after[-1]))
    fgs=[max(acc_after[tt][i] for tt in range(i,NTASK))-acc_after[-1][i] for i in range(NTASK)]
    return final_avg, float(np.mean(fgs))

if __name__=='__main__':
    seeds=[int(x) for x in sys.argv[1].split(',')]
    fa=[];fg=[]
    for s in seeds:
        a,f=run(s); fa.append(a); fg.append(f)
        print(f'RESULT l2p seed{s} acc {a:.4f} forget {f:.4f}', flush=True)
    print(f'SUMMARY l2p acc {np.mean(fa):.4f} +- {np.std(fa):.4f} forget {np.mean(fg):.4f} +- {np.std(fg):.4f}', flush=True)
