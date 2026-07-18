import os, torch, timm, glob
from PIL import Image
import torchvision.transforms as T
import numpy as np

ROOT=os.environ['PROJ']+'/data/tiny-imagenet-200'
OUT=os.environ['PROJ']+'/data/tin_vitfeat'
os.makedirs(OUT, exist_ok=True)
dev='cuda'
print('build model', flush=True)
model=timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=0).eval().to(dev)
cfg=timm.data.resolve_data_config({}, model=model)
tf=timm.data.create_transform(**cfg)
print('transform', cfg, flush=True)

wnids=sorted(os.listdir(os.path.join(ROOT,'train')))
wnid2idx={w:i for i,w in enumerate(wnids)}
assert len(wnids)==200, len(wnids)

@torch.no_grad()
def run(paths, labels, tag):
    feats=[]; ls=[]
    bs=256; buf=[]; bl=[]
    def flush(buf,bl):
        x=torch.stack(buf).to(dev)
        f=model(x).cpu()
        return f
    for i,(p,l) in enumerate(zip(paths,labels)):
        img=Image.open(p).convert('RGB')
        buf.append(tf(img)); bl.append(l)
        if len(buf)==bs:
            feats.append(flush(buf,bl)); ls+=bl; buf=[]; bl=[]
        if i%5000==0: print(tag,i,'/',len(paths), flush=True)
    if buf:
        feats.append(flush(buf,bl)); ls+=bl
    F=torch.cat(feats); L=torch.tensor(ls)
    torch.save({'feat':F,'label':L}, os.path.join(OUT,tag+'.pt'))
    print('SAVED',tag,F.shape,L.shape, flush=True)

# train
tp=[]; tl=[]
for w in wnids:
    d=os.path.join(ROOT,'train',w,'images')
    for fn in os.listdir(d):
        tp.append(os.path.join(d,fn)); tl.append(wnid2idx[w])
print('train imgs', len(tp), flush=True)
run(tp,tl,'train')

# val
vann=os.path.join(ROOT,'val','val_annotations.txt')
vp=[]; vl=[]
for line in open(vann):
    parts=line.strip().split('\t')
    fn=parts[0]; w=parts[1]
    vp.append(os.path.join(ROOT,'val','images',fn)); vl.append(wnid2idx[w])
print('val imgs', len(vp), flush=True)
run(vp,vl,'val')
print('ALL_DONE', flush=True)
