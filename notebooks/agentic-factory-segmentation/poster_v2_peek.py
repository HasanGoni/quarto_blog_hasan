import sys, numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
D='data/void_dataset_v2/'; sp=sys.argv[1]; seeds=[int(a) for a in sys.argv[2:]]
d=np.load(D+sp+'.npz'); X,M=d['images'],d['defect_masks']
col={1:np.array([255,60,60.]),2:np.array([40,220,255.])}
def ov(i,a=0.25):
    rgb=np.stack([X[i]]*3,-1).astype(float)
    for c,t in col.items():
        m=M[i]==c; e=m&~ndi.binary_erosion(m); rgb[m]=rgb[m]*(1-a)+t*a; rgb[e]=rgb[e]*.55+t*.45
    return Image.fromarray(rgb.clip(0,255).astype(np.uint8))
P=Image.new('RGB',(8*224,3*224*1),(24,24,28))
ids=[i for i in range(len(X)) if (M[i]>0).any()][::37][:12]
P=Image.new('RGB',(6*224,4*224))
for n,i in enumerate(ids):
    r,c=divmod(n,3)
    P.paste(Image.fromarray(X[i]).convert('RGB'),(c*2*224,r*224)); P.paste(ov(i),(c*2*224+224,r*224))
P.save('/tmp/claude-1000/-home-hasan-spark-workspace-projects-git-data/54c9f5c6-2a6b-49fd-8f62-bc95b29ec684/scratchpad/v/v2peek.png')
