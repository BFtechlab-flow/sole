import numpy as np, torch

PREDICTION_KEYS=("region","centerline","boundary","distance")

def positions(n,tile,overlap):
    if n<=tile:return [0]
    step=tile-overlap
    if step<=0: raise ValueError("overlap must be smaller than tile")
    p=list(range(0,n-tile+1,step))
    if p[-1]!=n-tile:p.append(n-tile)
    return p

def _inverse_flip(tensor,hflip=False,vflip=False):
    if hflip: tensor=torch.flip(tensor,dims=(-1,))
    if vflip: tensor=torch.flip(tensor,dims=(-2,))
    return tensor

@torch.no_grad()
def _predict_patch(model,tensor,tta=True):
    transforms=[(False,False)]
    if tta:
        transforms.extend([(True,False),(False,True),(True,True)])
    summed={k:None for k in PREDICTION_KEYS}
    for hflip,vflip in transforms:
        aug=_inverse_flip(tensor,hflip=hflip,vflip=vflip)
        raw=model(aug)
        for k in PREDICTION_KEYS:
            pred=torch.sigmoid(raw[k])
            pred=_inverse_flip(pred,hflip=hflip,vflip=vflip)
            summed[k]=pred if summed[k] is None else summed[k]+pred
    scale=1.0/len(transforms)
    return {k:(v*scale)[0,0].cpu().numpy() for k,v in summed.items()}

@torch.no_grad()
def predict_tiled(model,x,device,tile=1024,overlap=256,tta=True):
    _,h,w=x.shape
    acc={k:np.zeros((h,w),np.float32) for k in PREDICTION_KEYS}; weight=np.zeros((h,w),np.float32)
    win=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)).astype(np.float32),.05)
    for y in positions(h,tile,overlap):
        for z in positions(w,tile,overlap):
            p=x[:,y:y+tile,z:z+tile]; tensor=torch.from_numpy(p[None]).float().to(device)
            out=_predict_patch(model,tensor,tta=tta)
            hh,ww=p.shape[-2:]; q=win[:hh,:ww]
            for k in PREDICTION_KEYS: acc[k][y:y+hh,z:z+ww]+=out[k]*q
            weight[y:y+hh,z:z+ww]+=q
    return {k:v/np.maximum(weight,1e-6) for k,v in acc.items()}
