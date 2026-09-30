import numpy as np, torch

def positions(n,tile,overlap):
    if n<=tile:return [0]
    p=list(range(0,n-tile+1,tile-overlap))
    if p[-1]!=n-tile:p.append(n-tile)
    return p

@torch.no_grad()
def predict_tiled(model,x,device,tile=1024,overlap=256,tta=True):
    _,h,w=x.shape; keys=("region","centerline","boundary","distance")
    acc={k:np.zeros((h,w),np.float32) for k in keys}; weight=np.zeros((h,w),np.float32)
    win=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)).astype(np.float32),.05)
    for y in positions(h,tile,overlap):
        for z in positions(w,tile,overlap):
            p=x[:,y:y+tile,z:z+tile]; tensor=torch.from_numpy(p[None]).float().to(device)
            out={k:torch.sigmoid(v)[0,0].cpu().numpy() for k,v in model(tensor).items()}
            hh,ww=p.shape[-2:]; q=win[:hh,:ww]
            for k in keys:acc[k][y:y+hh,z:z+ww]+=out[k]*q
            weight[y:y+hh,z:z+ww]+=q
    return {k:v/np.maximum(weight,1e-6) for k,v in acc.items()}
