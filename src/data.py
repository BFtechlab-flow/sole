from collections import defaultdict
from pathlib import Path
import json, random, cv2, numpy as np, torch
from torch.utils.data import Dataset
from pycocotools import mask as mask_utils
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from .utils import observation_key

def robust_normalize(img):
    x=img.astype(np.float32); lo,hi=np.percentile(x,[1,99])
    return np.clip((x-lo)/(hi-lo+1e-8),0,1)

def build_input_channels(gray):
    raw=robust_normalize(gray)
    u8=(raw*255).astype(np.uint8)
    clahe=cv2.createCLAHE(2.0,(8,8)).apply(u8).astype(np.float32)/255
    b1=cv2.GaussianBlur(raw,(0,0),1.2); b2=cv2.GaussianBlur(raw,(0,0),5.0)
    ridge=b2-b1; ridge=(ridge-ridge.min())/(np.ptp(ridge)+1e-8)
    h,w=raw.shape; yy,xx=np.mgrid[:h,:w]; cx,cy=(w-1)/2,(h-1)/2
    radial=np.clip(np.sqrt(((xx-cx)/(cx+1e-8))**2+((yy-cy)/(cy+1e-8))**2),0,1)
    return np.stack([raw,clahe,ridge,radial],0).astype(np.float32)

def segmentation_to_mask(seg,h,w):
    if isinstance(seg,list): r=mask_utils.merge(mask_utils.frPyObjects(seg,h,w))
    else: r=seg
    m=mask_utils.decode(r)
    if m.ndim==3: m=np.any(m,2)
    return m.astype(np.uint8)

def load_coco_records(path):
    data=json.load(open(path,"r",encoding="utf-8")); by=defaultdict(list)
    for a in data["annotations"]: by[a["image_id"]].append(a)
    return [{"image_id":i["id"],"file_name":i["file_name"],"height":int(i["height"]),"width":int(i["width"]),
             "annotations":by[i["id"]],"group":observation_key(i["file_name"])} for i in data["images"]]

def build_targets(r):
    h,w=r["height"],r["width"]; region=np.zeros((h,w),np.uint8); center=region.copy(); boundary=region.copy()
    distance=np.zeros((h,w),np.float32); instances=[]; kernel=np.ones((3,3),np.uint8)
    for a in r["annotations"]:
        # Only segmentation GT is consumed. Centerline/boundary/distance are derived.
        m=segmentation_to_mask(a["segmentation"],h,w)
        if not m.any(): continue
        instances.append(m); region=np.maximum(region,m)
        s=cv2.dilate(skeletonize(m>0).astype(np.uint8),kernel,iterations=1); center=np.maximum(center,s)
        boundary=np.maximum(boundary,cv2.morphologyEx(m,cv2.MORPH_GRADIENT,kernel))
        d=distance_transform_edt(m); d=d/(d.max()+1e-8); distance=np.maximum(distance,d)
    return {"region":region.astype(np.float32),"centerline":center.astype(np.float32),
            "boundary":boundary.astype(np.float32),"distance":distance.astype(np.float32),"instances":instances}

class FilamentDataset(Dataset):
    def __init__(self,records,image_dir,patch_size=1024,train=True):
        self.records=records; self.image_dir=Path(image_dir); self.patch_size=patch_size; self.train=train
    def __len__(self): return len(self.records)
    def __getitem__(self,idx):
        r=self.records[idx]; img=cv2.imread(str(self.image_dir/r["file_name"]),cv2.IMREAD_GRAYSCALE)
        if img is None: raise FileNotFoundError(self.image_dir/r["file_name"])
        x=build_input_channels(img); t=build_targets(r); h,w=t["region"].shape; s=min(self.patch_size,h,w)
        if t["region"].any() and random.random()<.7:
            ys,xs=np.where(t["region"]>0); j=random.randrange(len(xs)); cy,cx=ys[j],xs[j]
            y=max(0,min(h-s,int(cy)-s//2)); z=max(0,min(w-s,int(cx)-s//2))
        else:
            y=random.randint(0,max(h-s,0)); z=random.randint(0,max(w-s,0))
        x=x[:,y:y+s,z:z+s]; out={"image":torch.from_numpy(x).float()}
        for k in ("region","centerline","boundary","distance"):
            a=t[k][y:y+s,z:z+s]
            if self.train and random.random()<.5: a=np.fliplr(a).copy()
            out[k]=torch.from_numpy(a[None].copy()).float()
        return out
