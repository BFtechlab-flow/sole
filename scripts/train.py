import argparse
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from src.utils import *
from src.data import *
from src.model import FILANet
from src.losses import total_loss

ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/filanet.yaml");ap.add_argument("--fold",type=int,default=0);ap.add_argument("--output",default="checkpoints")
a=ap.parse_args();c=load_yaml(a.config);seed_everything(c["seed"]);device="cuda" if torch.cuda.is_available() else "cpu"
root=Path(c["data"]["root"]); rec=load_coco_records(resolve_path(root,c["data"]["train_json"])); idx=np.arange(len(rec)); groups=np.array([r["group"] for r in rec])
tr,va=list(GroupKFold(c["n_folds"]).split(idx,groups=groups))[a.fold]; assert not ({rec[i]["group"] for i in tr}&{rec[i]["group"] for i in va})
ds=FilamentDataset([rec[i] for i in tr],resolve_path(root,c["data"]["train_images"]),c["train"]["patch_size"],True)
dl=DataLoader(ds,batch_size=c["train"]["batch_size"],shuffle=True,num_workers=c["train"]["workers"],pin_memory=True)
m=FILANet(c["model"]["encoder"],c["model"]["pretrained"],c["model"]["fpn_channels"],4).to(device)
opt=torch.optim.AdamW(m.parameters(),lr=c["train"]["lr"],weight_decay=c["train"]["weight_decay"]);out=Path(a.output);out.mkdir(exist_ok=True)
scaler=torch.amp.GradScaler("cuda",enabled=device=="cuda")
for epoch in range(1,c["train"]["epochs"]+1):
    m.train(); tot=0
    for b in dl:
        x=b["image"].to(device);t={k:b[k].to(device) for k in ("region","centerline","boundary","distance")};opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",enabled=device=="cuda"):loss=total_loss(m(x),t,c["loss"])
        scaler.scale(loss).backward();scaler.step(opt);scaler.update();tot+=loss.item()
    print(epoch,tot/max(len(dl),1));torch.save({"model":m.state_dict(),"config":c,"fold":a.fold},out/f"fold{a.fold}_best.pt")
