import argparse,json
from pathlib import Path
import cv2,numpy as np,torch
from sklearn.model_selection import GroupKFold
from src.utils import *
from src.data import *
from src.model import FILANet
from src.inference import predict_tiled
from src.reconstruct import reconstruct_instances
from src.metric import pq_score
ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/filanet.yaml");ap.add_argument("--fold",type=int,default=0);ap.add_argument("--weights",required=True);a=ap.parse_args()
c=load_yaml(a.config);device="cuda" if torch.cuda.is_available() else "cpu";root=Path(c["data"]["root"]);r=load_coco_records(resolve_path(root,c["data"]["train_json"]))
idx=np.arange(len(r));groups=np.array([x["group"] for x in r]);_,va=list(GroupKFold(c["n_folds"]).split(idx,groups=groups))[a.fold]
ck=torch.load(a.weights,map_location=device);m=FILANet(c["model"]["encoder"],False,c["model"]["fpn_channels"],4).to(device);m.load_state_dict(ck["model"]);m.eval()
scores=[];imgdir=resolve_path(root,c["data"]["train_images"]);inf=c["inference"]
for i in va:
    rec=r[i];im=cv2.imread(str(imgdir/rec["file_name"]),cv2.IMREAD_GRAYSCALE);p=predict_tiled(m,build_input_channels(im),device,inf["tile"],inf["overlap"],inf["tta"])
    pm=reconstruct_instances(p["region"],p["centerline"],p["boundary"],p["distance"],**inf);gt=build_targets(rec)["instances"];scores.append(pq_score(gt,pm,c["metric"]["iou_match_threshold"]))
print(json.dumps({k:float(np.mean([x[k] for x in scores])) for k in ("pq","sq","rq","tp","fp","fn")},indent=2))
