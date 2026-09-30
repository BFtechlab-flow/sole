import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from src.utils import load_yaml,resolve_path
from src.data import load_coco_records
ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/filanet.yaml");a=ap.parse_args()
c=load_yaml(a.config);root=Path(c["data"]["root"]);r=load_coco_records(resolve_path(root,c["data"]["train_json"]));g=Counter(x["group"] for x in r)
print("records",len(r));print("unique observations",len(g));print("repeated observations",sum(v>1 for v in g.values()))
print("sizes",sorted(set((x["height"],x["width"]) for x in r)));print("mean instances",np.mean([len(x["annotations"]) for x in r]))
