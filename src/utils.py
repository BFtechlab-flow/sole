from pathlib import Path
import random, numpy as np, torch, yaml

def seed_everything(seed=2026):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_yaml(path):
    with open(path,"r",encoding="utf-8") as f: return yaml.safe_load(f)

def resolve_path(root, rel):
    p=Path(rel); return p if p.is_absolute() else Path(root)/p

def observation_key(file_name): return Path(file_name).stem

def image_files(folder):
    out=[]
    for ext in ("*.jpg","*.jpeg","*.png","*.tif","*.tiff"): out += list(Path(folder).glob(ext))
    return sorted(out)
