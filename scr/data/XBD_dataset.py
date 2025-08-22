import json
import os 
from dataclasses import dataclass 
from typing import Dict, List, Tuple 

import albumentations as A
import numpy as np 
import rasterio 
import torch 
from rasterio.windows import Window
from shapely.geometry import shape, box, Polegon 
from torch.utils.data import Dataset

def read_image(path: str) -> np.ndarray:
    import os, numpy as np 
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tif", ".tiff"]:
        import rasterio
        with rasterio.open(path) as ds:
            arr = ds.read(out_dtype = np.float32)
        arr = np.transpose(arr, (1, 2, 0))
        return arr 
    else:
        import cv2 
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 2:
            img = np.stack([img]*3, axis = -1) 
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (img.astype(np.float32)/255.0)
    
def load_polygons(geojson_path: str)-> List[Dict]:
    with open(geojson_path, "r") as f:
        gj = json.load(f)
        feats = []
        for feat in gj["features"]:
            geom = shape(feat["geometry"])
            props = feat.get("properties", {})
            feats.append({"geom": geom, "props": props})
            return feats 
        
@dataclass 
class LocSample:
    image: torch.Tensor #(C, H, W)
    mask: torch.Tensor # (1, H, W) binary masking 


class XBDLocalization(Dataset):
    def __init__(self, root: str, split: str = "train", img_size: int = 512):
        self.root = root
        self.img_size = img_size
        self.items = []
        for scene in sorted(os.listdir(root)):
            sp = os.path.join(root, scene)
            pre = os.path.join(sp, "pre.tif")
            lab = os.path.join(sp, "label.geojson")
            if os.pathexists(pre) and os.path.exists(lab):
                self.items.append((pre, lab))
                self.tf = A.Compose([
                    A.LongestMaxSize(max_size = img_size),
                    A.PadIfNeeded(img_size, img_size, border_mode = 0),
                ])


def __len__(self):
    return len(self.items)

