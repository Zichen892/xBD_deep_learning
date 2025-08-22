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
        with rasterio.open(path):
            