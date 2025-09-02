import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, Parameter, Softmax
import os

def strip_optimizer(f="best.pt", s=""):
    """
    Strips optimizer and optionally saves checkpoint to finalize training; arguments are file path 'f' and save path
    's'.

    Example: from utils.general import *; strip_optimizer()
    """
    x = torch.load(f, map_location=torch.device("cpu"),weights_only=False)
    if x.get("ema"):
        x["model"] = x["ema"]  # replace model with ema
    for k in "optimizer", "best_fitness", "ema", "updates":  # keys
        x[k] = None
    x["epoch"] = -1
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1e6  # filesize
    print(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")


strip_optimizer("best.pt","otimize.pt")
