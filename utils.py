import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_clean_state_dict(model):
    """
    Remove _orig_mod prefix from model.state_dict()
    """
    raw_state_dict = model.state_dict()
    clean_state_dict = {
        (k[10:] if k.startswith('_orig_mod.') else k): v 
        for k, v in raw_state_dict.items()
    }
    return clean_state_dict

class L1Loss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.abs(pred - target)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class L2Loss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = (pred - target) ** 2

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss