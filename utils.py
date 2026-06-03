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

class ASMVideoConstraint(nn.Module):
    def __init__(self, patch_size: int, type: str, reduction: str = "mean"):
        super().__init__()
        self.patch_size = patch_size
        self.padding = self.patch_size // 2
        self.criterion = L1Loss(reduction) if type == 'l1' else L2Loss(reduction)
        self.ratio = 0.001
        self.eps = 1e-6
        self.t0 = 0.1 # minimum transmission
        return
    
    @torch.no_grad()
    def _calculate_dark(self, image: torch.Tensor)-> torch.Tensor:
        '''
        image: (B, C, H, W)
        '''
        min_c, _ = torch.min(image, dim=1, keepdim=True) # (B, 1, H, W)
        padded_min_c = F.pad(min_c, (self.padding, self.padding, self.padding, self.padding), mode='replicate')
        dark = - F.max_pool2d(-padded_min_c, kernel_size=self.patch_size, stride=1, padding=0) # (B, 1, H, W)
        return dark
    
    @torch.no_grad()
    def _calculate_transmission(self, image: torch.Tensor, A: torch.Tensor)-> torch.Tensor:
        min_c = torch.min(image / A, dim=1, keepdim=True)[0]
        padded_min_c = F.pad(min_c, (self.padding, self.padding, self.padding, self.padding), mode='replicate')
        min_patch = -F.max_pool2d(-padded_min_c, kernel_size=self.patch_size, stride=1, padding=0)
        transmission = 1 - min_patch
        transmission = torch.clamp(transmission, min=self.t0)
        return transmission
    
    @torch.no_grad()
    def _dcp(self, hazy_t: torch.Tensor, hazy_next: torch.Tensor)-> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        hazy_t, hazy_next : (B, C, H, W)
        '''
        B, C, H, W = hazy_t.shape
        num_pixels = H * W
        top_k = max(1, int(num_pixels * self.ratio)) 
        
        I_derivative = hazy_next - hazy_t
        
        # Dark
        dark_t = self._calculate_dark(hazy_t) # (B, 1, H, W)
        dark_next = self._calculate_dark(hazy_next) # (B, 1, H, W)

        # Atmospheric Light
        dark_combined = torch.cat([dark_t.view(B, -1), dark_next.view(B, -1)], dim=-1)
        hazy_combined = torch.cat([hazy_t.view(B, C, -1), hazy_next.view(B, C, -1)], dim=-1)
        _, topk_indices = torch.topk(dark_combined, k=top_k, dim=-1) # (B, top_k)
        gather_indices = topk_indices.unsqueeze(1).expand(-1, C, -1)
        A_val = torch.gather(hazy_combined, dim=-1, index=gather_indices).mean(dim=-1, keepdim=True)
        A_val = torch.clamp(A_val, min=self.eps) 
        A = A_val.view(B, C, 1, 1).expand(B, C, H, W)

        # Transmission Map
        T_t = self._calculate_transmission(hazy_t, A) # (B, 1, H, W)
        T_next = self._calculate_transmission(hazy_next, A) # (B, 1, H, W)

        B_t = 1 / T_t
        B_next = 1 / T_next
        B_t_derivative = B_next - B_t

        return A, B_t, B_t_derivative, I_derivative
        
    def forward(self, output_t: torch.Tensor, output_next: torch.Tensor, 
                hazy_t: torch.Tensor, hazy_next: torch.Tensor
        )-> torch.Tensor:
        '''
        output_t, output-next : (B, C, H, W)
        hazy_t, hazy_next : (B, C, H, W)
        '''
        J_t_pred = output_next - output_t

        A, B, B_t, I_t = self._dcp(hazy_t, hazy_next) 
        
        J_t_theoretical = (B_t * hazy_t) + (B * I_t) - (A * B_t)

        loss = self.criterion(J_t_pred, J_t_theoretical)

        return loss
