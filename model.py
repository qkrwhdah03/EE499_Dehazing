import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
from omegaconf import DictConfig, OmegaConf

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=ch),
        )

    def forward(self, x):
        return x + self.conv(x)

class DehazeEncoder(nn.Module):
    def __init__(self, in_ch: int= 3, feat: int= 64)-> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, feat, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            ResidualBlock(feat),
            ResidualBlock(feat),
            ResidualBlock(feat),
            ResidualBlock(feat),
            ResidualBlock(feat),
            ResidualBlock(feat),
            ResidualBlock(feat),
            ResidualBlock(feat),
        )

        self.head = nn.Sequential(
            nn.Conv2d(feat, feat, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        return
    
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
    
class DehazeDecoder(nn.Module):
    def __init__(self, num_frames:int, feat:int= 64, out_ch: int= 3)-> None:
        super().__init__()
        
        self.fusion = nn.Conv3d(
            feat, feat,
            kernel_size=(num_frames, 3, 3),
            padding=(0, 1, 1)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(feat, feat, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(feat),
            ResidualBlock(feat),
            ResidualBlock(feat),
            ResidualBlock(feat),
            ResidualBlock(feat),
            ResidualBlock(feat),
        )

        self.out = nn.Conv2d(feat, out_ch, 3, padding=1)
        return
    
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        x: [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape

        x = x.permute(0, 2, 1, 3, 4)

        x = self.fusion(x)  # [B, C, t, H, W]
        x = x.squeeze(2)   # [B, C, H, W]

        x = self.refine(x)
        x = self.out(x)
        return x


class VideoDehazeModel(nn.Module):
    def __init__(self, num_frames: int, feat: int, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.num_frames = num_frames

        self.encoder = encoder
        self.decoder = decoder

        self.empty_pixel = nn.Parameter(torch.zeros(1, feat, 1, 1))

    def forward(self, frames, masks= None):
        """
        Forward function for training

        frames: [B, T, 3, H, W]
        """

        B, T, C, H, W = frames.shape

        feats = []

        if masks is not None:
            masks_ = masks.view(B, T, 1, 1, 1)

        for t in range(T):
            f = self.encoder(frames[:, t])  # [B, F, H, W]

            if masks is not None:
                m = masks_[:, t] # [B, 1, 1, 1]
                empty_frame = self.empty_pixel.expand(B, -1, H, W)
                f = f + (1 - m) * empty_frame

            feats.append(f)

            if len(feats) > self.num_frames:
                feats.pop(0)

        x = torch.stack(feats, dim=1)  # [B, T, F, H, W]
        out = self.decoder(x)

        return out
    
def build_model(cfg: DictConfig)-> nn.Module:
    encoder = DehazeEncoder(in_ch= 3, feat= cfg.encoder.embed_dim)
    decoder = DehazeDecoder(num_frames= cfg.decoder.num_frames, feat= cfg.decoder.embed_dim, out_ch= 3)
    model = VideoDehazeModel(num_frames= cfg.num_frames, feat= cfg.embed_dim, encoder= encoder, decoder= decoder)
    return model

def load_model(checkpoint_path: str)-> nn.Module:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    config = OmegaConf.load(config_path)
    model = build_model(config.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt)
    return model