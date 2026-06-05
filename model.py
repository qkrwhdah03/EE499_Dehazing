import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
from omegaconf import DictConfig, OmegaConf

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2), 
            ConvBlock(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

  
class UNet(nn.Module):
    def __init__(self, num_frames:int, base_ch: int):
        super().__init__()
        self.num_frames = num_frames
        self.inc = ConvBlock(3 * num_frames, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)

        self.up1 = Up(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up3 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up4 = Up(base_ch * 2, base_ch, base_ch)

        self.clean_head = nn.Sequential(
            nn.Conv2d(base_ch, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        '''
        Input:
            hazy video frames I: [B,T,3,H,W]
        Outputs:
            clean frame J: [B,3,H,W]
        """
        '''
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        y = self.up1(x5, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)

        y = self.clean_head(y)

        return y


class VideoDehazeModel(nn.Module):
    def __init__(self, num_frames: int, dehazer: nn.Module):
        super().__init__()
        self.num_frames = num_frames
        self.dehazer = dehazer
        self.empty_pixel = nn.Parameter(torch.zeros(1, 1, 3, 1, 1))

    def forward(self, frames, masks):
        '''
        Input:
            frames I: [B,T,3,H,W]
            masks: [B, T]
        Outputs:
            clean frame J: [B,3,H,W]
        '''
        B, T, C, H, W = frames.shape

        masks = masks.view(B, T, 1, 1, 1)
        empty = self.empty_pixel.expand(B, T, C, H, W)
        frames = masks * frames + (1 - masks) * empty
        x = frames.reshape(B, T * C, H, W)
        x = self.dehazer(x)
        
        return x


def build_model(cfg: DictConfig)-> nn.Module:
    dehazer = UNet(num_frames= cfg.num_frames, base_ch= cfg.embed_dim)
    model = VideoDehazeModel(num_frames= cfg.num_frames, dehazer= dehazer)
    return model

def load_model(checkpoint_path: str)-> nn.Module:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    config = OmegaConf.load(config_path)
    model = build_model(config.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt)
    return model