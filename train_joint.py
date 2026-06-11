import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

from omegaconf import OmegaConf
from accelerate import Accelerator
from tqdm.auto import tqdm
from utils import set_seed, L1Loss, L2Loss, get_clean_state_dict, ASMVideoConstraintV2, save_loss_curve
from model import build_model
from dataset import Transform, REVIDEPairDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to yaml config file",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=datetime.now().strftime("%y%m%d_%H%M"),
        help="run name (default: YYMMDD_HHMM)",
    )

    return parser.parse_args()

def train():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    accelerator = Accelerator()

    save_dir_path = os.path.join(cfg.train.save_path, args.name)

    if accelerator.is_main_process:
        os.makedirs(save_dir_path, exist_ok=False)
        OmegaConf.save(cfg, os.path.join(save_dir_path, "config.yaml"))

    accelerator.print("=" * 50)
    accelerator.print("Loaded Config")
    accelerator.print(OmegaConf.to_yaml(cfg))
    accelerator.print("=" * 50)
    accelerator.print("Start Training...")

    set_seed(cfg.train.seed)
    model = build_model(cfg.model)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = L1Loss() if cfg.train.loss_type == 'l1' else L2Loss()
    constraint = ASMVideoConstraintV2(type= cfg.train.prior_loss_type)
    smooth = L1Loss() if cfg.train.loss_type == 'l1' else L2Loss()

    transform = Transform(crop_size= cfg.train.transform.crop_size)
    dataset = REVIDEPairDataset(root_dir= cfg.train.dataset.root_dir, num_frames= cfg.train.dataset.num_frames, transform= transform)
    accelerator.print(f"Found {len(dataset)} number of pairs")
    dataloader = DataLoader(
        dataset,
        batch_size= cfg.train.dataset.batch_size,
        shuffle= cfg.train.dataset.shuffle,
        num_workers= cfg.train.dataset.num_workers,
        pin_memory= cfg.train.dataset.pin_memory,
        drop_last= cfg.train.dataset.drop_last,
        persistent_workers= cfg.train.dataset.persistent_workers,
    )

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    epoch_bar = tqdm(
        range(1, cfg.train.epochs + 1),
        disable=not accelerator.is_local_main_process,
        desc="Epochs",
    )

    losses = []

    for epoch in epoch_bar:
        
        step_bar = tqdm(
            dataloader,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch}",
            leave=False,
        )

        loss_sum = 0.0
        cnt = 0

        for step, (batch_t, batch_next) in enumerate(step_bar):
            hazy_frames_t, gt_frame_t, masks_t = batch_t
            hazy_frames_next, gt_frame_next, masks_next = batch_next

            current_hazy_t = hazy_frames_t[:, -1, :, :, :] 
            current_hazy_next = hazy_frames_next[:, -1, :, :, :]

            output_t, a_t, t_t = model(hazy_frames_t, masks_t)
            output_next, a_next, t_next = model(hazy_frames_next, masks_next)
            
            loss_t = criterion(output_t, gt_frame_t)
            loss_next = criterion(output_next, gt_frame_next)
            
            loss_smooth = smooth(output_t, output_next)

            loss_prior = constraint(output_t, a_t, t_t, current_hazy_t, output_next, a_next, t_next, current_hazy_next)

            loss = loss_t + loss_next + cfg.train.smooth_weight * loss_smooth + cfg.train.prior_weight * loss_prior

            accelerator.backward(loss)

            if cfg.train.clip_grad:
                accelerator.clip_grad_norm_(model.parameters(), cfg.train.max_grad)

            optimizer.step()
            optimizer.zero_grad()

            reduced_loss = accelerator.reduce(loss.detach(), reduction="mean")
            
            loss_sum += reduced_loss.item()
            cnt += 1

            if accelerator.is_main_process:
                step_bar.set_postfix({
                    "loss": f"{reduced_loss.item():.4f}"
                })

        losses.append(loss_sum/cnt)    
        
        if epoch % cfg.train.save_interval == 0 and accelerator.is_main_process:
            checkpoint_save_path = os.path.join(save_dir_path, f"checkpoint_{epoch}.pt")
            uwrapped_model = accelerator.unwrap_model(model)
            torch.save(get_clean_state_dict(uwrapped_model), checkpoint_save_path)

    if accelerator.is_main_process:
        checkpoint_save_path = os.path.join(save_dir_path, "final.pt")
        uwrapped_model = accelerator.unwrap_model(model)
        torch.save(get_clean_state_dict(uwrapped_model), checkpoint_save_path)

        save_loss_curve(losses, save_dir_path)

if __name__ == "__main__":
    train()