import os
import argparse
import imageio
import torch
import numpy as np
from tqdm import tqdm
from model import load_model
from dataset import REVIDEInferenceDataset
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to dataset root"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt)"
    )

    parser.add_argument(
        "--save_root",
        type=str,
        default="./results",
        help="Directory to save outputs"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="Test"
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=4
    )

    parser.add_argument(
        "--crop_size",
        type=int,
        default=512
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=20
    )

    return parser.parse_args()


def inference():
    args = parse_args()

    transform = transforms.Compose([
        transforms.CenterCrop((args.crop_size, args.crop_size)),
    ])

    dataset = REVIDEInferenceDataset(
        root_dir=args.data_root,
        split=args.split,
        num_frames=args.num_frames,
        transform=transform
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.checkpoint).to(device)
    model.eval()

    os.makedirs(args.save_root, exist_ok=True)

    for idx in range(len(dataset)):

        clips, gts, masks, name = dataset[idx]

        T = clips.shape[0]

        preds = []
        inputs = []
        gt_list = []

        for t in tqdm(range(T), desc=name):

            clip = clips[t].unsqueeze(0).to(device)
            mask = masks[t].unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(clip, mask)

            pred = pred.squeeze(0).cpu()

            preds.append(pred)
            inputs.append(clips[t][-1])
            gt_list.append(gts[t])

        preds = torch.stack(preds, dim=0)

        save_path = os.path.join(args.save_root, f"{name}.mp4")

        writer = imageio.get_writer(save_path, fps=args.fps)

        for inp, pred, gt in zip(inputs, preds, gt_list):

            inp = inp.clamp(0, 1)
            pred = pred.clamp(0, 1)
            gt = gt.clamp(0, 1)

            inp = inp.permute(1, 2, 0).numpy()
            pred = pred.permute(1, 2, 0).numpy()
            gt = gt.permute(1, 2, 0).numpy()

            concat = np.concatenate([inp, pred, gt], axis=1)
            concat = (concat * 255).astype(np.uint8)

            writer.append_data(concat)

        writer.close()

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    inference()