import os
import argparse
import torch
from tqdm import tqdm
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

from model import load_model
from dataset import REVIDEInferenceDataset

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True
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

    return parser.parse_args()

def eval():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.checkpoint).to(device)
    model.eval()

    dataset = REVIDEInferenceDataset(
        root_dir=args.data_root,
        split=args.split,
        num_frames= model.num_frames,
        transform= None
    )

    checkpoint_dir_name = os.path.basename(os.path.dirname(args.checkpoint))
    save_dir = os.path.join(args.save_root,checkpoint_dir_name)
    os.makedirs(save_dir, exist_ok=True)

    total_psnr = []
    total_ssim = []
    video_results = []

    for idx in range(len(dataset)):
        clips, gts, masks, name = dataset[idx]

        T = clips.shape[0]

        frame_psnr = []
        frame_ssim = []

        for t in tqdm(range(T), desc=name):
            clip = clips[t].unsqueeze(0).to(device)
            mask = masks[t].unsqueeze(0).to(device)
            gt = gts[t].unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(clip, mask)

            pred = pred.clamp(0, 1)
            gt = gt.clamp(0, 1)

            psnr = peak_signal_noise_ratio(pred, gt, data_range=1.0).item()
            ssim = structural_similarity_index_measure(pred, gt, data_range=1.0).item()

            frame_psnr.append(psnr)
            frame_ssim.append(ssim)

            total_psnr.append(psnr)
            total_ssim.append(ssim)

        avg_video_psnr = sum(frame_psnr) / len(frame_psnr)
        avg_video_ssim = sum(frame_ssim) / len(frame_ssim)

        video_results.append({
            "name": name,
            "frame_psnr": frame_psnr,
            "frame_ssim": frame_ssim,
            "avg_psnr": avg_video_psnr,
            "avg_ssim": avg_video_ssim
        })

        print(f"{name} | PSNR: {avg_video_psnr:.4f} | SSIM: {avg_video_ssim:.6f}")

    avg_total_psnr = sum(total_psnr) / len(total_psnr)
    avg_total_ssim = sum(total_ssim) / len(total_ssim)

    eval_path = os.path.join(save_dir, "eval.txt")

    with open(eval_path, "w") as f:
        f.write("========== Evaluation ==========\n\n")

        for result in video_results:
            f.write(f"[Video] {result['name']}\n")

            for idx, (psnr, ssim) in enumerate(zip(result["frame_psnr"], result["frame_ssim"])):
                f.write(f"Frame {idx:04d} | PSNR: {psnr:.4f} | SSIM: {ssim:.6f}\n")

            f.write(f"Video Average | PSNR: {result['avg_psnr']:.4f} | SSIM: {result['avg_ssim']:.6f}\n\n")

        f.write("========== Overall ==========\n")
        f.write(f"Average PSNR: {avg_total_psnr:.4f}\n")
        f.write(f"Average SSIM: {avg_total_ssim:.6f}\n")

    print("\n========== Overall ==========")
    print(f"Average PSNR: {avg_total_psnr:.4f}")
    print(f"Average SSIM: {avg_total_ssim:.6f}")
    print(f"\nSaved to: {eval_path}")

    return

if __name__ == "__main__":
    eval()