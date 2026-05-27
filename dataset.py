import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as F

NORMALIZE = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

class Transform:
    def __init__(self, crop_size):
        """
        crop_size:
            int or (h, w)
        """
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_h = crop_size[0]
        self.crop_w = crop_size[1]

    def __call__(self, hazy_frames, gt_frame):
        """
        hazy_frames:
            [T, C, H, W]

        gt_frame:
            [C, H, W]
        """

        _, _, H, W = hazy_frames.shape

        if H < self.crop_h or W < self.crop_w:
            raise ValueError(
                f"Crop size {(self.crop_h, self.crop_w)} "
                f"is larger than image size {(H, W)}"
            )

        top = random.randint(0, H - self.crop_h)
        left = random.randint(0, W - self.crop_w)

        hazy_frames = F.crop(
            hazy_frames,
            top=top,
            left=left,
            height=self.crop_h,
            width=self.crop_w,
        )

        gt_frame = F.crop(
            gt_frame,
            top=top,
            left=left,
            height=self.crop_h,
            width=self.crop_w,
        )
        return hazy_frames, gt_frame

class REVIDEDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="Train",
        num_frames=4,
        transform: Transform | None = None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.normalize = NORMALIZE

        self.gt_root = os.path.join(root_dir, split, "gt")
        self.hazy_root = os.path.join(root_dir, split, "hazy")

        self.samples = []

        seq_dirs = sorted(os.listdir(self.gt_root))
        for seq_name in seq_dirs:
            gt_seq_dir = os.path.join(self.gt_root, seq_name)
            hazy_seq_dir = os.path.join(self.hazy_root, seq_name)

            if not os.path.isdir(gt_seq_dir):
                continue

            frame_names = sorted(os.listdir(gt_seq_dir))

            for idx in range(len(frame_names)):

                hazy_paths = []

                for frame_idx in range(idx - num_frames + 1, idx + 1):

                    if frame_idx < 0:
                        hazy_paths.append(None)
                    else:
                        frame_name = frame_names[frame_idx]

                        hazy_path = os.path.join(hazy_seq_dir, frame_name)

                        hazy_paths.append(hazy_path)

                gt_path = os.path.join(gt_seq_dir, frame_names[idx])
                self.samples.append((hazy_paths, gt_path))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        hazy_paths, gt_path = self.samples[idx]

        hazy_frames = []
        mask = []
        ref_shape = None

        for path in hazy_paths:

            if path is None:
                hazy_frames.append(None)
                mask.append(0)
            else:
                img = Image.open(path).convert("RGB")
                img = self.normalize(img)

                if ref_shape is None:
                    ref_shape = img.shape

                hazy_frames.append(img)
                mask.append(1)

        C, H, W = ref_shape

        for i in range(len(hazy_frames)):
            if hazy_frames[i] is None:
                hazy_frames[i] = torch.zeros((C, H, W))

        hazy_frames = torch.stack(hazy_frames, dim=0)
        mask = torch.tensor(mask, dtype=torch.float32)

        gt_frame = Image.open(gt_path).convert("RGB")
        gt_frame = self.normalize(gt_frame)

        if self.transform is not None:
            hazy_frames, gt_frame = self.transform(hazy_frames, gt_frame)

        return hazy_frames, gt_frame, mask


class REVIDEInferenceDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="Test",
        num_frames=4,
        transform=None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.num_frames = num_frames
        self.transform = transform

        self.normalize = NORMALIZE
        self.hazy_root = os.path.join(root_dir, split, "hazy")
        self.gt_root = os.path.join(root_dir, split, "gt")

        self.video_list = []

        seq_dirs = sorted(os.listdir(self.hazy_root))

        for seq_name in seq_dirs:

            seq_dir = os.path.join(self.hazy_root, seq_name)

            if not os.path.isdir(seq_dir):
                continue

            frame_names = sorted(os.listdir(seq_dir))
            frame_paths = [os.path.join(seq_dir, fname) for fname in frame_names]
            gt_paths = [os.path.join(self.gt_root, seq_name, fname) for fname in frame_names]
            self.video_list.append((seq_name, frame_paths, gt_paths))

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        video_name, frame_paths, gt_paths = self.video_list[idx]

        clips = []
        masks = []
        gt_frames = []

        ref_shape = None

        for idx in range(len(frame_paths)):

            clip_frames = []
            clip_mask = []

            for frame_idx in range(idx - self.num_frames + 1, idx + 1):

                if frame_idx < 0:
                    clip_frames.append(None)
                    clip_mask.append(0)
                else:
                    img = Image.open(frame_paths[frame_idx]).convert("RGB")
                    img = self.normalize(img)

                    if self.transform is not None:
                        img = self.transform(img)

                    if ref_shape is None:
                        ref_shape = img.shape

                    clip_frames.append(img)
                    clip_mask.append(1)

            C, H, W = ref_shape

            for i in range(len(clip_frames)):
                if clip_frames[i] is None:
                    clip_frames[i] = torch.zeros((C, H, W))

            clip_frames = torch.stack(clip_frames, dim=0)
            clip_mask = torch.tensor(clip_mask, dtype=torch.float32)

            gt_img = Image.open(gt_paths[idx]).convert("RGB")
            gt_img = self.normalize(gt_img)

            if self.transform is not None:
                gt_img = self.transform(gt_img)

            clips.append(clip_frames)
            masks.append(clip_mask)
            gt_frames.append(gt_img)

        clips = torch.stack(clips, dim=0)
        masks = torch.stack(masks, dim=0)
        gt_frames = torch.stack(gt_frames, dim=0)

        return clips, gt_frames, masks, video_name