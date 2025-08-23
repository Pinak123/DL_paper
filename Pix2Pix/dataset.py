import os
from typing import Callable, Optional, Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import random


class Pix2PixDataset(Dataset):
    """
    Pix2Pix dataset supporting two layouts:
    - aligned: images are single files with A|B concatenated horizontally
    - separate: two folders 'A' and 'B' that contain paired images with the same filenames

    Returns dict with keys: 'A', 'B', and 'path'.
    """

    def __init__(
        self,
        root: str,
        mode: str = "aligned",
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 256,
        random_jitter: bool = True,
        normalize: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert mode in {"aligned", "separate"}, "mode must be 'aligned' or 'separate'"
        self.root = root
        self.mode = mode
        self.split = split
        self.image_size = image_size
        self.random_jitter = random_jitter
        self.normalize = normalize
        self.user_transform = transform
        if seed is not None:
            random.seed(seed)

        if mode == "aligned":
            # Expect images in root/{split}
            self.dir = os.path.join(root, split)
            self.paths = sorted(
                [
                    os.path.join(self.dir, f)
                    for f in os.listdir(self.dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
                ]
            )
        else:
            # Expect images in root/{split}/A and root/{split}/B
            self.dirA = os.path.join(root, split, "A")
            self.dirB = os.path.join(root, split, "B")
            a_files = set(
                f
                for f in os.listdir(self.dirA)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            )
            b_files = set(
                f
                for f in os.listdir(self.dirB)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            )
            common = sorted(list(a_files & b_files))
            self.paths = [(os.path.join(self.dirA, f), os.path.join(self.dirB, f)) for f in common]

        # Default transforms per Pix2Pix: random jitter (resize→random crop), horizontal flip, normalize [-1,1]
        base: List[T.transforms] = []
        if self.random_jitter:
            # As in paper/code: resize to 286 then random crop to 256
            base.extend([
                T.Resize((image_size + 30, image_size + 30), interpolation=T.InterpolationMode.BICUBIC),
                # random crop will be applied jointly on A and B via coordinates
            ])
        else:
            base.append(T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC))

        self.to_tensor = T.ToTensor()
        self.normalize_tf = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) if normalize else None

    def __len__(self) -> int:
        return len(self.paths)

    def _load_pair(self, idx: int) -> Tuple[Image.Image, Image.Image, str]:
        if self.mode == "aligned":
            path = self.paths[idx]
            img = Image.open(path).convert("RGB")
            w, h = img.size
            assert w % 2 == 0, f"Aligned image width must be even, got {w} for {path}"
            w2 = w // 2
            A = img.crop((0, 0, w2, h))
            B = img.crop((w2, 0, w, h))
            return A, B, path
        else:
            pathA, pathB = self.paths[idx]
            A = Image.open(pathA).convert("RGB")
            B = Image.open(pathB).convert("RGB")
            return A, B, pathA

    @staticmethod
    def _random_crop_params(img_w: int, img_h: int, crop_w: int, crop_h: int) -> Tuple[int, int]:
        if img_w == crop_w and img_h == crop_h:
            return 0, 0
        i = random.randint(0, img_h - crop_h) if img_h > crop_h else 0
        j = random.randint(0, img_w - crop_w) if img_w > crop_w else 0
        return i, j

    def _apply_joint_transforms(self, A: Image.Image, B: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resize (for jitter) or to fixed size
        if self.random_jitter:
            resize_size = (self.image_size + 30, self.image_size + 30)
        else:
            resize_size = (self.image_size, self.image_size)
        A = A.resize(resize_size, Image.BICUBIC)
        B = B.resize(resize_size, Image.BICUBIC)

        # Random crop to image_size x image_size (jitter)
        if self.random_jitter:
            i, j = self._random_crop_params(resize_size[0], resize_size[1], self.image_size, self.image_size)
            A = A.crop((j, i, j + self.image_size, i + self.image_size))
            B = B.crop((j, i, j + self.image_size, i + self.image_size))

        # Random horizontal flip synchronously
        if random.random() < 0.5:
            A = A.transpose(Image.FLIP_LEFT_RIGHT)
            B = B.transpose(Image.FLIP_LEFT_RIGHT)

        A = self.to_tensor(A)
        B = self.to_tensor(B)
        if self.normalize_tf is not None:
            A = self.normalize_tf(A)
            B = self.normalize_tf(B)
        return A, B

    def __getitem__(self, idx: int):
        A_img, B_img, path = self._load_pair(idx)

        if self.user_transform is None:
            A, B = self._apply_joint_transforms(A_img, B_img)
        else:
            # If a custom transform is provided, it should accept PIL images (A, B)
            A, B = self.user_transform(A_img, B_img)

        return {
            "A": A,  # condition
            "B": B,  # target
            "path": path,
        }


__all__ = [
    "Pix2PixDataset",
]


