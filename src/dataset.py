from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}


def encode_label(text: str) -> list[int]:
    return [CHAR_TO_IDX[c] for c in text]


def decode_prediction(outputs: torch.Tensor) -> str:
    indices = outputs.argmax(dim=-1)
    return "".join(IDX_TO_CHAR[i.item()] for i in indices)


def get_train_transforms(height: int = 64, width: int = 192) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height, width),
            # Reduce noise intensity (was std_range=(0.02, 0.10), p=0.5)
            A.GaussNoise(std_range=(0.01, 0.05), p=0.4),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            # Reduce geometric distortions to keep characters readable
            A.Affine(
                scale=(0.97, 1.03),
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                rotate=(-3, 3),
                shear=(-3, 3),
                p=0.4,
            ),
            # Smaller dropout holes (was 4-8px)
            A.CoarseDropout(
                num_holes_range=(1, 2),
                hole_height_range=(2, 4),
                hole_width_range=(2, 4),
                p=0.2,
            ),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ]
    )


def get_val_transforms(height: int = 64, width: int = 192) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height, width),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ]
    )


class CaptchaDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        transform: A.Compose | None = None,
        num_chars: int = 5,
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.num_chars = num_chars

        self.image_paths = list(self.image_dir.glob("*.jpg")) + list(
            self.image_dir.glob("*.png")
        )

        self.image_paths = [
            p for p in self.image_paths if len(p.stem) == num_chars
        ]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        label_text = image_path.stem

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image).unsqueeze(0).float() / 255.0

        label = torch.tensor(encode_label(label_text), dtype=torch.long)

        return image, label
