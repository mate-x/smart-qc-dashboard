from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils.image_utils import apply_preprocessing

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class MVTecDataset(Dataset):
    """
    MVTec AD 폴더 구조를 읽어 (image_tensor, label, mask_tensor, image_path) 반환.

    split == "train" : train/good/ 이미지만 로드, label=0, mask=None
    split == "test"  : test/{class}/ 이미지 전체 로드,
                       class == "good" → label=0
                       class != "good" → label=1
                       gt mask: ground_truth/{class}/{filename} (없으면 zeros)
    """

    def __init__(
        self,
        dataset_path: str,
        split: Literal["train", "test"],
        preprocessing_config: dict,
    ) -> None:
        self.preprocessing_config = preprocessing_config
        self.items: list[dict] = []
        self._build_index(Path(dataset_path), split)

    def _build_index(self, root: Path, split: str) -> None:
        if split == "train":
            good_dir = root / "train" / "good"
            for p in sorted(good_dir.iterdir()):
                if p.suffix.lower() in SUPPORTED_EXTS:
                    self.items.append({"path": str(p), "label": 0, "mask_path": None})
        else:
            test_dir = root / "test"
            for class_dir in sorted(test_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                label = 0 if class_dir.name == "good" else 1
                gt_dir = root / "ground_truth" / class_dir.name
                for p in sorted(class_dir.iterdir()):
                    if p.suffix.lower() not in SUPPORTED_EXTS:
                        continue
                    mask_path = None
                    if label == 1:
                        candidate = gt_dir / p.name
                        if not candidate.exists():
                            candidate = gt_dir / (p.stem + ".png")
                        mask_path = str(candidate) if candidate.exists() else None
                    self.items.append({
                        "path": str(p),
                        "label": label,
                        "mask_path": mask_path,
                    })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]

        _, image_tensor = apply_preprocessing(item["path"], self.preprocessing_config)

        image_size = self.preprocessing_config["image_size"]
        if item["mask_path"] and Path(item["mask_path"]).exists():
            mask = Image.open(item["mask_path"]).convert("L")
            mask = mask.resize((image_size, image_size), Image.NEAREST)
            mask_tensor = torch.from_numpy(
                (np.array(mask) > 0).astype(np.float32)
            ).unsqueeze(0)
        else:
            mask_tensor = torch.zeros(1, image_size, image_size, dtype=torch.float32)

        return {
            "image":      image_tensor,
            "label":      torch.tensor(item["label"], dtype=torch.long),
            "mask":       mask_tensor,
            "image_path": item["path"],
        }


def build_dataloaders(
    dataset_path: str,
    preprocessing_config: dict,
    batch_size: int,
    random_seed: int,
) -> tuple[DataLoader, DataLoader]:
    """
    반환: (train_loader, test_loader)
    train_loader: shuffle=True, drop_last=True
    test_loader:  shuffle=False, batch_size=1
    """
    train_ds = MVTecDataset(dataset_path, "train", preprocessing_config)
    test_ds = MVTecDataset(dataset_path, "test", preprocessing_config)

    g = torch.Generator()
    g.manual_seed(random_seed)

    num_workers = 0  # Windows 환경 호환성
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        generator=g,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, test_loader
