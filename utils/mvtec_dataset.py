from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils.dataset_converter import detect_ok_ng_dirs
from utils.image_utils import apply_preprocessing

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_DEFAULT_TRAIN_RATIO = 0.8


class MVTecDataset(Dataset):
    """
    MVTec AD 또는 OK/NG 폴더 구조를 읽어 (image_tensor, label, mask_tensor, image_path) 반환.

    MVTec 형식:
      split == "train" : train/good/ 이미지만 로드, label=0
      split == "test"  : test/{class}/ 이미지 전체 로드 (good→0, 나머지→1)

    OK/NG 형식 (oking):
      split == "train" : OK 이미지의 앞 80%를 train으로 사용, label=0
      split == "test"  : OK 이미지의 뒤 20% (label=0) + NG 이미지 전부 (label=1)
    """

    def __init__(
        self,
        dataset_path: str,
        split: Literal["train", "test"],
        preprocessing_config: dict,
        random_seed: int = 42,
        train_ratio: float = _DEFAULT_TRAIN_RATIO,
    ) -> None:
        self.preprocessing_config = preprocessing_config
        self.random_seed = random_seed
        self.train_ratio = train_ratio
        self.items: list[dict] = []
        root = Path(dataset_path)
        if self._is_oking_format(root):
            self._build_index_oking(root, split)
        else:
            self._build_index_mvtec(root, split)

    # ------------------------------------------------------------------
    # Format detection
    # ------------------------------------------------------------------

    def _is_oking_format(self, root: Path) -> bool:
        ok_dir, _ = detect_ok_ng_dirs(root)
        return ok_dir is not None and not (root / "train" / "good").exists()

    # ------------------------------------------------------------------
    # MVTec index builder (original logic)
    # ------------------------------------------------------------------

    def _build_index_mvtec(self, root: Path, split: str) -> None:
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

    # ------------------------------------------------------------------
    # OK/NG index builder
    # ------------------------------------------------------------------

    def _build_index_oking(self, root: Path, split: str) -> None:
        ok_dir, ng_dir = detect_ok_ng_dirs(root)

        ok_images = sorted(
            p for p in ok_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        )
        rng = random.Random(self.random_seed)
        shuffled = list(ok_images)
        rng.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * self.train_ratio))
        train_images = shuffled[:split_idx]
        test_good_images = shuffled[split_idx:]

        ng_images: list[Path] = []
        if ng_dir is not None:
            ng_images = sorted(
                p for p in ng_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
            )

        if split == "train":
            for p in train_images:
                self.items.append({"path": str(p), "label": 0, "mask_path": None})
        else:
            for p in test_good_images:
                self.items.append({"path": str(p), "label": 0, "mask_path": None})
            for p in ng_images:
                self.items.append({"path": str(p), "label": 1, "mask_path": None})

    # ------------------------------------------------------------------
    # Kept for backwards compatibility (routes to MVTec builder)
    # ------------------------------------------------------------------

    def _build_index(self, root: Path, split: str) -> None:
        self._build_index_mvtec(root, split)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]

        _, image_tensor = apply_preprocessing(item["path"], self.preprocessing_config)

        image_size = self.preprocessing_config.get("image_size", 256)
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
    train_ds = MVTecDataset(dataset_path, "train", preprocessing_config, random_seed=random_seed)
    test_ds = MVTecDataset(dataset_path, "test", preprocessing_config, random_seed=random_seed)

    g = torch.Generator()
    g.manual_seed(random_seed)

    use_cuda = torch.cuda.is_available()
    num_workers = 0  # Windows 백그라운드 스레드에서 spawn 데드락 방지
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=True,
        generator=g,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    return train_loader, test_loader
