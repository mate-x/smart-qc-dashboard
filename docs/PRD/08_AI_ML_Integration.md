# 08. AI/ML Integration

> **참조 기준**: [00_Global_Context_Document.md](./00_Global_Context_Document.md)
> **선행 문서**: [04_System_Architecture.md](./04_System_Architecture.md)
> **버전**: v1.0
> **작성일**: 2026-05-08
> **중요**: 이 문서는 ML 구현의 Single Source of Truth다. Anomalib API 클래스명·파라미터 매핑·알고리즘 수식은 이 문서에서 확정되며, `model_factory.py`, `training_worker.py`, `image_utils.py` 구현 시 이 문서와 100% 일치해야 한다.

---

## 목차

- [A. Objective & Scope](#a-objective--scope)
- [B. Detailed Specification](#b-detailed-specification)
  - [B.1 Anomalib v1.0+ API 기준](#b1-anomalib-v10-api-기준)
  - [B.2 데이터 파이프라인 — Custom Dataset](#b2-데이터-파이프라인--custom-dataset)
  - [B.3 이미지 전처리 알고리즘](#b3-이미지-전처리-알고리즘)
  - [B.4 EfficientAD 구현](#b4-efficientad-구현)
  - [B.5 PatchCore 구현](#b5-patchcore-구현)
  - [B.6 TrainingWorker 통합 구현](#b6-trainingworker-통합-구현)
  - [B.7 추론 및 Anomaly Map 생성](#b7-추론-및-anomaly-map-생성)
  - [B.8 Anomaly Score 정규화 및 Threshold 계산](#b8-anomaly-score-정규화-및-threshold-계산)
  - [B.9 메트릭 계산](#b9-메트릭-계산)
- [C. System & Data Design](#c-system--data-design)
- [D. API Contracts](#d-api-contracts)
- [E. AI/ML Details](#e-aiml-details)
- [F. Non-Functional Requirements](#f-non-functional-requirements)
- [G. Observability](#g-observability)
- [H. QA & Validation](#h-qa--validation)
- [I. Implementation Plan](#i-implementation-plan)

---

## A. Objective & Scope

### A.1 이 문서의 목적

ML 구현에서 팀 간 발산이 발생하는 4개 핵심 영역을 확정한다:
1. Anomalib v1.0+ 클래스명·API 사용 방식
2. model_config 파라미터 → Anomalib 생성자 파라미터 매핑 테이블
3. 이미지 전처리 필터 알고리즘 (Homomorphic / HE / CLAHE) 수식·구현
4. 학습 루프·추론 파이프라인 통합 구조

### A.2 설계 제약

| 제약 | 내용 |
|------|------|
| **DA-01** | Anomalib Engine은 사용하지 않는다. 모델 객체를 직접 제어하여 stop_event 체크·Queue 보고를 가능하게 한다 |
| **DA-02** | Anomalib의 DataModule 대신 커스텀 `MVTecDataset`을 사용한다. 전처리 파이프라인이 image_utils.py 단일 구현으로 보장된다 (ADR-05) |
| **DA-03** | EfficientAD는 Anomalib 모델 객체의 파라미터를 직접 설정 후 커스텀 학습 루프로 실행한다 |
| **DA-04** | PatchCore는 그라디언트 업데이트 없음. 특징 추출 → coreset 구성 → KNN 인덱스 순서로 실행한다 |
| **DA-05** | 추론 시 image_utils.apply_preprocessing()을 학습과 동일하게 적용한다. 추론 전처리 불일치는 AUC 저하의 주요 원인이므로 엄격히 지켜야 한다 |

---

## B. Detailed Specification

---

### B.1 Anomalib v1.0+ API 기준

#### B.1.1 확정 import 경로

```python
# EfficientAD
from anomalib.models.image.efficient_ad.lightning_model import EfficientAd
from anomalib.models.image.efficient_ad.lightning_model import EfficientAdModelSize

# PatchCore
from anomalib.models.image.patchcore.lightning_model import Patchcore

# 공통
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
```

> **주의**: Anomalib v1.0+에서 클래스명은 `EfficientAd` (소문자 d), `Patchcore` (소문자 c)이다.
> 이전 버전의 `EfficientAD`, `PatchCore` 클래스명과 다르다.

#### B.1.2 EfficientAdModelSize 열거값

```python
# anomalib.models.image.efficient_ad.lightning_model
class EfficientAdModelSize(str, Enum):
    S = "small"   # model_config.params.model_size == "small"
    M = "medium"  # model_config.params.model_size == "medium"
```

매핑:
```python
SIZE_MAP = {
    "small":  EfficientAdModelSize.S,
    "medium": EfficientAdModelSize.M,
}
model_size_enum = SIZE_MAP[model_config["params"]["model_size"]]
```

---

### B.2 데이터 파이프라인 — Custom Dataset

#### B.2.1 MVTecDataset 클래스

```python
# utils/mvtec_dataset.py

import os
from pathlib import Path
from typing import Literal
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
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
        preprocessing_config: dict,    # 00_Global_Context 1.6절
    ):
        self.preprocessing_config = preprocessing_config
        self.items: list[dict] = []    # {"path": str, "label": int, "mask_path": str|None}
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
                        # .png 확장자로 시도 (MVTec AD GT 마스크는 .png)
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

        # 이미지 로드 + 전처리 전체 파이프라인 (ADR-05)
        _, image_tensor = apply_preprocessing(item["path"], self.preprocessing_config)
        # image_tensor: torch.Tensor (C, H, W), float32, 정규화 적용

        # GT 마스크 로드
        image_size = self.preprocessing_config["image_size"]
        if item["mask_path"] and Path(item["mask_path"]).exists():
            mask = Image.open(item["mask_path"]).convert("L")
            mask = mask.resize((image_size, image_size), Image.NEAREST)
            mask_tensor = torch.from_numpy(
                (np.array(mask) > 0).astype(np.float32)
            ).unsqueeze(0)  # (1, H, W)
        else:
            mask_tensor = torch.zeros(1, image_size, image_size, dtype=torch.float32)

        return {
            "image":      image_tensor,          # (C, H, W) float32
            "label":      torch.tensor(item["label"], dtype=torch.long),
            "mask":       mask_tensor,           # (1, H, W) float32
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
    train_loader: shuffle=True, drop_last=True (batch 수 일관성)
    test_loader:  shuffle=False, batch_size=1 (이미지별 개별 평가)
    """
    train_ds = MVTecDataset(dataset_path, "train", preprocessing_config)
    test_ds  = MVTecDataset(dataset_path, "test",  preprocessing_config)

    g = torch.Generator()
    g.manual_seed(random_seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        generator=g,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, test_loader
```

#### B.2.2 ImageNet Penalty DataLoader (EfficientAD 전용)

EfficientAD는 학습 중 ImageNet 샘플을 이용해 패널티 손실을 계산한다. `torchvision`의 FakeData 또는 미니 ImageNet 서브셋을 사용한다.

```python
def build_imagenet_penalty_loader(
    batch_size: int,          # model_config.params.penalty_batch_size
    image_size: int,
    device: str,
) -> DataLoader:
    """
    ImageNet penalty 배치용 DataLoader.
    실제 ImageNet 대신 torchvision.datasets.FakeData 사용 (추론 목적만).
    실제 환경에서는 /app/dataset/imagenet_penalty/ 폴더의 소규모 서브셋 사용 권장.
    가정: /app/dataset/imagenet_penalty/ 없으면 FakeData fallback.
    """
    imagenet_path = Path("/app/dataset/imagenet_penalty")
    if imagenet_path.exists():
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        ds = datasets.ImageFolder(str(imagenet_path), transform=transform)
    else:
        from torchvision.datasets import FakeData
        from torchvision import transforms
        ds = FakeData(
            size=1000,
            image_size=(3, image_size, image_size),
            transform=transforms.ToTensor(),
        )
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=2, drop_last=True)
```

---

### B.3 이미지 전처리 알고리즘

아래 구현이 `utils/image_utils.py`의 `apply_filter()` 내부 로직이다. 수식과 파라미터가 완전히 확정되어 있으므로 이대로 구현한다.

#### B.3.1 Homomorphic Filter

```python
import numpy as np
import cv2
from PIL import Image

def _homomorphic_filter(
    image: Image.Image,   # RGB PIL Image
    sigma: float,         # 가우시안 필터의 표준편차
    gamma_H: float,       # 고주파 게인 (> 1.0)
    gamma_L: float,       # 저주파 게인 (< 1.0)
    cutoff: float,        # 주파수 컷오프 반경 (픽셀 단위)
    normalize: bool,      # True: 출력을 [0, 255] 정규화
) -> Image.Image:
    """
    주파수 도메인 기반 조명 정규화 필터.
    각 채널(R, G, B)에 독립적으로 적용한다.

    알고리즘:
      1. 채널별로 분리
      2. float32로 변환, [0,1] 범위
      3. log(1 + I) 변환 (log 도메인으로 이동)
      4. 2D FFT → 주파수 도메인
      5. Gaussian High-Pass 필터 H(u,v) 적용:
         H(u,v) = (gamma_H - gamma_L) * [1 - exp(-D²(u,v) / (2*cutoff²))] + gamma_L
         D(u,v) = sqrt((u-R)² + (v-C)²)  (R,C: 영상 중심)
      6. 역 FFT → 공간 도메인
      7. exp(result) - 1 변환 (log 역변환)
      8. normalize=True이면 cv2.normalize()로 [0,255] 클리핑
      9. uint8로 변환
    """
    img_np = np.array(image, dtype=np.float32) / 255.0  # (H, W, 3), [0,1]
    rows, cols = img_np.shape[:2]
    crow, ccol = rows // 2, cols // 2

    # 주파수 필터 사전 계산 (채널 공통)
    y_idx, x_idx = np.mgrid[-crow:rows - crow, -ccol:cols - ccol]
    D_sq = (x_idx ** 2 + y_idx ** 2).astype(np.float32)
    H = (gamma_H - gamma_L) * (1.0 - np.exp(-D_sq / (2.0 * cutoff ** 2))) + gamma_L
    # H shape: (H, W), float32

    channels_out = []
    for c in range(3):
        channel = img_np[:, :, c]

        # Log 변환
        log_ch = np.log1p(channel)

        # FFT + fftshift
        fft_ch = np.fft.fftshift(np.fft.fft2(log_ch))

        # 필터 적용
        filtered = fft_ch * H

        # 역 FFT
        img_back = np.real(np.fft.ifft2(np.fft.ifftshift(filtered)))

        # Exp 역변환
        result = np.expm1(img_back)

        if normalize:
            result = cv2.normalize(
                result, None, 0.0, 255.0, cv2.NORM_MINMAX
            ).astype(np.float32)
        else:
            result = np.clip(result * 255.0, 0.0, 255.0)

        channels_out.append(result.astype(np.uint8))

    out_np = np.stack(channels_out, axis=2)  # (H, W, 3)
    return Image.fromarray(out_np, mode="RGB")
```

#### B.3.2 Histogram Equalization (HE)

```python
def _histogram_equalization(image: Image.Image) -> Image.Image:
    """
    각 채널(R, G, B)에 독립적으로 히스토그램 평탄화 적용.
    cv2.equalizeHist()를 채널별로 사용.
    파라미터 없음.
    """
    img_np = np.array(image)  # (H, W, 3), uint8
    channels = [cv2.equalizeHist(img_np[:, :, c]) for c in range(3)]
    return Image.fromarray(np.stack(channels, axis=2), mode="RGB")
```

#### B.3.3 CLAHE

```python
def _clahe_filter(image: Image.Image, clip_limit: float) -> Image.Image:
    """
    각 채널(R, G, B)에 독립적으로 CLAHE 적용.
    tile_grid_size: (8, 8) 고정 (MVP 파라미터 노출 없음).
    clip_limit: model_config의 clip_limit 값.
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8),     # 고정
    )
    img_np = np.array(image)  # (H, W, 3), uint8
    channels = [clahe.apply(img_np[:, :, c]) for c in range(3)]
    return Image.fromarray(np.stack(channels, axis=2), mode="RGB")
```

#### B.3.4 Resize + Padding

```python
def _resize_with_padding(image: Image.Image, target_size: int) -> Image.Image:
    """
    비율 유지 Resize 후 검정(0,0,0) 패딩으로 target_size × target_size 생성.
    resize_mode = "padding" (고정).

    알고리즘:
      1. scale = min(target_size / W, target_size / H)
      2. new_W = int(W * scale), new_H = int(H * scale)
      3. 리사이즈: image.resize((new_W, new_H), Image.LANCZOS)
      4. 검정 배경 캔버스: Image.new("RGB", (target_size, target_size), (0,0,0))
      5. paste: offset_x = (target_size - new_W) // 2
                offset_y = (target_size - new_H) // 2
    """
    W, H = image.size
    scale = min(target_size / W, target_size / H)
    new_W = int(W * scale)
    new_H = int(H * scale)
    resized = image.resize((new_W, new_H), Image.LANCZOS)

    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    offset_x = (target_size - new_W) // 2
    offset_y = (target_size - new_H) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas
```

#### B.3.5 apply_filter() 통합 구현

```python
def apply_filter(
    image: Image.Image,
    method: str,
    params: dict | None,
) -> Image.Image:
    """
    method에 따라 B.3.1 ~ B.3.3 중 하나를 호출.
    입력/출력 모두 RGB PIL.Image.
    """
    if method == "none" or params is None:
        return image
    elif method == "homomorphic":
        return _homomorphic_filter(
            image,
            sigma=params.get("sigma", 10.0),
            gamma_H=params.get("gamma_H", 1.5),
            gamma_L=params.get("gamma_L", 0.5),
            cutoff=params.get("cutoff", 30.0),
            normalize=params.get("normalize", True),
        )
    elif method == "he":
        return _histogram_equalization(image)
    elif method == "clahe":
        return _clahe_filter(image, clip_limit=params.get("clip_limit", 2.0))
    else:
        raise ValueError(f"알 수 없는 전처리 방식: {method}")
```

---

### B.4 EfficientAD 구현

#### B.4.1 model_config → EfficientAd 파라미터 매핑

| model_config 키 | Anomalib EfficientAd 파라미터 | 타입 | 비고 |
|----------------|-------------------------------|------|------|
| `params.model_size` | `model_size` | `EfficientAdModelSize` | SIZE_MAP으로 변환 |
| `params.learning_rate` | `lr` | `float` | |
| `params.weight_decay` | `weight_decay` | `float` | |
| `params.out_channels` | `teacher_out_channels` | `int` | |
| `params.padding` | `padding` | `bool` | |
| `params.ae_loss_weight` | `map_combination_alpha` | `float` | ae 가중치 |
| `params.st_loss_weight` | 암묵적: `1 - map_combination_alpha` | | |
| `params.autoencoder_lr` | `autoencoder_lr` | `float` | 고급 설정 |
| `params.autoencoder_weight_decay` | `autoencoder_weight_decay` | `float` | 고급 설정 |
| `params.imagenet_penalty_weight` | `penalized_normalized` | `bool` + weight | True if > 0 |
| `params.train_steps` | 학습 루프 max_steps | `int` | Anomalib 생성자 아님, 루프에서 사용 |
| `common.image_size` | 데이터 전처리에서 처리, 모델 생성자 불필요 | | |

```python
def _create_efficientad_model(model_config: dict) -> EfficientAd:
    params = model_config["params"]
    return EfficientAd(
        teacher_out_channels=params["out_channels"],
        model_size=SIZE_MAP[params["model_size"]],
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
        padding=params["padding"],
        map_combination_alpha=params["ae_loss_weight"],
        autoencoder_lr=params.get("autoencoder_lr", params["learning_rate"]),
        autoencoder_weight_decay=params.get("autoencoder_weight_decay", 1e-5),
        penalized_normalized=(params.get("imagenet_penalty_weight", 1.0) > 0),
    )
```

#### B.4.2 EfficientAD 학습 루프

EfficientAD는 내부적으로 다음 컴포넌트를 가진다:
- `teacher`: 사전학습 패치 기술자 네트워크 (고정)
- `student`: teacher를 모방하는 네트워크 (학습)
- `autoencoder`: 이상 탐지를 위한 AE (학습)

```python
def _train_efficientad(
    model: EfficientAd,
    train_loader: DataLoader,
    penalty_loader: DataLoader,
    model_config: dict,
    device: torch.device,
    stop_event: threading.Event,
    result_queue: queue.Queue,
    log_file: Path,
) -> tuple[bool, float]:
    """
    반환: (completed: bool, final_loss: float)
    completed=True: 정상 완료
    completed=False: stop_event로 중단
    """
    params = model_config["params"]
    total_steps = params["train_steps"]
    report_every = 500    # 500 step마다 progress 보고 (가정 A-08)

    model = model.to(device)
    model.train()

    # 옵티마이저 생성
    optimizer_st  = _build_optimizer(model.student, params)
    optimizer_ae  = _build_optimizer(model.autoencoder, {
        **params,
        "learning_rate": params.get("autoencoder_lr", params["learning_rate"]),
        "weight_decay":  params.get("autoencoder_weight_decay", 1e-5),
    })

    # 스케줄러 생성
    scheduler_st = _build_scheduler(optimizer_st, params, total_steps)
    scheduler_ae = _build_scheduler(optimizer_ae, params, total_steps)

    # 무한 반복 DataLoader iterator
    train_iter   = _infinite_loader(train_loader)
    penalty_iter = _infinite_loader(penalty_loader)

    start_time = time.time()
    step = 0
    last_loss = 0.0

    while step < total_steps:
        if stop_event.is_set():
            return False, last_loss

        batch         = next(train_iter)
        penalty_batch = next(penalty_iter)

        images  = batch["image"].to(device)           # (B, C, H, W)
        penalty = penalty_batch[0].to(device)         # FakeData returns (img, label)

        # Forward
        loss_dict = model.training_step_impl(
            images=images,
            penalty_images=penalty,
        )
        # loss_dict: {"loss_st": Tensor, "loss_ae": Tensor, "loss_total": Tensor}

        total_loss = loss_dict["loss_total"]
        last_loss  = total_loss.item()

        # Backward
        optimizer_st.zero_grad()
        optimizer_ae.zero_grad()
        total_loss.backward()
        optimizer_st.step()
        optimizer_ae.step()
        scheduler_st.step()
        scheduler_ae.step()

        step += 1

        if step % report_every == 0:
            elapsed = time.time() - start_time
            msg = {
                "type":    "progress",
                "step":    step,
                "total":   total_steps,
                "loss":    round(last_loss, 6),
                "elapsed": round(elapsed, 1),
            }
            result_queue.put(msg)
            log_line = (
                f"{_now_kst()}\t"
                f"[Step {step}/{total_steps}] "
                f"Loss: {last_loss:.4f} | "
                f"경과: {elapsed:.1f}s"
            )
            result_queue.put({"type": "log", "message": log_line})
            _append_log(log_file, log_line)

    return True, last_loss


def _build_optimizer(
    params_group,
    config: dict,
) -> torch.optim.Optimizer:
    opt_name = config.get("optimizer", "adam").lower()
    lr = config["learning_rate"]
    wd = config["weight_decay"]
    if opt_name == "adam":
        return torch.optim.Adam(params_group.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return torch.optim.AdamW(params_group.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(params_group.parameters(), lr=lr,
                               weight_decay=wd, momentum=0.9)
    raise ValueError(f"지원하지 않는 옵티마이저: {opt_name}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    params: dict,
    total_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    scheduler_name = params.get("scheduler", "StepLR")
    if scheduler_name == "StepLR":
        step_size = params.get("lr_decay_epochs", 50000)
        gamma     = params.get("lr_decay_factor", 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )
    raise ValueError(f"지원하지 않는 스케줄러: {scheduler_name}")


def _infinite_loader(loader: DataLoader):
    """DataLoader를 무한 반복하는 generator."""
    while True:
        for batch in loader:
            yield batch
```

#### B.4.3 EfficientAD training_step_impl

Anomalib의 EfficientAd 모델 객체에 `training_step_impl`이 없는 경우, 직접 forward를 호출한다:

```python
# model_factory.py 내 _efficientad_training_step() 헬퍼

def _efficientad_training_step(
    model: EfficientAd,
    images: torch.Tensor,            # (B, C, H, W)
    penalty_images: torch.Tensor,    # (B, C, H, W)
) -> dict:
    """
    EfficientAd.training_step()을 직접 사용하거나,
    Anomalib 버전이 다를 경우 아래 fallback 사용.
    반환: {"loss_total": Tensor}
    """
    # Anomalib v1.0 방식 시도
    if hasattr(model, "training_step"):
        batch = {"image": images, "penalty_images": penalty_images}
        loss = model.training_step(batch, batch_idx=0)
        if isinstance(loss, dict):
            return loss
        return {"loss_total": loss}

    # Fallback: student-teacher loss 직접 계산
    with torch.no_grad():
        teacher_out = model.teacher(images)   # (B, C', H', W')
    student_out = model.student(images)
    ae_out      = model.autoencoder(images)

    loss_st = torch.mean((teacher_out - student_out) ** 2)
    loss_ae = torch.mean((images - ae_out) ** 2)
    alpha   = model.map_combination_alpha if hasattr(model, "map_combination_alpha") else 0.5
    loss_total = alpha * loss_ae + (1 - alpha) * loss_st
    return {"loss_st": loss_st, "loss_ae": loss_ae, "loss_total": loss_total}
```

---

### B.5 PatchCore 구현

#### B.5.1 model_config → Patchcore 파라미터 매핑

| model_config 키 | Anomalib Patchcore 파라미터 | 타입 | 비고 |
|----------------|----------------------------|------|------|
| `params.backbone` | `backbone` | `str` | `"wide_resnet50_2"` 등 |
| `params.pretrained_source` | `pre_trained` | `bool` | `"torchvision"` → `True`, `"local"` → `False` |
| `params.pretrained_path` | 별도 `torch.load()` 후 `model.backbone.load_state_dict()` | | `pretrained_source == "local"`인 경우 |
| `params.coreset_sampling_ratio` | `coreset_sampling_ratio` | `float` | |
| `params.neighbourhood_kernel_size` | `num_neighbors` | `int` | |
| `params.knn` | `num_neighbors` | `int` | 고급 설정. neighbourhood_kernel_size와 동일 파라미터, knn 우선 |
| `params.top_k_ratio` | `sampling_ratio` (내부 사용) | `float` | Anomalib 버전에 따라 다름 |
| `params.max_train` | DataLoader 샘플 수 제한 (`Subset` 사용) | `int` | |

```python
def _create_patchcore_model(model_config: dict) -> Patchcore:
    params = model_config["params"]
    pre_trained = (params["pretrained_source"] == "torchvision")

    # knn 파라미터: neighbourhood_kernel_size와 knn 중 knn 우선
    num_neighbors = params.get("knn", params["neighbourhood_kernel_size"])

    model = Patchcore(
        backbone=params["backbone"],
        layers=["layer2", "layer3"],    # 고정값: WideResNet50/ResNet 기준 최적 레이어
        pre_trained=pre_trained,
        coreset_sampling_ratio=params["coreset_sampling_ratio"],
        num_neighbors=num_neighbors,
    )

    # 로컬 가중치 로드
    if not pre_trained and params.get("pretrained_path"):
        state_dict = torch.load(params["pretrained_path"], map_location="cpu")
        # state_dict가 {"model": {...}} 형태일 수 있음
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.backbone.load_state_dict(state_dict, strict=False)

    return model
```

#### B.5.2 PatchCore 학습 루프 (특징 추출 + Coreset)

PatchCore는 그라디언트 업데이트 없이 특징 추출 → coreset 구성 → KNN 인덱스 빌드 순서로 동작한다.

```python
def _train_patchcore(
    model: Patchcore,
    train_loader: DataLoader,
    model_config: dict,
    device: torch.device,
    stop_event: threading.Event,
    result_queue: queue.Queue,
    log_file: Path,
) -> tuple[bool, float]:
    """
    PatchCore는 단일 에포크 특징 추출 후 메모리 뱅크 구성.
    반환: (completed: bool, 0.0)  — loss 개념 없음
    """
    params = model_config["params"]
    max_train = params.get("max_train", 1000)

    model = model.to(device)
    model.eval()

    total_batches = min(len(train_loader), max_train // train_loader.batch_size + 1)
    all_features: list[torch.Tensor] = []
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if stop_event.is_set():
                return False, 0.0

            if batch_idx >= total_batches:
                break

            images = batch["image"].to(device)

            # 멀티 레이어 특징 추출
            features = _extract_patchcore_features(model, images)
            # features: (B * N_patches, C_feature) 형태로 reshape
            all_features.append(features.cpu())

            elapsed = time.time() - start_time
            pct = (batch_idx + 1) / total_batches
            msg = {
                "type":    "progress",
                "step":    batch_idx + 1,
                "total":   total_batches,
                "loss":    0.0,           # PatchCore는 loss 없음
                "elapsed": round(elapsed, 1),
            }
            result_queue.put(msg)

            log_line = (
                f"{_now_kst()}\t"
                f"[배치 {batch_idx+1}/{total_batches}] "
                f"특징 추출 중 | 경과: {elapsed:.1f}s"
            )
            result_queue.put({"type": "log", "message": log_line})
            _append_log(log_file, log_line)

    # 전체 특징 벡터 concat
    feature_stack = torch.cat(all_features, dim=0)  # (N_total, C_feature)

    # Coreset 서브샘플링 (Anomalib 내장 메서드 사용)
    if hasattr(model, "fit"):
        model.fit(feature_stack)
    else:
        # Fallback: 직접 coreset 샘플링
        coreset_size = max(1, int(len(feature_stack) * params["coreset_sampling_ratio"]))
        indices = torch.randperm(len(feature_stack))[:coreset_size]
        model.memory_bank = feature_stack[indices].to(device)

    elapsed_total = time.time() - start_time
    log_line = f"{_now_kst()}\t[완료] 메모리 뱅크 구성 완료 | 경과: {elapsed_total:.1f}s"
    result_queue.put({"type": "log", "message": log_line})
    _append_log(log_file, log_line)

    return True, 0.0


def _extract_patchcore_features(
    model: Patchcore,
    images: torch.Tensor,   # (B, C, H, W)
) -> torch.Tensor:
    """
    model.backbone의 layer2, layer3 출력을 추출하여
    패치 레벨 특징 벡터로 반환.
    반환 shape: (B * H' * W', C_combined)
    """
    features = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            features[name] = output
        return hook

    # layer2, layer3에 hook 등록
    hooks.append(model.backbone.layer2.register_forward_hook(make_hook("layer2")))
    hooks.append(model.backbone.layer3.register_forward_hook(make_hook("layer3")))

    with torch.no_grad():
        _ = model.backbone(images)

    for h in hooks:
        h.remove()

    # 두 레이어 특징을 동일 공간 크기로 맞춰 concat
    f2 = features["layer2"]  # (B, C2, H2, W2)
    f3 = features["layer3"]  # (B, C3, H3, W3)

    # f3를 f2 크기로 upsample
    f3_up = nn.functional.interpolate(f3, size=f2.shape[-2:], mode="bilinear",
                                       align_corners=False)
    combined = torch.cat([f2, f3_up], dim=1)  # (B, C2+C3, H2, W2)

    B, C, H, W = combined.shape
    # (B, C, H, W) → (B*H*W, C) 패치 단위로 reshape
    patches = combined.permute(0, 2, 3, 1).reshape(B * H * W, C)
    return patches
```

---

### B.6 TrainingWorker 통합 구현

`utils/training_worker.py`의 `TrainingWorker.run()` 전체 구현:

```python
class TrainingWorker(threading.Thread):

    def __init__(
        self,
        model_config: dict,
        preprocessing_config: dict,
        dataset_path: str,
        device_info: dict,
        exp_id: str,
        stop_event: threading.Event,
        result_queue: queue.Queue,
    ):
        super().__init__(daemon=True)
        self.model_config        = model_config
        self.preprocessing_config = preprocessing_config
        self.dataset_path        = dataset_path
        self.device_str          = device_info["device"]   # "cuda" | "cpu"
        self.exp_id              = exp_id
        self.stop_event          = stop_event
        self.result_queue        = result_queue
        self.log_file            = Path(f"./logs/{exp_id}.log")

    def run(self) -> None:
        try:
            self._run_impl()
        except Exception as e:
            import traceback
            self.result_queue.put({
                "type":      "error",
                "exception": e,
                "traceback": traceback.format_exc(),
            })

    def _run_impl(self) -> None:
        # 1. 재현성 시드 고정 (R-SEED-01)
        seed = self.model_config["random_seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device_str == "cuda":
            torch.cuda.manual_seed_all(seed)

        device = torch.device(self.device_str)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # 2. DataLoader 구성
        train_loader, test_loader = build_dataloaders(
            dataset_path=self.dataset_path,
            preprocessing_config=self.preprocessing_config,
            batch_size=self.model_config["batch_size"],
            random_seed=seed,
        )

        # 3. 시작 로그
        start_msg = (
            f"{_now_kst()}\t"
            f"[시작] 실험: {self.exp_id} | "
            f"모델: {self.model_config['model_type']} | "
            f"디바이스: {self.device_str}"
        )
        self.result_queue.put({"type": "log", "message": start_msg})
        _append_log(self.log_file, start_msg)

        model_type = self.model_config["model_type"]

        # 4. 모델 생성 + 학습
        if model_type == "efficientad":
            model = _create_efficientad_model(self.model_config)
            penalty_loader = build_imagenet_penalty_loader(
                batch_size=self.model_config["params"].get("penalty_batch_size", 8),
                image_size=self.model_config["image_size"],
                device=self.device_str,
            )
            completed, _ = _train_efficientad(
                model, train_loader, penalty_loader,
                self.model_config, device,
                self.stop_event, self.result_queue, self.log_file,
            )

        elif model_type == "patchcore":
            model = _create_patchcore_model(self.model_config)
            completed, _ = _train_patchcore(
                model, train_loader,
                self.model_config, device,
                self.stop_event, self.result_queue, self.log_file,
            )

        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

        if not completed:
            self.result_queue.put({"type": "stopped"})
            return

        # 5. 전체 테스트셋 추론 → Anomaly Score + Map 수집
        self.result_queue.put({"type": "log",
                               "message": f"{_now_kst()}\t[평가] 테스트셋 추론 중..."})
        y_true, anomaly_scores, anomaly_maps = _run_full_test_inference(
            model, test_loader, device, self.stop_event
        )

        if self.stop_event.is_set():
            self.result_queue.put({"type": "stopped"})
            return

        # 6. 완료 메시지 전송
        self.result_queue.put({
            "type":          "completed",
            "model":         model.cpu(),     # CPU로 이동 (저장용)
            "y_true":        y_true,
            "anomaly_scores": anomaly_scores,
            "anomaly_maps":  anomaly_maps,    # dict[image_path: str → np.ndarray (H,W)]
        })
```

---

### B.7 추론 및 Anomaly Map 생성

#### B.7.1 전체 테스트셋 추론

```python
def _run_full_test_inference(
    model,
    test_loader: DataLoader,
    device: torch.device,
    stop_event: threading.Event,
) -> tuple[list[int], list[float], dict]:
    """
    반환:
      y_true:         list[int]   — 이미지별 GT 레이블 (0=정상, 1=결함)
      anomaly_scores: list[float] — 이미지별 Anomaly Score (이미지 레벨)
      anomaly_maps:   dict[str → np.ndarray(H,W)] — 경로→픽셀별 Anomaly Map
    """
    model = model.to(device)
    model.eval()

    y_true         = []
    anomaly_scores = []
    anomaly_maps   = {}

    with torch.no_grad():
        for batch in test_loader:
            if stop_event.is_set():
                return y_true, anomaly_scores, anomaly_maps

            image      = batch["image"].to(device)      # (1, C, H, W)
            label      = batch["label"].item()           # int
            image_path = batch["image_path"][0]          # str

            anomaly_map = _get_anomaly_map(model, image)
            # anomaly_map: np.ndarray (H, W), float32

            # 이미지 레벨 Score = Anomaly Map의 최댓값
            score = float(anomaly_map.max())

            y_true.append(label)
            anomaly_scores.append(round(score, 6))
            anomaly_maps[image_path] = anomaly_map

    return y_true, anomaly_scores, anomaly_maps


def _get_anomaly_map(model, image: torch.Tensor) -> np.ndarray:
    """
    단일 이미지 추론 → Anomaly Map (H, W) float32 반환.

    EfficientAd:
      model.forward(image) → {"anomaly_map": Tensor (1,1,H,W)}
      또는 model.predict(image) → AnomalyMapOrScore

    Patchcore:
      1. layer2, layer3 특징 추출
      2. 각 패치 특징에 대해 memory_bank와의 최소 k-NN 거리 계산
      3. 패치 거리 맵을 원본 이미지 크기로 upsample

    공통 후처리:
      - 출력 shape: (H, W) float32
      - H, W: image 입력 크기 (image_size × image_size)
    """
    if hasattr(model, "anomaly_map_generator"):
        # Anomalib v1.0 EfficientAd/Patchcore 공통 경로
        output = model(image)
        if isinstance(output, dict) and "anomaly_map" in output:
            amap = output["anomaly_map"]
        elif hasattr(output, "anomaly_map"):
            amap = output.anomaly_map
        else:
            amap = output
        # amap: Tensor (1, 1, H, W) 또는 (1, H, W) 또는 (H, W)
        amap = amap.squeeze().cpu().numpy().astype(np.float32)
        return amap

    # Fallback: PatchCore 수동 추론
    if hasattr(model, "memory_bank"):
        features = _extract_patchcore_features(model, image)  # (H'*W', C)
        # k-NN 거리 계산
        dists = torch.cdist(
            features.unsqueeze(0),
            model.memory_bank.unsqueeze(0),
            p=2,
        ).squeeze(0)  # (H'*W', M)
        # Top-k 최솟값 평균
        k = min(model.num_neighbors if hasattr(model, "num_neighbors") else 9,
                dists.shape[1])
        patch_scores, _ = torch.topk(dists, k, dim=1, largest=False)
        patch_scores = patch_scores.mean(dim=1)  # (H'*W',)
        # 패치 맵 복원
        spatial_size = int(patch_scores.shape[0] ** 0.5)
        patch_map = patch_scores.reshape(spatial_size, spatial_size).cpu().numpy()
        # 원본 이미지 크기로 upsample
        H = W = image.shape[-1]
        amap = cv2.resize(patch_map, (W, H), interpolation=cv2.INTER_LINEAR)
        return amap.astype(np.float32)

    raise NotImplementedError(f"알 수 없는 모델 구조: {type(model)}")
```

#### B.7.2 단일 이미지 추론 (탭6용)

```python
# utils/model_factory.py

def load_model_for_inference(
    model_path: str,         # experiment.model_path = "./models/{exp_id}/"
    model_config: dict,
    device: str,
) -> object:
    """
    저장된 state_dict 로드 후 추론용 모델 반환.
    model_config.model_type에 따라 EfficientAd 또는 Patchcore 초기화.
    """
    pth_path = Path(model_path) / "model_state_dict.pth"
    state_dict = torch.load(str(pth_path), map_location=device)

    model_type = model_config["model_type"]
    if model_type == "efficientad":
        model = _create_efficientad_model(model_config)
    elif model_type == "patchcore":
        model = _create_patchcore_model(model_config)
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def run_inference(
    model,
    image_path: str,
    preprocessing_config: dict,
) -> np.ndarray:
    """
    단일 이미지 추론. Anomaly Map (H, W) float32 반환.
    학습과 동일한 preprocessing_config 사용 (DA-05 준수).
    """
    _, image_tensor = apply_preprocessing(image_path, preprocessing_config)
    image_tensor = image_tensor.unsqueeze(0)   # (1, C, H, W)
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        anomaly_map = _get_anomaly_map(model, image_tensor)

    return anomaly_map   # (H, W) float32
```

---

### B.8 Anomaly Score 정규화 및 Threshold 계산

#### B.8.1 Score 정규화

원시 Anomaly Score는 모델에 따라 스케일이 다르다. 탭5 비교 차트에서 모델 간 Score를 직접 비교하지 않으므로 MVP에서는 정규화를 적용하지 않는다.

각 실험의 Score는 해당 실험 내에서만 의미 있는 상대적 값이다.

#### B.8.2 Threshold 계산

```python
# utils/metrics.py

def compute_threshold(
    normal_scores: list[float],   # 정상(label=0) 이미지의 Anomaly Score 목록
    threshold_method: str,        # "percentile" | "absolute"
    threshold_value: float,       # percentile: 0~100, absolute: 직접 값
) -> float:
    """
    threshold_method == "percentile":
      np.percentile(normal_scores, threshold_value)
      예: normal_scores의 95번째 백분위수

    threshold_method == "absolute":
      threshold_value를 그대로 반환 (사용자 직접 지정)
    """
    if threshold_method == "percentile":
        if len(normal_scores) == 0:
            return 0.5   # 정상 이미지가 없는 경우 fallback
        return float(np.percentile(normal_scores, threshold_value))
    elif threshold_method == "absolute":
        return float(threshold_value)
    raise ValueError(f"지원하지 않는 threshold_method: {threshold_method}")
```

학습 완료 후 threshold 계산 흐름:
```python
# training_worker.py _run_impl() 내부 — 완료 후
normal_scores = [
    score for score, label in zip(anomaly_scores, y_true)
    if label == 0
]
threshold = compute_threshold(
    normal_scores=normal_scores,
    threshold_method=model_config["threshold_method"],
    threshold_value=model_config["threshold_value"],
)
# 이후 metrics 계산에 이 threshold 사용
```

---

### B.9 메트릭 계산

```python
# utils/metrics.py — compute_metrics() 전체 구현

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    fbeta_score, roc_auc_score, roc_curve,
    confusion_matrix as sk_confusion_matrix,
)

def compute_metrics(
    y_true: list[int],
    anomaly_scores: list[float],
    threshold: float,
) -> dict:
    """
    반환: 00_Global_Context 1.2절 metrics 스키마
    """
    y_pred = [1 if s >= threshold else 0 for s in anomaly_scores]

    # zero_division=0: 예측이 전부 한쪽인 경우 0 반환 (예외 발생 금지)
    accuracy  = round(accuracy_score(y_true, y_pred), 6)
    precision = round(precision_score(y_true, y_pred, zero_division=0), 6)
    recall    = round(recall_score(y_true, y_pred, zero_division=0), 6)
    f1        = round(fbeta_score(y_true, y_pred, beta=1, zero_division=0), 6)
    f2        = round(fbeta_score(y_true, y_pred, beta=2, zero_division=0), 6)

    # AUC: 정상/결함 레이블이 1종류만 있으면 0.0 처리
    try:
        auc = round(roc_auc_score(y_true, anomaly_scores), 6)
    except ValueError:
        auc = 0.0

    cm = sk_confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "accuracy":         accuracy,
        "precision":        precision,
        "recall":           recall,
        "f1_score":         f1,
        "f2_score":         f2,
        "auc":              auc,
        "confusion_matrix": {"tp": int(tp), "fp": int(fp),
                             "tn": int(tn), "fn": int(fn)},
        "anomaly_scores":   [round(s, 6) for s in anomaly_scores],
        "image_labels":     [int(l) for l in y_true],
    }


def compute_roc_curve(
    y_true: list[int],
    anomaly_scores: list[float],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    반환: (fpr, tpr, auc)
    정상/결함 레이블이 1종류만 있으면 ([0,1], [0,1], 0.0) 반환
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
        auc = round(roc_auc_score(y_true, anomaly_scores), 6)
        return fpr, tpr, auc
    except ValueError:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.0
```

---

## C. System & Data Design

### C.1 ML 관련 파일 I/O

| 이벤트 | 파일 경로 | 형식 | 수행 위치 |
|--------|-----------|------|-----------|
| 학습 완료 — 모델 저장 | `./models/{exp_id}/model_state_dict.pth` | `torch.save(model.state_dict(), ...)` | `tab4_training.py` (메인 스레드) |
| 학습 완료 — 설정 저장 | `./models/{exp_id}/configs.yaml` | `shutil.copy("./configs.yaml", ...)` | `tab4_training.py` (메인 스레드) |
| 학습 로그 | `./logs/{exp_id}.log` | 텍스트, 탭 구분자 | `training_worker.py` (백그라운드) |
| 추론용 모델 로드 | `./models/{exp_id}/model_state_dict.pth` | `torch.load(...)` | `model_factory.load_model_for_inference()` |

> 모델 파일 저장은 백그라운드 스레드가 아닌 **메인 스레드**에서 수행한다 (R-THREAD-01). `result_queue`로 `completed` 메시지를 수신한 후 메인 스레드에서 `torch.save()`를 호출한다.

### C.2 anomaly_maps 메모리 관리

`result_queue`를 통해 전달된 `anomaly_maps` (dict[str → np.ndarray])는 크기가 클 수 있다. 탭6 세션 내 캐시 방식:

```python
# tabs/tab4_training.py — 완료 처리 시
msg = result_queue.get_nowait()   # type == "completed"

# session_state에 저장 (탭6에서 재사용)
# 각 anomaly_map: float32 (256×256) ≈ 256KB
# 테스트 이미지 100장 기준 ≈ 25MB — session_state 허용 범위
st.session_state[f"_anomaly_maps_{exp_id}"] = msg["anomaly_maps"]
```

탭6에서 사용:
```python
exp_id = st.session_state.selected_experiment_id
cached_maps = st.session_state.get(f"_anomaly_maps_{exp_id}")
if cached_maps and image_path in cached_maps:
    anomaly_map = cached_maps[image_path]
else:
    # 캐시 미스: 모델 재로드 후 추론 (가정 A-05)
    model = load_model_for_inference(model_path, model_config, device)
    anomaly_map = run_inference(model, image_path, preprocessing_config)
```

---

## D. API Contracts

```
N/A — REST API 없음. 모듈 간 함수 인터페이스는 B절 코드 시그니처로 확정.
```

---

## E. AI/ML Details

### E.1 모델 선택 최종 근거

| 항목 | EfficientAD | PatchCore |
|------|-------------|-----------|
| **학습 방식** | 그라디언트 기반 (70,000 steps) | 비파라메트릭 (단일 에포크 특징 추출) |
| **추론 방식** | Student-Teacher 차이 맵 + AE 재구성 오차 | KNN 거리 기반 패치 이상도 |
| **결함 탐지 강점** | 텍스처 이상, 미세 결함 | 구조적 이상, 형상 변화 |
| **VRAM 요구량** | ~3GB (medium, batch=8) | ~6GB (WideResNet50 + memory bank) |
| **볼트 결함 적합성** | scratch, crack 탐지에 강 | thread 손상 등 구조적 변형에 강 |

### E.2 레이어 선택 근거 (PatchCore)

`layer2`, `layer3` 고정 사용 이유:
- layer1: 너무 저수준 (엣지, 색상만)
- layer2: 중간 수준 텍스처 표현 (C=512)
- layer3: 고수준 의미론적 표현 (C=1024)
- layer4: 너무 고수준 (공간 해상도 낮음)

WideResNet50 기준 layer2+layer3 조합이 MVTec AD 벤치마크에서 최고 성능이다.

### E.3 Anomaly Map 해상도

| 모델 | Anomaly Map 원시 해상도 | 출력 해상도 |
|------|------------------------|------------|
| EfficientAD | image_size × image_size (모델 내부 upsampling) | image_size × image_size |
| PatchCore | ~image_size/8 × image_size/8 (layer2 stride) | image_size × image_size (cv2.resize) |

PatchCore의 경우 `cv2.resize(patch_map, (W, H), interpolation=cv2.INTER_LINEAR)`로 원본 크기로 upsample한다.

---

## F. Non-Functional Requirements

[00_Global_Context_Document.md 6절](./00_Global_Context_Document.md#6-global-non-functional-requirements) 전체 상속.

ML 레이어 추가 요구사항:

| 항목 | 요구사항 |
|------|----------|
| **EfficientAD VRAM** | batch_size=8, medium 기준 ≤ 8GB. T4 16GB에서 여유 있음 |
| **PatchCore VRAM** | WideResNet50, 특징 추출 배치=32 기준 ≤ 8GB |
| **전처리 속도** | 1024×1024 이미지 Homomorphic 적용 ≤ 200ms/장 (CPU) |
| **anomaly_map 캐시 크기** | float32, 256×256, 100장 = ~25MB. 허용 범위 |
| **모델 파일 크기** | EfficientAD-medium: ~200MB / PatchCore WRN50: ~400MB |
| **추론 속도 (탭6)** | 단일 이미지 추론 ≤ 500ms (CUDA 기준) |

---

## G. Observability

[00_Global_Context_Document.md 7절](./00_Global_Context_Document.md#7-observability-standards) 전체 상속.

ML 레이어 추가 로그 이벤트:

| 이벤트 | 위치 | 형식 |
|--------|------|------|
| 모델 초기화 완료 | `training_worker._run_impl()` | `[초기화] {model_type} 모델 준비 완료` |
| Coreset 구성 완료 | `_train_patchcore()` | `[완료] 메모리 뱅크 {N}개 벡터 구성 완료` |
| 추론 시작 | `_run_full_test_inference()` | `[평가] 테스트셋 {N}장 추론 시작` |
| 최종 AUC | `training_worker._run_impl()` 완료 시 | `[결과] AUC: {auc:.4f}, 소요: {sec}s` |

---

## H. QA & Validation

### H.1 ML 구현 검증 기준

| # | 기준 | 검증 방법 |
|---|------|-----------|
| ML-01 | `apply_filter("homomorphic", ...)` 출력이 PIL.Image RGB이고 값 범위 [0,255] | 단위 테스트 |
| ML-02 | `apply_filter("he", ...)` 출력의 히스토그램이 원본보다 평탄함 | 히스토그램 엔트로피 비교 |
| ML-03 | `resize_with_padding(image, 256)` 출력이 정확히 256×256 | `assert out.size == (256, 256)` |
| ML-04 | 동일 seed로 EfficientAD 2회 학습 시 step 1000 기준 loss 소수점 4자리 일치 | 재현성 테스트 |
| ML-05 | `compute_metrics()` 반환 dict가 00_Global_Context 1.2절 스키마 완전 일치 | 스키마 검증 테스트 |
| ML-06 | PatchCore `_extract_patchcore_features()` 출력 shape: `(B*H'*W', C2+C3)` | assert 검증 |
| ML-07 | `_get_anomaly_map()` 출력 shape: `(image_size, image_size)` | assert 검증 |
| ML-08 | 학습 완료 후 `./models/{exp_id}/model_state_dict.pth` 로드 → 추론 정상 동작 | 통합 테스트 |

### H.2 Given-When-Then 시나리오

#### TC-ML-01: Homomorphic Filter 채널 독립성

```
Given:  256×256 RGB 이미지, gamma_H=1.5, gamma_L=0.5, sigma=10, cutoff=30, normalize=True
When:   _homomorphic_filter() 호출
Then:   출력 PIL.Image의 mode == "RGB"
        출력 shape == (256, 256, 3)
        출력 값 범위: [0, 255] (normalize=True 보장)
        R/G/B 각 채널이 서로 독립적으로 처리됨
        (채널 간 교차 영향 없음: 채널별 FFT 독립 적용)
```

#### TC-ML-02: PatchCore 재현성

```
Given:  동일 데이터셋, random_seed=42, coreset_sampling_ratio=0.1
When:   PatchCore 학습을 2회 실행
Then:   두 실험의 metrics.auc 차이 < 0.0001
        두 실험의 anomaly_scores 각 요소 차이 < 0.001
        (PatchCore는 그라디언트 없음, 재현성 매우 높아야 함)
```

#### TC-ML-03: 전처리 학습·추론 일관성 (DA-05)

```
Given:  preprocessing_config.method = "clahe", clip_limit = 2.0
        동일 이미지를 학습 DataLoader와 탭6 추론에서 각각 처리
When:   MVTecDataset.__getitem__()과 apply_preprocessing() 각각 호출
Then:   두 경로에서 생성된 image_tensor의 max 차이 < 1e-5
        (image_utils.apply_preprocessing() 단일 구현 보장)
```

#### TC-ML-04: Threshold 계산 일관성

```
Given:  정상 이미지 100장의 anomaly_scores = [0.1, 0.2, ..., 1.0] (균등 분포 가정)
        threshold_method = "percentile", threshold_value = 95
When:   compute_threshold(normal_scores, "percentile", 95) 호출
Then:   반환값 ≈ np.percentile(normal_scores, 95)
        소수점 6자리 일치
```

---

## I. Implementation Plan

이 문서가 확정되면 아래 순서로 ML 관련 모듈을 구현한다.

| 순서 | 파일 | 의존 선행 구현 | 담당 (역할분담표 기준) |
|------|------|--------------|----------------------|
| 1 | `utils/image_utils.py` (B.3절 전체) | 없음 | A (이미지 처리 배경) |
| 2 | `utils/mvtec_dataset.py` (B.2절) | image_utils.py | A |
| 3 | `utils/metrics.py` (B.9절) | 없음 | B 또는 C |
| 4 | `utils/model_factory.py` EfficientAD 파트 (B.4절) | mvtec_dataset.py | B (EfficientAD 배경) |
| 5 | `utils/model_factory.py` PatchCore 파트 (B.5절) | mvtec_dataset.py | C (PatchCore 배경) |
| 6 | `utils/training_worker.py` (B.6절) | model_factory.py | B+C 공동 |
| 7 | `tabs/tab4_training.py` (03_FR B.5절) | training_worker.py | B+C |

---

*다음 문서*: [05_Data_Model_and_Storage_Strategy.md](./05_Data_Model_and_Storage_Strategy.md)
