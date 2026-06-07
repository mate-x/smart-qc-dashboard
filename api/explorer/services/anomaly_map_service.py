"""
api/explorer/services/anomaly_map_service.py

탭5 · Anomaly Map:
    start_build(exp_id)                       비동기 build job 시작 → job_id
    get_job_status(job_id)                    job 상태 조회
    get_images(exp_id, threshold, defect_class)  이미지 목록 + 통계
    get_triplet_image(exp_id, class_name, image_name)  triplet PIL Image 반환
    get_csv(exp_id, threshold, defect_class)  CSV bytes (UTF-8 BOM)
    start_zip(exp_id, threshold, defect_class)  비동기 ZIP job 시작 → job_id
    get_zip_result(job_id)                    ZIP bytes 반환
"""
from __future__ import annotations

import asyncio
import io
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from api.explorer.jobs import (
    create_job,
    get_job,
    set_completed,
    set_failed,
    set_running,
)
from api.explorer.state import get_state
from utils.image_utils import (
    anomaly_map_to_heatmap,
    build_gt_mask_path,
    create_triplet_image,
    load_image,
)
from utils.storage import load_history

_MAX_CACHE = 3


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _get_cache(exp_id: str) -> dict | None:
    return get_state()["anomaly_map_cache"].get(exp_id)


def _set_cache(exp_id: str, data: dict) -> None:
    cache = get_state()["anomaly_map_cache"]
    while len(cache) >= _MAX_CACHE:
        oldest = min(cache, key=lambda k: cache[k].get("cached_at", 0))
        del cache[oldest]
    cache[exp_id] = {**data, "cached_at": time.time()}


# ---------------------------------------------------------------------------
# Experiment lookup
# ---------------------------------------------------------------------------

def _get_experiment(exp_id: str) -> dict:
    record = next(
        (r for r in load_history() if r.get("experiment_id") == exp_id),
        None,
    )
    if record is None:
        raise LookupError(f"실험을 찾을 수 없습니다: {exp_id}")
    return record


# ---------------------------------------------------------------------------
# Build (async)
# ---------------------------------------------------------------------------

async def start_build(exp_id: str) -> str:
    exp = _get_experiment(exp_id)

    if _get_cache(exp_id) is not None:
        job_id = create_job("build")
        set_completed(job_id)
        return job_id

    job_id = create_job("build")
    asyncio.create_task(_run_build(job_id, exp_id, exp))
    return job_id


async def _run_build(job_id: str, exp_id: str, exp: dict) -> None:
    set_running(job_id)
    try:
        data = await asyncio.to_thread(_build_sync, exp_id, exp)
        _set_cache(exp_id, data)
        set_completed(job_id)
    except Exception as e:
        set_failed(job_id, str(e))


def _build_sync(exp_id: str, exp: dict) -> dict:
    """모델 로드 → 테스트셋 순차 추론. Thread에서 실행."""
    from utils.model_factory import load_model_for_inference, run_inference
    from utils.mvtec_dataset import MVTecDataset

    image_size = exp.get("image_size", 256)
    model_config = {
        "model_type":       exp["model_type"],
        "image_size":       image_size,
        "batch_size":       16,
        "random_seed":      42,
        "params":           exp.get("model_params", {}),
        "threshold_method": exp.get("threshold_method", "percentile"),
        "threshold_value":  exp.get("threshold_value", 95.0),
    }
    preprocessing_config = {
        "method":     exp.get("preprocessing_method", "none"),
        "params":     exp.get("preprocessing_params"),
        "image_size": image_size,
    }

    device = (get_state().get("device_info") or {}).get("device", "cpu")
    model = load_model_for_inference(exp_id, exp["model_path"], model_config, device)

    test_ds = MVTecDataset(exp["dataset_path"], "test", preprocessing_config)

    image_paths: list[str] = []
    maps_list: list[np.ndarray] = []
    for i in range(len(test_ds)):
        item = test_ds[i]
        maps_list.append(run_inference(model, item["image"]))
        image_paths.append(item["image_path"])

    return {
        "anomaly_maps": np.stack(maps_list, axis=0),
        "image_paths":  image_paths,
    }


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------

def get_job_status(job_id: str) -> dict:
    job = get_job(job_id)
    if job is None:
        raise LookupError(f"Job을 찾을 수 없습니다: {job_id}")
    return {"status": job["status"], "error": job.get("error")}


# ---------------------------------------------------------------------------
# Images list
# ---------------------------------------------------------------------------

def get_images(exp_id: str, threshold: float, defect_class: str) -> dict:
    cache = _get_cache(exp_id)
    if cache is None:
        raise ValueError("Anomaly Map 캐시가 없습니다. 먼저 build를 실행하세요.")

    exp = _get_experiment(exp_id)
    metrics = exp.get("metrics") or {}
    anomaly_scores_raw: list[float] = metrics.get("anomaly_scores", [])
    image_labels: list[int]         = metrics.get("image_labels", [])
    image_paths: list[str]          = cache["image_paths"]

    # Min-Max 정규화 (0~1)
    arr = np.array(anomaly_scores_raw, dtype=np.float64)
    s_min, s_max = float(arr.min()), float(arr.max())
    if s_max > s_min:
        anomaly_scores_norm = [
            float(max(0.0, min(1.0, (s - s_min) / (s_max - s_min))))
            for s in anomaly_scores_raw
        ]
    else:
        anomaly_scores_norm = [0.0] * len(anomaly_scores_raw)

    rows = _build_table_rows(image_paths, anomaly_scores_norm, image_labels, threshold)

    if defect_class != "전체":
        rows = [r for r in rows if r["defect_class"] == defect_class]

    scores = [r["anomaly_score"] for r in rows]
    cls_counts: dict[str, int] = {}
    for r in rows:
        cls_counts[r["classification"]] = cls_counts.get(r["classification"], 0) + 1

    return {
        "images":    rows,
        "score_max": max(scores) if scores else 0.0,
        "score_avg": sum(scores) / len(scores) if scores else 0.0,
        "tp": cls_counts.get("TP", 0),
        "fp": cls_counts.get("FP", 0),
        "tn": cls_counts.get("TN", 0),
        "fn": cls_counts.get("FN", 0),
    }


def _build_table_rows(
    image_paths: list[str],
    anomaly_scores: list[float],
    image_labels: list[int],
    threshold: float,
) -> list[dict]:
    n = min(len(image_paths), len(anomaly_scores), len(image_labels))
    rows = []
    for i in range(n):
        score = anomaly_scores[i]
        label = image_labels[i]
        pred = 1 if score >= threshold else 0

        if label == 0 and pred == 1:
            cls = "FP"
        elif label == 1 and pred == 0:
            cls = "FN"
        elif label == 0:
            cls = "TN"
        else:
            cls = "TP"

        p = Path(image_paths[i])
        rows.append({
            "image_name":     p.name,
            "defect_class":   p.parent.name,
            "anomaly_score":  round(score, 6),
            "verdict":        "NG" if score >= threshold else "OK",
            "gt_match":       cls in ("TN", "TP"),
            "classification": cls,
            "image_path":     f"{p.parent.name}/{p.name}",
        })
    return rows


# ---------------------------------------------------------------------------
# Triplet image
# ---------------------------------------------------------------------------

def get_triplet_image(exp_id: str, class_name: str, image_name: str) -> Image.Image:
    cache = _get_cache(exp_id)
    if cache is None:
        raise ValueError("Anomaly Map 캐시가 없습니다. 먼저 build를 실행하세요.")

    exp = _get_experiment(exp_id)
    image_paths: list[str]  = cache["image_paths"]
    anomaly_maps: np.ndarray = cache["anomaly_maps"]
    dataset_path: str        = exp.get("dataset_path", "")

    target = (class_name, image_name)
    idx = next(
        (i for i, p in enumerate(image_paths)
         if (Path(p).parent.name, Path(p).name) == target),
        None,
    )
    if idx is None:
        raise LookupError(f"이미지를 찾을 수 없습니다: {class_name}/{image_name}")

    original    = load_image(image_paths[idx])
    gt_mask_pil = _load_gt_mask(image_paths[idx], dataset_path)
    heatmap     = anomaly_map_to_heatmap(anomaly_maps[idx])

    if gt_mask_pil is not None:
        heatmap = _overlay_contour(heatmap, gt_mask_pil)

    return create_triplet_image(original, gt_mask_pil, heatmap)


# ---------------------------------------------------------------------------
# CSV (synchronous)
# ---------------------------------------------------------------------------

def get_csv(exp_id: str, threshold: float, defect_class: str) -> bytes:
    result = get_images(exp_id, threshold, defect_class)
    df = pd.DataFrame(result["images"])
    df = df[["image_name", "defect_class", "anomaly_score", "verdict", "gt_match", "classification"]]
    df.columns = ["이미지명", "결함 유형", "Anomaly Score", "판정", "GT 일치", "오분류"]
    return df.to_csv(index=False).encode("utf-8-sig")


# ---------------------------------------------------------------------------
# ZIP (async)
# ---------------------------------------------------------------------------

async def start_zip(exp_id: str, threshold: float, defect_class: str) -> str:
    if _get_cache(exp_id) is None:
        raise ValueError("Anomaly Map 캐시가 없습니다. 먼저 build를 실행하세요.")

    job_id = create_job("zip")
    asyncio.create_task(_run_zip(job_id, exp_id, threshold, defect_class))
    return job_id


async def _run_zip(job_id: str, exp_id: str, threshold: float, defect_class: str) -> None:
    set_running(job_id)
    try:
        zip_bytes = await asyncio.to_thread(_build_zip_sync, exp_id, threshold, defect_class)
        set_completed(job_id, result=zip_bytes)
    except Exception as e:
        set_failed(job_id, str(e))


def _build_zip_sync(exp_id: str, threshold: float, defect_class: str) -> bytes:
    cache = _get_cache(exp_id)
    if cache is None:
        raise ValueError("Anomaly Map 캐시가 없습니다.")

    exp                      = _get_experiment(exp_id)
    dataset_path: str        = exp.get("dataset_path", "")
    image_paths: list[str]   = cache["image_paths"]
    anomaly_maps: np.ndarray = cache["anomaly_maps"]

    result = get_images(exp_id, threshold, defect_class)
    rows   = result["images"]

    path_to_idx = {
        f"{Path(p).parent.name}/{Path(p).name}": i
        for i, p in enumerate(image_paths)
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for row in rows:
            cache_idx = path_to_idx.get(row["image_path"])
            if cache_idx is None:
                continue

            actual_path = image_paths[cache_idx]
            stem = Path(actual_path).stem

            try:
                original = load_image(actual_path)
            except Exception:
                continue

            gt_mask_pil = _load_gt_mask(actual_path, dataset_path)
            heatmap     = anomaly_map_to_heatmap(anomaly_maps[cache_idx])

            if gt_mask_pil is not None:
                heatmap = _overlay_contour(heatmap, gt_mask_pil)

            triplet = create_triplet_image(original, gt_mask_pil, heatmap)
            img_buf = io.BytesIO()
            triplet.save(img_buf, format="PNG")
            zf.writestr(f"{exp_id}_{stem}_anomaly.png", img_buf.getvalue())

    buf.seek(0)
    return buf.getvalue()


def get_build_status(exp_id: str) -> dict:
    _get_experiment(exp_id)  # raises LookupError if not found
    cache = _get_cache(exp_id)
    if cache is None:
        return {"built": False, "image_count": 0}
    return {"built": True, "image_count": len(cache["image_paths"])}


def get_zip_result(job_id: str) -> bytes:
    job = get_job(job_id)
    if job is None:
        raise LookupError(f"Job을 찾을 수 없습니다: {job_id}")
    if job["status"] != "completed":
        raise ValueError(f"ZIP이 아직 준비되지 않았습니다. 현재 상태: {job['status']}")
    if job.get("result") is None:
        raise RuntimeError("ZIP 결과가 없습니다.")
    return job["result"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_gt_mask(image_path: str, dataset_path: str) -> Image.Image | None:
    gt_path = build_gt_mask_path(image_path, dataset_path)
    if not gt_path.exists():
        return None
    try:
        return Image.open(str(gt_path)).convert("RGB")
    except Exception:
        return None


def _overlay_contour(heatmap: Image.Image, gt_mask_pil: Image.Image) -> Image.Image:
    """GT 마스크 윤곽선(빨간색)을 히트맵 위에 오버레이."""
    gt_arr = np.array(gt_mask_pil.convert("L"))
    binary = (gt_arr > 127).astype(np.uint8)
    if not binary.any():
        return heatmap
    arr = np.array(heatmap.convert("RGB"))
    mask_u8 = binary * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(arr, contours, -1, (255, 0, 0), 2)
    return Image.fromarray(arr)
