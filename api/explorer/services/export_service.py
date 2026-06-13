"""
api/explorer/services/export_service.py

EfficientAD 모델 비동기 변환 서비스.
지원 포맷: ONNX / OpenVINO / TensorRT(조건부)

PatchCore 미지원:
  TODO: anomalib Issue #2303 — ONNX export 후 모든 예측 이상 판정 버그 미해결.
  anomalib to_openvino() 도 내부적으로 ONNX 체이닝 → 동일 버그 영향.
  버그 검증 후 지원 추가 예정.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from api.explorer.jobs import create_job, set_completed, set_failed, set_running
from utils.storage import load_history


def _make_export_wrapper(inner_model):
    """EfficientAdModel을 감싸 plain 2-tuple (pred_score, anomaly_map) 반환하는 wrapper 생성.

    InferenceBatch는 (pred_score, pred_label=None, anomaly_map, pred_mask=None)
    4-field NamedTuple이다. torch.onnx.export 및 ov.convert_model의 None 처리
    방식이 버전마다 다를 수 있으므로, plain tuple로 명시적으로 변환한다.
    """
    import torch

    class _Wrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.inner = model

        def forward(self, x: torch.Tensor):
            out = self.inner(x)
            return out.pred_score, out.anomaly_map

    return _Wrapper(inner_model)


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
# Public API
# ---------------------------------------------------------------------------

async def start_export(exp_id: str, fmt: str) -> str:
    exp = _get_experiment(exp_id)

    # TODO: anomalib Issue #2303 해결 후 PatchCore 지원 추가 예정
    if exp.get("model_type") != "efficientad":
        raise ValueError(
            "EfficientAD 모델만 내보내기를 지원합니다. "
            "(PatchCore: anomalib Issue #2303 미해결)"
        )

    if exp.get("status") != "completed":
        raise ValueError("완료된 실험만 내보내기를 지원합니다.")

    if fmt not in ("onnx", "openvino", "trt"):
        raise ValueError(f"지원하지 않는 포맷: {fmt}")

    model_dir = Path(exp["model_path"])
    _check_output_exists(fmt, model_dir)

    job_id = create_job("export")
    asyncio.create_task(_run_export(job_id, exp, fmt, model_dir))
    return job_id


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------

def _check_output_exists(fmt: str, model_dir: Path) -> None:
    """동일 파일 존재 시 FileExistsError 발생."""
    targets = {
        "onnx":     model_dir / "model.onnx",
        "openvino": model_dir / "model.xml",
        "trt":      model_dir / "model.trt",
    }
    path = targets[fmt]
    if path.exists():
        raise FileExistsError(f"이미 내보내기 파일이 존재합니다: {path.name}")


async def _run_export(job_id: str, exp: dict, fmt: str, model_dir: Path) -> None:
    set_running(job_id)
    try:
        result = await asyncio.to_thread(_export_sync, exp, fmt, model_dir)
        set_completed(job_id, result)
    except Exception as e:
        set_failed(job_id, str(e))


# ---------------------------------------------------------------------------
# Sync worker (runs in thread)
# ---------------------------------------------------------------------------

def _export_sync(exp: dict, fmt: str, model_dir: Path) -> dict:
    """모델 로드 → 변환 → 저장. Thread에서 실행."""
    import torch
    from utils.model_factory import load_model_for_inference

    exp_id    = exp["experiment_id"]
    image_size = exp.get("image_size", 256)
    model_config = {
        "model_type": exp["model_type"],
        "params":     exp.get("model_params", {}),
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model       = load_model_for_inference(exp_id, exp["model_path"], model_config, device)
    torch_model = getattr(model, "model", model)        # EfficientAdModel (nn.Module)
    dummy_input = torch.zeros(1, 3, image_size, image_size, device=device)

    if fmt == "onnx":
        saved_path = _export_onnx(torch_model, dummy_input, model_dir)
    elif fmt == "openvino":
        saved_path = _export_openvino(torch_model, dummy_input, model_dir)
    else:  # trt
        saved_path = _export_trt(torch_model, dummy_input, model_dir)

    return {"saved_path": str(saved_path), "format": fmt}


# ---------------------------------------------------------------------------
# Format-specific exporters
# ---------------------------------------------------------------------------

def _export_onnx(torch_model, dummy_input, model_dir: Path) -> Path:
    import torch

    out_path = model_dir / "model.onnx"
    wrapper  = _make_export_wrapper(torch_model)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(out_path),
        opset_version=14,
        input_names=["image"],
        output_names=["pred_score", "anomaly_map"],
    )

    # onnxruntime 설치 시 dummy inference 검증, 없으면 skip
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        sess.run(None, {"image": dummy_input.cpu().numpy()})
    except ImportError:
        pass

    return out_path


def _export_openvino(torch_model, dummy_input, model_dir: Path) -> Path:
    import openvino as ov

    out_xml = model_dir / "model.xml"
    # OV convert_model은 CPU 모델 + CPU 텐서 필요
    wrapper   = _make_export_wrapper(torch_model.cpu())
    dummy_cpu = dummy_input.cpu()

    ov_model = ov.convert_model(wrapper, example_input=dummy_cpu)
    ov.save_model(ov_model, str(out_xml))

    return out_xml


def _export_trt(torch_model, dummy_input, model_dir: Path) -> Path:
    import io
    import torch
    import tensorrt as trt

    # ONNX 버퍼로 먼저 export (파일 저장 없이 메모리에서 TRT로 직접 변환)
    buf     = io.BytesIO()
    wrapper = _make_export_wrapper(torch_model)
    torch.onnx.export(
        wrapper,
        dummy_input,
        buf,
        opset_version=14,
        input_names=["image"],
        output_names=["pred_score", "anomaly_map"],
    )
    onnx_bytes = buf.getvalue()

    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_bytes):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"TRT ONNX 파싱 실패: {'; '.join(errors)}")

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    engine_bytes = builder.build_serialized_network(network, cfg)
    if engine_bytes is None:
        raise RuntimeError("TRT 엔진 빌드 실패")

    trt_path = model_dir / "model.trt"
    trt_path.write_bytes(bytes(engine_bytes))
    return trt_path
