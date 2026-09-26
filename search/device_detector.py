"""Explicit CUDA device selection for staged dual-GPU execution."""
import gc
import os
from typing import Dict, Any, Tuple

try:
    import torch
except Exception:
    torch = None


def cuda_device(index: int = 0) -> str:
    if torch is None or not torch.cuda.is_available():
        return "cpu"
    count = torch.cuda.device_count()
    safe_index = max(0, min(int(index), count - 1))
    return f"cuda:{safe_index}"


def best_dtype(device: str):
    if torch is None:
        return None
    if not str(device).startswith("cuda"):
        return torch.float32
    index = int(str(device).split(":")[-1])
    major, _ = torch.cuda.get_device_capability(index)
    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def configured_devices() -> Dict[str, str]:
    """GPU 0 is reserved for embeddings/ingestion; GPU 1 for search models."""
    count = torch.cuda.device_count() if torch is not None and torch.cuda.is_available() else 0
    ingest_index = int(os.environ.get("VIDEO_RAG_INGEST_GPU", "0"))
    search_default = "1" if count >= 2 else "0"
    search_index = int(os.environ.get("VIDEO_RAG_SEARCH_GPU", search_default))
    return {
        "ingest": cuda_device(ingest_index),
        "search": cuda_device(search_index),
        # EasyOCR DataParallel requires cuda:0. It runs after Whisper is
        # released and before BGE/SigLIP are loaded.
        "ocr": os.environ.get(
            "VIDEO_RAG_OCR_DEVICE",
            cuda_device(ingest_index),
        ),
    }


def release_cuda(device: str = None):
    gc.collect()
    if torch is None or not torch.cuda.is_available():
        return
    try:
        if device and str(device).startswith("cuda"):
            with torch.cuda.device(int(str(device).split(":")[-1])):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        else:
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
    except Exception:
        torch.cuda.empty_cache()


def device_diagnostics() -> Dict[str, Any]:
    devices = configured_devices()
    result = {
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "gpu_count": int(torch.cuda.device_count()) if torch is not None and torch.cuda.is_available() else 0,
        "assignment": devices,
        "gpus": [],
    }
    if result["cuda_available"]:
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            result["gpus"].append({
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": torch.cuda.get_device_capability(index),
                "dtype": str(best_dtype(f"cuda:{index}")),
                "free_gb": round(free / (1024 ** 3), 2),
                "total_gb": round(total / (1024 ** 3), 2),
            })
    return result


def get_optimal_device() -> Tuple[str, Any]:
    """Backward-compatible helper: returns the ingestion device."""
    device = configured_devices()["ingest"]
    return device, best_dtype(device)
