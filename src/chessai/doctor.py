"""Machine-readiness checks for local development and cloud training."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

from chessai.engine import GameState
from chessai.engine.vocabulary import action_vocab_hash
from chessai.native import available as native_available
from chessai.native import selected_backend

MIN_TRAIN_CPU = 12
MIN_TRAIN_MEMORY_GB = 32.0
MIN_TRAIN_DISK_GB = 25.0
MIN_TRAIN_VRAM_GB = 24.0


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _physical_memory_gb() -> float | None:
    """Read physical RAM without adding a runtime dependency."""

    try:
        if os.name == "nt":
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            win_dll: Any = vars(ctypes)["WinDLL"]
            function = win_dll("kernel32", use_last_error=True).GlobalMemoryStatusEx
            function.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
            function.restype = ctypes.c_int
            if not function(ctypes.byref(status)):
                return None
            return round(int(status.ullTotalPhys) / 2**30, 2)

        sysconf = getattr(os, "sysconf", None)
        if sysconf is None:
            return None
        page_size = sysconf("SC_PAGE_SIZE")
        physical_pages = sysconf("SC_PHYS_PAGES")
        if not isinstance(page_size, int) or not isinstance(physical_pages, int):
            return None
        return round(page_size * physical_pages / 2**30, 2)
    except (OSError, ValueError):
        return None


def _torch_inference_probe(torch: Any) -> dict[str, Any]:
    """Execute a real tiny policy/value forward pass on the selected device."""

    from chessai.ai.features import INPUT_PLANES
    from chessai.ai.model import ModelConfig, PolicyValueModel

    use_cuda = bool(torch.cuda.is_available())
    device = torch.device("cuda:0" if use_cuda else "cpu")
    use_bf16 = use_cuda and bool(torch.cuda.is_bf16_supported())
    rng_devices = [0] if use_cuda else []

    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(0)
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        model = PolicyValueModel(ModelConfig.tiny()).eval().to(device)
        inputs = torch.zeros((1, INPUT_PLANES, 10, 9), dtype=torch.float32, device=device)
        if use_cuda:
            torch.cuda.synchronize(device)
        started = perf_counter()
        with torch.inference_mode():
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, values = model(inputs)
            else:
                logits, values = model(inputs)
        if use_cuda:
            torch.cuda.synchronize(device)
        elapsed_ms = (perf_counter() - started) * 1000

        finite = bool(torch.isfinite(logits).all().item() and torch.isfinite(values).all().item())
        probe: dict[str, Any] = {
            "ok": finite and tuple(logits.shape) == (1, 2086) and tuple(values.shape) == (1,),
            "device": str(device),
            "precision": "bf16-autocast" if use_bf16 else "fp32",
            "logits_shape": list(logits.shape),
            "value_shape": list(values.shape),
            "outputs_finite": finite,
            "elapsed_ms": round(elapsed_ms, 2),
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
        if use_cuda:
            probe["peak_allocated_mb"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 2)
        return probe


def _is_playable_training_ready(report: dict[str, Any], memory_gb: float | None) -> bool:
    torch_report = report["torch"]
    engine_report = report["engine"]
    return bool(
        torch_report.get("cuda_available")
        and torch_report.get("bf16_supported")
        and torch_report.get("required_device_arch") == "sm_120"
        and torch_report.get("device_arch_compatible")
        and torch_report.get("inference_probe", {}).get("ok")
        and (torch_report.get("vram_gb") or 0) >= MIN_TRAIN_VRAM_GB
        and report["disk_free_gb"] >= MIN_TRAIN_DISK_GB
        and (report["cpu_count"] or 0) >= MIN_TRAIN_CPU
        and memory_gb is not None
        and memory_gb >= MIN_TRAIN_MEMORY_GB
        and engine_report["native_backend"]
        and engine_report["selected_backend"] == "native"
    )


def collect_doctor_report(workspace: str | Path = ".") -> dict[str, Any]:
    memory_gb = _physical_memory_gb()
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory_gb": memory_gb,
        "disk_free_gb": round(shutil.disk_usage(Path(workspace).resolve()).free / 2**30, 2),
        "playable_training_requirements": {
            "cpu_count_min": MIN_TRAIN_CPU,
            "memory_gb_min": MIN_TRAIN_MEMORY_GB,
            "disk_free_gb_min": MIN_TRAIN_DISK_GB,
            "vram_gb_min": MIN_TRAIN_VRAM_GB,
        },
        "engine": {
            "initial_legal_moves": len(GameState.initial().legal_moves),
            "action_vocab_hash": action_vocab_hash(),
            "native_backend": native_available(),
            "selected_backend": selected_backend(),
        },
        "torch": {"installed": False},
        "recommendations": [],
    }
    try:
        import torch

        torch_report: dict[str, Any] = {
            "installed": True,
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            compute_capability = f"{properties.major}.{properties.minor}"
            target_arch = f"sm_{properties.major}{properties.minor}"
            get_arch_list: Any = getattr(torch.cuda, "get_arch_list", lambda: [])
            get_gencode_flags: Any = getattr(torch.cuda, "get_gencode_flags", lambda: "")
            compiled_arches = [str(arch) for arch in get_arch_list()]
            gencode_flags = str(get_gencode_flags())
            arch_compatible = target_arch in compiled_arches or (
                f"compute_{properties.major}{properties.minor}" in gencode_flags
            )
            torch_report.update(
                {
                    "device": properties.name,
                    "vram_gb": round(properties.total_memory / 2**30, 2),
                    "compute_capability": compute_capability,
                    "required_device_arch": target_arch,
                    "compiled_arches": compiled_arches,
                    "gencode_flags": gencode_flags,
                    "device_arch_compatible": arch_compatible,
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                }
            )
        try:
            torch_report["inference_probe"] = _torch_inference_probe(torch)
        except Exception as exc:  # diagnostic boundary: report, do not hide readiness failure
            torch_report["inference_probe"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        report["torch"] = torch_report
    except ImportError:
        report["recommendations"].append("Install the train extra to enable model checks.")

    if report["torch"].get("installed") and not report["torch"].get("cuda_available"):
        report["recommendations"].append(
            "Installed PyTorch is CPU-only; use a CUDA 12.8+ wheel on RTX 5090/Blackwell."
        )
    if report["torch"].get("cuda_available") and not report["torch"].get("bf16_supported"):
        report["recommendations"].append("The selected CUDA device/build does not support BF16.")
    if report["torch"].get("cuda_available") and not report["torch"].get("device_arch_compatible"):
        report["recommendations"].append(
            "PyTorch does not advertise the active GPU architecture; RTX 5090 requires sm_120."
        )
    if (
        report["torch"].get("cuda_available")
        and report["torch"].get("required_device_arch") != "sm_120"
    ):
        report["recommendations"].append(
            "Playable training is gated to the configured RTX 5090 (sm_120) host."
        )
    if not report["torch"].get("inference_probe", {}).get("ok"):
        report["recommendations"].append(
            "The policy/value inference probe failed; resolve it before self-play or training."
        )
    if (report["torch"].get("vram_gb") or 0) < MIN_TRAIN_VRAM_GB:
        report["recommendations"].append(
            f"Playable training expects at least {MIN_TRAIN_VRAM_GB:g} GB GPU memory."
        )
    if (report["cpu_count"] or 0) < MIN_TRAIN_CPU:
        report["recommendations"].append(
            f"Playable self-play expects at least {MIN_TRAIN_CPU} vCPU."
        )
    if memory_gb is None or memory_gb < MIN_TRAIN_MEMORY_GB:
        report["recommendations"].append(
            f"Playable training expects at least {MIN_TRAIN_MEMORY_GB:g} GB physical RAM."
        )
    if report["disk_free_gb"] < MIN_TRAIN_DISK_GB:
        report["recommendations"].append(
            f"Playable training needs at least {MIN_TRAIN_DISK_GB:g} GB free on its persistent volume."
        )
    if not native_available():
        report["recommendations"].append(
            "Python rules backend is active; build the C++20 backend before cloud-scale self-play."
        )
    elif selected_backend() != "native":
        report["recommendations"].append(
            "The native backend is built but not selected; set CHESSAI_RULES_BACKEND=native."
        )
    inference_ok = bool(report["torch"].get("inference_probe", {}).get("ok"))
    report["ready_for_local_smoke"] = bool(
        report["engine"]["initial_legal_moves"] == 44 and inference_ok
    )
    report["ready_for_playable_training"] = _is_playable_training_ready(report, memory_gb)
    return report
