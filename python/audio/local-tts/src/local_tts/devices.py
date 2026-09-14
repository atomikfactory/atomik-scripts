"""Compute device detection (CUDA / CPU) without hard-coding anything."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .errors import DeviceError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    device: str  # "cuda", "cuda:1", "cpu"
    name: str
    total_vram_gb: float | None = None

    @property
    def is_cuda(self) -> bool:
        return self.device.startswith("cuda")


def _import_torch():
    try:
        import torch  # noqa: WPS433 (runtime import on purpose)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DeviceError(
            "PyTorch is not installed, so no TTS model can run.",
            hints=[
                "Install the default engine: pip install -r requirements-chatterbox.txt",
                "See README.md -> Installation for the CUDA-enabled PyTorch command.",
            ],
        ) from exc
    return torch


def cuda_available() -> bool:
    try:
        torch = _import_torch()
    except DeviceError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive
        return False


def resolve_device(requested: str = "auto", *, cpu_supported: bool = True, model_label: str = "") -> DeviceInfo:
    """Turn ``auto``/``cuda``/``cpu`` into a concrete device, validating it.

    ``auto`` -> CUDA when available, otherwise CPU (with a warning).
    Raises :class:`DeviceError` for impossible requests.
    """
    torch = _import_torch()
    requested = (requested or "auto").strip().lower()
    has_cuda = bool(torch.cuda.is_available())

    if requested == "auto":
        if has_cuda:
            device = "cuda"
        else:
            if not cpu_supported:
                raise DeviceError(
                    f"No CUDA GPU detected and {model_label or 'this model'} does not support CPU inference.",
                    hints=cuda_hints(torch),
                )
            log.warning(
                "CUDA is not available - falling back to CPU. Generation will be much slower. "
                "Use --model kokoro or --model chatterbox-nano for practical CPU speed."
            )
            device = "cpu"
    elif requested == "cpu":
        if not cpu_supported:
            raise DeviceError(f"{model_label or 'This model'} does not support CPU inference.")
        device = "cpu"
    elif requested.startswith("cuda"):
        if not has_cuda:
            raise DeviceError("CUDA was requested (--device cuda) but PyTorch cannot see a CUDA GPU.", hints=cuda_hints(torch))
        device = requested
        if ":" in requested:
            idx = int(requested.split(":", 1)[1])
            if idx >= torch.cuda.device_count():
                raise DeviceError(f"GPU index {idx} does not exist (found {torch.cuda.device_count()} CUDA device(s)).")
    else:
        raise DeviceError(f"Unknown device '{requested}'. Use auto, cuda, cuda:N or cpu.")

    if device.startswith("cuda"):
        idx = torch.device(device).index or 0
        props = torch.cuda.get_device_properties(idx)
        return DeviceInfo(device=device, name=props.name, total_vram_gb=round(props.total_memory / 1024**3, 1))
    return DeviceInfo(device="cpu", name="CPU")


def cuda_hints(torch=None) -> list[str]:
    hints = [
        "Check that an NVIDIA driver is installed (run `nvidia-smi`).",
        "Make sure you installed the CUDA build of PyTorch, e.g. "
        "pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124",
        "Run with --device cpu to use the CPU instead (slow).",
    ]
    if torch is not None:
        try:
            built = torch.version.cuda
            hints.insert(0, f"Installed PyTorch {torch.__version__} was built for CUDA {built or 'none (CPU-only build)'}.")
        except Exception:  # pragma: no cover
            pass
    return hints


def free_cuda_memory() -> None:
    """Best-effort release of cached GPU memory (used between models / after OOM)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass
