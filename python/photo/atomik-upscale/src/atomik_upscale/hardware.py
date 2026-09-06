"""Hardware probing and tuning decisions.

Everything the rest of the package needs to know about the machine it is
running on is decided exactly once, here, and cached on disk.
"""
from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, asdict

import torch

from .paths import CACHE_DIR

MB = 1024 * 1024


@dataclass(frozen=True)
class Hardware:
    device: str
    gpu_name: str
    vram_total_mb: int
    compute_cap: tuple[int, int] | None
    cuda_version: str | None
    driver_ok: bool
    cpu: str
    cpu_threads: int
    ram_gb: int

    # --- derived tuning knobs ------------------------------------------
    @property
    def supports_fp16(self) -> bool:
        # Tensor cores from Volta (7.0) onward make fp16 a strict win.
        return self.compute_cap is not None and self.compute_cap >= (7, 0)

    @property
    def supports_bf16(self) -> bool:
        return self.compute_cap is not None and self.compute_cap >= (8, 0)

    @property
    def supports_tf32(self) -> bool:
        return self.compute_cap is not None and self.compute_cap >= (8, 0)

    @property
    def dtype(self) -> torch.dtype:
        return torch.float16 if self.supports_fp16 else torch.float32

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"

    def vram_budget_mb(self) -> int:
        """VRAM we are willing to hand to tile activations.

        Leaves headroom for the model weights, the display driver and
        whatever else the desktop is already holding.
        """
        if not self.is_cuda:
            return 2048
        free, _total = torch.cuda.mem_get_info()
        return int(free / MB * 0.80)

    def summary(self) -> str:
        if not self.is_cuda:
            return f"CPU only — {self.cpu} ({self.cpu_threads} threads), {self.ram_gb} GB RAM"
        cc = ".".join(map(str, self.compute_cap or ()))
        return (
            f"{self.gpu_name} — {self.vram_total_mb / 1024:.0f} GB VRAM, "
            f"compute {cc}, CUDA {self.cuda_version}\n"
            f"{self.cpu} ({self.cpu_threads} threads), {self.ram_gb} GB RAM"
        )


def _cpu_name() -> str:
    if os.name == "nt":
        # PROCESSOR_IDENTIFIER is the family/model string, not the marketing
        # name; the registry holds the one a human recognises.
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            with key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if name:
                return " ".join(str(name).split())
        except Exception:
            pass
        return " ".join(os.environ.get("PROCESSOR_IDENTIFIER", "").split()) or "unknown CPU"
    return " ".join((platform.processor() or "").split()) or "unknown CPU"


def _ram_gb() -> int:
    try:
        if os.name == "nt":
            import ctypes

            class MemStatus(ctypes.Structure):
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

            st = MemStatus()
            st.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return round(st.ullTotalPhys / 1024**3)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3)
    except Exception:
        return 0


_CACHED: Hardware | None = None


def detect(force: bool = False) -> Hardware:
    global _CACHED
    if _CACHED is not None and not force:
        return _CACHED

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        hw = Hardware(
            device="cuda",
            gpu_name=props.name,
            vram_total_mb=int(props.total_memory / MB),
            compute_cap=(props.major, props.minor),
            cuda_version=torch.version.cuda,
            driver_ok=True,
            cpu=_cpu_name(),
            cpu_threads=os.cpu_count() or 1,
            ram_gb=_ram_gb(),
        )
    else:
        hw = Hardware(
            device="cpu",
            gpu_name="none",
            vram_total_mb=0,
            compute_cap=None,
            cuda_version=torch.version.cuda,
            driver_ok=False,
            cpu=_cpu_name(),
            cpu_threads=os.cpu_count() or 1,
            ram_gb=_ram_gb(),
        )
    _CACHED = hw
    return hw


def apply_global_tuning(hw: Hardware) -> None:
    """Process-wide switches that only ever need setting once."""
    torch.set_grad_enabled(False)
    if hw.is_cuda:
        # Uniform tile shapes mean cuDNN's autotuner pays for itself immediately.
        torch.backends.cudnn.benchmark = True
        if hw.supports_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    # Leave a couple of cores for the I/O and encode threads.
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 4))


# --- tile-size memo ----------------------------------------------------------
_TILE_MEMO = CACHE_DIR / "tile_sizes.json"


def _memo_key(model_id: str, hw: Hardware) -> str:
    return f"{model_id}|{hw.gpu_name}|{hw.vram_total_mb}"


def remembered_profile(model_id: str, hw: Hardware) -> dict[int, float] | None:
    """Cached tile-cost curve: {tile side -> seconds for one tile}."""
    try:
        data = json.loads(_TILE_MEMO.read_text("utf-8"))
        entry = data.get(_memo_key(model_id, hw))
        if not entry:
            return None
        return {int(k): float(v) for k, v in entry.items()}
    except Exception:
        return None


def remember_profile(model_id: str, hw: Hardware,
                     profile: dict[int, float]) -> None:
    try:
        data = {}
        if _TILE_MEMO.exists():
            data = json.loads(_TILE_MEMO.read_text("utf-8"))
        data[_memo_key(model_id, hw)] = {str(k): round(v, 6)
                                         for k, v in profile.items()}
        _TILE_MEMO.write_text(json.dumps(data, indent=2), "utf-8")
    except Exception:
        pass


def hardware_report(hw: Hardware) -> dict:
    d = asdict(hw)
    d["supports_fp16"] = hw.supports_fp16
    d["supports_bf16"] = hw.supports_bf16
    d["supports_tf32"] = hw.supports_tf32
    d["torch"] = torch.__version__
    if hw.is_cuda:
        free, total = torch.cuda.mem_get_info()
        d["vram_free_mb"] = int(free / MB)
        d["vram_total_mb"] = int(total / MB)
    return d
