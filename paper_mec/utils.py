from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - fallback is intentionally broad
    _tqdm = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def normalize_advantage(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / (values.std(unbiased=False) + 1.0e-8)


def format_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0.0:
        return "?"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _enable_fast_cuda_math() -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True


def _format_compute_capability(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


def _blackwell_upgrade_message(device: torch.device, exc: Exception) -> str:
    name = torch.cuda.get_device_name(device)
    sm = _format_compute_capability(device)
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        f"CUDA runtime test failed on {name} ({sm}). This usually means the installed PyTorch binary was not built for this GPU architecture. "
        f"For NVIDIA Blackwell GPUs, PyTorch's official 2.7 release added Blackwell support with CUDA 12.8 wheels, and the current Start Locally page requires Python 3.10+ for latest stable. "
        f"Your current interpreter is Python {pyver}. Create a Python 3.10+ environment and install a CUDA 12.8+ PyTorch wheel from the official index, for example: "
        f"pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128. Original error: {exc}"
    )


def validate_torch_device(device: torch.device, requested: str | None) -> tuple[torch.device, str]:
    if device.type != "cuda":
        return device, "cpu"
    try:
        torch.empty(1, device=device)
        torch.cuda.synchronize(device)
    except Exception as exc:
        message = _blackwell_upgrade_message(device, exc)
        if (requested or "auto").strip().lower() == "auto":
            print(f"[Config] {message} Falling back to CPU.", flush=True)
            return torch.device("cpu"), "cpu (CUDA incompatible, fell back automatically)"
        raise RuntimeError(message) from exc
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return torch.device(f"cuda:{idx}"), f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"


def resolve_torch_device(requested: str | None) -> tuple[torch.device, str]:
    target = (requested or "auto").strip().lower()
    if target == "auto":
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            _enable_fast_cuda_math()
            return validate_torch_device(torch.device(f"cuda:{idx}"), requested)
        return torch.device("cpu"), "cpu"
    if target.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA device. Use --device cpu or install a CUDA-enabled PyTorch build.")
        _enable_fast_cuda_math()
        return validate_torch_device(torch.device(target), requested)
    if target == "cpu":
        return torch.device("cpu"), "cpu"
    return torch.device(target), target


class _NullProgressBar:
    def update(self, n: int = 1) -> None:
        return None

    def set_postfix(self, ordered_dict: dict[str, Any] | None = None, refresh: bool = True, **kwargs: Any) -> None:
        return None

    def write(self, message: str) -> None:
        if message:
            print(message, flush=True)

    def close(self) -> None:
        return None


class _TextProgressBar:
    def __init__(self, total: int | None, desc: str, refresh_interval_s: float = 5.0) -> None:
        self.total = total
        self.desc = desc
        self.refresh_interval_s = refresh_interval_s
        self.current = 0
        self.postfix: dict[str, Any] = {}
        self.last_emit = 0.0
        self.start_time = time.time()
        self.closed = False

    def _format_value(self, value: Any) -> str:
        if isinstance(value, float):
            if not np.isfinite(value):
                return "nan"
            if abs(value) >= 1000.0 or (0.0 < abs(value) < 1.0e-3):
                return f"{value:.3e}"
            return f"{value:.4f}"
        return str(value)

    def _emit(self, force: bool = False) -> None:
        if self.closed:
            return
        now = time.time()
        if not force and (now - self.last_emit) < self.refresh_interval_s:
            return
        elapsed = max(now - self.start_time, 0.0)
        if self.total is None or self.total <= 0:
            progress = f"{self.current}"
            eta = None
        else:
            progress = f"{self.current}/{self.total} ({100.0 * self.current / self.total:5.1f}%)"
            eta = None if self.current <= 0 else elapsed * max(self.total - self.current, 0) / self.current
        time_bits = [f"elapsed={format_duration(elapsed)}"]
        if eta is not None:
            time_bits.append(f"eta={format_duration(eta)}")
        extras = ""
        if self.postfix:
            extras = " | " + ", ".join(f"{key}={self._format_value(value)}" for key, value in self.postfix.items())
        print(f"[{self.desc}] {progress} | {', '.join(time_bits)}{extras}", flush=True)
        self.last_emit = now

    def update(self, n: int = 1) -> None:
        self.current += n
        should_force = self.total is not None and self.current >= self.total
        self._emit(force=should_force)

    def set_postfix(self, ordered_dict: dict[str, Any] | None = None, refresh: bool = True, **kwargs: Any) -> None:
        merged: dict[str, Any] = {}
        if ordered_dict:
            merged.update(ordered_dict)
        if kwargs:
            merged.update(kwargs)
        self.postfix = merged
        if refresh:
            self._emit(force=True)

    def write(self, message: str) -> None:
        if message:
            print(message, flush=True)

    def close(self) -> None:
        if not self.closed:
            self._emit(force=True)
            self.closed = True


def create_progress(
    total: int | None,
    desc: str,
    *,
    disable: bool = False,
    leave: bool = True,
    refresh_interval_s: float = 5.0,
) -> Any:
    if disable:
        return _NullProgressBar()
    if _tqdm is not None:
        return _tqdm(
            total=total,
            desc=desc,
            leave=leave,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
    return _TextProgressBar(total=total, desc=desc, refresh_interval_s=refresh_interval_s)
