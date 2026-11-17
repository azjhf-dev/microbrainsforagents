"""Atomic file replacement with Windows sharing-violation retry.

This is a mitigation, not a fix, for concurrent reader handle contention on
Windows. When all retry attempts are exhausted, the original PermissionError is
re-raised, not swallowed.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable


def _is_sharing_violation(error: BaseException) -> bool:
    """Return True only for Windows sharing/access violations (winerror 5 or 32)."""
    if os.name != "nt":
        return False
    if not isinstance(error, PermissionError):
        return False
    winerr = getattr(error, "winerror", None)
    return winerr in (5, 32)


def atomic_replace(
    src: Path | str,
    dst: Path | str,
    *,
    attempts: int = 8,
    delay_s: float = 0.015,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Replace *dst* with *src* via ``os.replace``, retrying Windows sharing violations."""
    if attempts <= 0:
        raise ValueError(f"attempts must be >= 1, got {attempts}")
    last_exc: PermissionError | None = None
    for i in range(1, attempts + 1):
        try:
            os.replace(src, dst)
            return
        except Exception as exc:
            if not _is_sharing_violation(exc):
                raise
            last_exc = exc  # type: ignore[assignment]
            if i < attempts:
                sleep(round(delay_s * i, 6))
    assert last_exc is not None
    raise last_exc


def atomic_write(path: Path | str, data: Any, indent: int = 2) -> None:
    """Atomically write ``data`` as JSON to ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    atomic_replace(str(tmp), str(target))


def atomic_read(path: Path | str, default: Any = None) -> Any:
    """Atomically read and parse JSON from ``path``."""
    target = Path(path)
    if not target.exists():
        return default
    try:
        with open(target, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


__all__ = ["atomic_write", "atomic_read", "atomic_replace", "_is_sharing_violation"]
