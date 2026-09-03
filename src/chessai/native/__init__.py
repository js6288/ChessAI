"""Optional C++20 accelerator with an explicit Python fallback."""

from __future__ import annotations

import os
from typing import Any

from chessai.compat import RULE_VERSION

try:
    import _chessai_native as _native  # type: ignore[import-not-found]
except ImportError:
    _native = None


def available() -> bool:
    return _native is not None


def module() -> Any:
    if _native is None:
        raise RuntimeError(
            "native Xiangqi backend is not installed; using Python reference backend"
        )
    return _native


def preference() -> str:
    """Return ``auto``, ``native``, or ``reference`` from the runtime contract."""

    value = os.getenv("CHESSAI_RULES_BACKEND", "auto").strip().lower()
    if value not in {"auto", "native", "reference"}:
        raise ValueError("CHESSAI_RULES_BACKEND must be auto, native, or reference")
    return value


def selected_backend() -> str:
    requested = preference()
    if requested == "reference":
        return "reference"
    if available():
        if getattr(_native, "RULE_VERSION", None) != RULE_VERSION:
            raise RuntimeError(
                "native rules version does not match Python rules version: "
                f"{getattr(_native, 'RULE_VERSION', None)!r} != {RULE_VERSION!r}"
            )
        return "native"
    if requested == "native":
        raise RuntimeError("native Xiangqi backend was required but is not installed")
    return "reference"


def legal_moves(fen: str) -> tuple[str, ...] | None:
    """Return native legal moves, or ``None`` when reference mode is selected."""

    if selected_backend() == "reference":
        return None
    return tuple(_native.legal_moves(fen))
