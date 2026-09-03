"""Optional C++20 accelerator with an explicit Python fallback."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from chessai.compat import RULE_VERSION

if TYPE_CHECKING:
    from chessai.engine.board import Color

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


def legal_move_codes(fen: str) -> tuple[int, ...] | None:
    """Return compact ``from * 90 + to`` codes, or ``None`` in reference mode."""

    if selected_backend() == "reference":
        return None
    if not hasattr(_native, "legal_move_codes"):
        # A source update can precede the required native rebuild. Keep the
        # old extension usable, with a small conversion cost, until CMake runs.
        codes = []
        for text in _native.legal_moves(fen):
            from_index = int(text[1]) * 9 + (ord(text[0]) - ord("a"))
            to_index = int(text[3]) * 9 + (ord(text[2]) - ord("a"))
            codes.append(from_index * 90 + to_index)
        return tuple(codes)
    return tuple(int(code) for code in _native.legal_move_codes(fen))


def is_in_check(fen: str, color: Color | None = None) -> bool | None:
    """Use the native attack detector when selected, otherwise request fallback."""

    if selected_backend() == "reference":
        return None
    return bool(_native.is_in_check(fen, color.value if color is not None else None))


def is_in_check_board(board: tuple[str, ...], color: Color) -> bool | None:
    """Check a compact 90-square board without formatting/parsing FEN."""

    if selected_backend() == "reference" or not hasattr(_native, "is_in_check_board"):
        return None
    return bool(_native.is_in_check_board("".join(board), color.value))
