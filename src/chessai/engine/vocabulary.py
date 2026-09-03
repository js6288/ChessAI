"""Frozen 2,086-action ICCS move vocabulary used by every model."""

from __future__ import annotations

import hashlib
from functools import lru_cache

from chessai.compat import ACTION_VOCAB_VERSION
from chessai.engine.move import Move, Square


def _append_unique(labels: list[str], seen: set[str], move: Move) -> None:
    label = str(move)
    if label not in seen:
        labels.append(label)
        seen.add(label)


@lru_cache(maxsize=1)
def action_labels() -> tuple[str, ...]:
    labels: list[str] = []
    seen: set[str] = set()
    horse_offsets = ((-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2))

    for file in range(9):
        for rank in range(10):
            origin = Square(file, rank)
            for target_file in range(9):
                if target_file != file:
                    _append_unique(labels, seen, Move(origin, Square(target_file, rank)))
            for target_rank in range(10):
                if target_rank != rank:
                    _append_unique(labels, seen, Move(origin, Square(file, target_rank)))
            for df, dr in horse_offsets:
                target_file, target_rank = file + df, rank + dr
                if 0 <= target_file < 9 and 0 <= target_rank < 10:
                    _append_unique(labels, seen, Move(origin, Square(target_file, target_rank)))

    advisor_edges = (
        "d7e8",
        "e8d7",
        "e8f9",
        "f9e8",
        "d0e1",
        "e1d0",
        "e1f2",
        "f2e1",
        "d2e1",
        "e1d2",
        "e1f0",
        "f0e1",
        "d9e8",
        "e8d9",
        "e8f7",
        "f7e8",
    )
    elephant_edges = (
        "a2c4",
        "c4a2",
        "c0e2",
        "e2c0",
        "e2g4",
        "g4e2",
        "g0i2",
        "i2g0",
        "a7c9",
        "c9a7",
        "c5e7",
        "e7c5",
        "e7g9",
        "g9e7",
        "g5i7",
        "i7g5",
        "a2c0",
        "c0a2",
        "c4e2",
        "e2c4",
        "e2g0",
        "g0e2",
        "g4i2",
        "i2g4",
        "a7c5",
        "c5a7",
        "c9e7",
        "e7c9",
        "e7g5",
        "g5e7",
        "g9i7",
        "i7g9",
    )
    for label in (*advisor_edges, *elephant_edges):
        _append_unique(labels, seen, Move.from_iccs(label))

    if len(labels) != 2086:
        raise RuntimeError(f"{ACTION_VOCAB_VERSION} generated {len(labels)} labels, expected 2086")
    return tuple(labels)


@lru_cache(maxsize=1)
def label_to_index() -> dict[str, int]:
    return {label: index for index, label in enumerate(action_labels())}


def action_vocab_hash() -> str:
    return hashlib.sha256(action_vocab_payload()).hexdigest()


def action_vocab_payload() -> bytes:
    """Return the canonical LF-only, version-prefixed export payload."""

    payload = f"{ACTION_VOCAB_VERSION}\n" + "\n".join(action_labels()) + "\n"
    return payload.encode("ascii")


def encode_move(move: Move, *, canonical_black: bool = False) -> int:
    canonical = move.rotate() if canonical_black else move
    try:
        return label_to_index()[str(canonical)]
    except KeyError as exc:
        raise ValueError(f"move is outside the fixed action vocabulary: {move}") from exc


def decode_move(index: int, *, canonical_black: bool = False) -> Move:
    labels = action_labels()
    if not 0 <= index < len(labels):
        raise ValueError(f"action index outside vocabulary: {index}")
    move = Move.from_iccs(labels[index])
    return move.rotate() if canonical_black else move
