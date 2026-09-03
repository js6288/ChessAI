"""Streaming supervised examples reconstructed from normalized game records."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from chessai.ai.features import encode_state
from chessai.engine import Color, GameState, Move
from chessai.engine.vocabulary import encode_move

try:
    import torch
    from torch.utils.data import IterableDataset, get_worker_info
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("install the 'train' extra to stream training examples") from exc


def result_value(result: str, perspective: Color) -> float:
    if result == "1/2-1/2":
        return 0.0
    red_value = 1.0 if result == "1-0" else -1.0
    return red_value if perspective is Color.RED else -red_value


PHASE_NAMES = ("opening", "middlegame", "endgame")


def position_phase(state: GameState) -> int:
    """Return a documented, deterministic phase bucket for reporting only."""

    piece_count = sum(piece != "." for piece in state.board)
    if state.ply < 40 and piece_count >= 24:
        return 0
    if state.ply >= 120 or piece_count <= 14:
        return 2
    return 1


def iter_record_examples(record: dict[str, Any]) -> Iterator[tuple[np.ndarray, int, float]]:
    state = GameState.from_fen(record["initial_fen"])
    result = record["result"]
    for move_text in record["moves"]:
        move = Move.from_iccs(move_text)
        if move not in state.ordinary_legal_moves:
            raise ValueError(
                f"prepared record contains illegal move {move} in game {record['game_id']}"
            )
        action = encode_move(move, canonical_black=state.side_to_move is Color.BLACK)
        yield encode_state(state), action, result_value(result, state.side_to_move)
        # Imported historical games may continue through a position that the
        # configured repetition adjudicator would already call terminal. The
        # data gate validated each move against ordinary Xiangqi legality, so
        # reconstruct the same contract here instead of truncating the record.
        state = state.apply(move, validate=False)


def iter_record_evaluation_examples(
    record: dict[str, Any],
) -> Iterator[tuple[np.ndarray, int, float, np.ndarray, int]]:
    state = GameState.from_fen(record["initial_fen"])
    result = record["result"]
    for move_text in record["moves"]:
        move = Move.from_iccs(move_text)
        legal_moves = state.ordinary_legal_moves
        if move not in legal_moves:
            raise ValueError(
                f"prepared record contains illegal move {move} in game {record['game_id']}"
            )
        canonical_black = state.side_to_move is Color.BLACK
        legal_mask = np.zeros(2086, dtype=np.bool_)
        legal_mask[
            [encode_move(legal, canonical_black=canonical_black) for legal in legal_moves]
        ] = True
        yield (
            encode_state(state),
            encode_move(move, canonical_black=canonical_black),
            result_value(result, state.side_to_move),
            legal_mask,
            position_phase(state),
        )
        state = state.apply(move, validate=False)


class JsonlPositionDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, path: str | Path, *, max_games: int | None = None) -> None:
        super().__init__()
        self.path = Path(path)
        self.max_games = max_games

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        with self.path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if self.max_games is not None and line_index >= self.max_games:
                    break
                if line_index % worker_count != worker_id or not line.strip():
                    continue
                record = json.loads(line)
                for features, action, value in iter_record_examples(record):
                    yield (
                        torch.from_numpy(features),
                        torch.tensor(action, dtype=torch.long),
                        torch.tensor(value, dtype=torch.float32),
                    )


class JsonlEvaluationDataset(
    IterableDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(self, path: str | Path, *, max_games: int | None = None) -> None:
        super().__init__()
        self.path = Path(path)
        self.max_games = max_games

    def __iter__(
        self,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        with self.path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if self.max_games is not None and line_index >= self.max_games:
                    break
                if line_index % worker_count != worker_id or not line.strip():
                    continue
                record = json.loads(line)
                for features, action, value, legal_mask, phase in iter_record_evaluation_examples(
                    record
                ):
                    yield (
                        torch.from_numpy(features),
                        torch.tensor(action, dtype=torch.long),
                        torch.tensor(value, dtype=torch.float32),
                        torch.from_numpy(legal_mask),
                        torch.tensor(phase, dtype=torch.long),
                    )
