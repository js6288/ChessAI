"""Versioned 117-plane feature encoding for Xiangqi policy/value models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from chessai.engine import Color, GameState, Move, Square
from chessai.engine.board import EMPTY, NO_CAPTURE_DRAW_PLIES
from chessai.engine.vocabulary import action_labels, encode_move

HISTORY_LENGTH = 8
PIECE_PLANES_PER_FRAME = 14
INPUT_PLANES = 117
PIECE_ORDER = "KABNRCPkabnrcp"
PIECE_TO_PLANE = {piece: index for index, piece in enumerate(PIECE_ORDER)}

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


def _canonical_board(board: tuple[str, ...], side_to_move: Color) -> tuple[str, ...]:
    if side_to_move is Color.RED:
        return board
    rotated = [EMPTY] * 90
    for index, piece in enumerate(board):
        target = Square.from_index(index).rotate().index
        rotated[target] = piece.swapcase() if piece != EMPTY else EMPTY
    return tuple(rotated)


def canonical_move(move: Move, side_to_move: Color) -> Move:
    return move.rotate() if side_to_move is Color.BLACK else move


def encode_state(state: GameState) -> FloatArray:
    """Encode a state as ``[117, 10, 9]`` from the mover's perspective."""

    features = np.zeros((INPUT_PLANES, 10, 9), dtype=np.float32)
    history = tuple(reversed(state.board_history[-HISTORY_LENGTH:]))
    for history_index, board in enumerate(history):
        canonical = _canonical_board(board, state.side_to_move)
        plane_offset = history_index * PIECE_PLANES_PER_FRAME
        for square_index, piece in enumerate(canonical):
            if piece == EMPTY:
                continue
            square = Square.from_index(square_index)
            features[plane_offset + PIECE_TO_PLANE[piece], square.rank, square.file] = 1.0

    if state.move_records:
        move = canonical_move(state.move_records[-1].move, state.side_to_move)
        features[112, move.from_square.rank, move.from_square.file] = 1.0
        features[113, move.to_square.rank, move.to_square.file] = 1.0

    repetitions = state.repetition_count()
    if repetitions >= 2:
        features[114, :, :] = 1.0
    if repetitions >= 3:
        features[115, :, :] = 1.0
    features[116, :, :] = min(state.halfmove_clock, NO_CAPTURE_DRAW_PLIES) / float(
        NO_CAPTURE_DRAW_PLIES
    )
    return features


def encode_batch(states: Sequence[GameState]) -> FloatArray:
    if not states:
        return np.empty((0, INPUT_PLANES, 10, 9), dtype=np.float32)
    return np.stack([encode_state(state) for state in states], axis=0)


def legal_action_mask(state: GameState) -> BoolArray:
    mask = np.zeros(len(action_labels()), dtype=np.bool_)
    canonical_black = state.side_to_move is Color.BLACK
    for move in state.legal_moves:
        mask[encode_move(move, canonical_black=canonical_black)] = True
    return mask
