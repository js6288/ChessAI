"""Human-readable Chinese move notation and PGN export helpers."""

from __future__ import annotations

from chessai.engine.board import Color, GameState
from chessai.engine.move import Move, Square

RED_NUMERALS = "〇一二三四五六七八九"
RED_GLYPHS = {"K": "帅", "A": "仕", "B": "相", "N": "马", "R": "车", "C": "炮", "P": "兵"}
BLACK_GLYPHS = {"K": "将", "A": "士", "B": "象", "N": "马", "R": "车", "C": "炮", "P": "卒"}


def _file_number(file: int, color: Color) -> int:
    return 9 - file if color is Color.RED else file + 1


def _number(value: int, color: Color) -> str:
    return RED_NUMERALS[value] if color is Color.RED else str(value)


def move_to_chinese(state: GameState, move: Move) -> str:
    if move not in state.legal_moves:
        raise ValueError(f"cannot notate illegal move {move}")
    piece = state.piece_at(move.from_square)
    assert piece is not None
    color = state.side_to_move
    kind = piece.upper()
    glyph = (RED_GLYPHS if color is Color.RED else BLACK_GLYPHS)[kind]

    same_file = [
        square
        for index, other in enumerate(state.board)
        if other == piece and (square := Square.from_index(index)).file == move.from_square.file
    ]
    if len(same_file) > 1:
        ordered = sorted(same_file, key=lambda square: square.rank, reverse=color is Color.RED)
        position = ordered.index(move.from_square)
        if position == 0:
            prefix = "前"
        elif position == len(ordered) - 1:
            prefix = "后"
        elif len(ordered) % 2 == 1 and position == len(ordered) // 2:
            prefix = "中"
        else:
            prefix = _number(position + 1, color)
        first = prefix + glyph
    else:
        first = glyph + _number(_file_number(move.from_square.file, color), color)

    delta_rank = move.to_square.rank - move.from_square.rank
    if delta_rank == 0:
        action = "平"
        destination = _file_number(move.to_square.file, color)
    else:
        advancing = delta_rank > 0 if color is Color.RED else delta_rank < 0
        action = "进" if advancing else "退"
        destination = (
            _file_number(move.to_square.file, color) if kind in {"N", "B", "A"} else abs(delta_rank)
        )
    return first + action + _number(destination, color)


def export_pgn(
    initial_fen: str,
    moves: list[Move],
    *,
    result: str = "*",
    event: str = "ChessAI local game",
) -> str:
    state = GameState.from_fen(initial_fen)
    notations: list[str] = []
    for move in moves:
        notations.append(move_to_chinese(state, move))
        state = state.apply(move)
    tags = [
        '[Game "Chinese Chess"]',
        f'[Event "{event}"]',
        f'[Result "{result}"]',
    ]
    if initial_fen != GameState.initial().to_fen():
        tags.append(f'[FEN "{initial_fen}"]')
    body: list[str] = []
    for index in range(0, len(notations), 2):
        pair = " ".join(notations[index : index + 2])
        body.append(f"{index // 2 + 1}. {pair}")
    return "\n".join([*tags, "", *body, result, ""])
