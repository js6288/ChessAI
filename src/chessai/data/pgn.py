"""Chinese Xiangqi PGN parsing and notation-to-ICCS conversion."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from chessai.engine import Color, GameState, Move
from chessai.engine.board import INITIAL_FEN

TAG_RE = re.compile(r'^\[([A-Za-z][A-Za-z0-9_]*)\s+"(.*)"\]\s*$')
PIECE_CHARS = "帥帅將将仕士相象馬马傌車车俥炮砲兵卒"
NUMBER_CHARS = "一二三四五六七八九123456789"
ACTION_CHARS = "進进退平+-."
MOVE_TOKEN_RE = re.compile(
    rf"(?:[前中後后][{PIECE_CHARS}][{ACTION_CHARS}][{NUMBER_CHARS}]|"
    rf"[{PIECE_CHARS}][{NUMBER_CHARS}][{ACTION_CHARS}][{NUMBER_CHARS}]|"
    r"[a-i][0-9][a-i][0-9])",
    re.IGNORECASE,
)
RESULTS = {"1-0", "0-1", "1/2-1/2"}
NUMBERS = {
    **{character: index for index, character in enumerate("一二三四五六七八九", 1)},
    **{str(index): index for index in range(1, 10)},
}
PIECE_KIND = {
    "帥": "K",
    "帅": "K",
    "將": "K",
    "将": "K",
    "仕": "A",
    "士": "A",
    "相": "B",
    "象": "B",
    "馬": "N",
    "马": "N",
    "傌": "N",
    "車": "R",
    "车": "R",
    "俥": "R",
    "炮": "C",
    "砲": "C",
    "兵": "P",
    "卒": "P",
}


class PgnError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ParsedPgn:
    tags: dict[str, str]
    initial_fen: str
    result: str
    notations: tuple[str, ...]
    moves: tuple[Move, ...]
    final_state: GameState


def decode_pgn_bytes(payload: bytes) -> tuple[str, str]:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "big5", "gb18030"):
        try:
            text = payload.decode(encoding, errors="strict")
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
            continue
        if '[Game "Chinese Chess"]' in text or "[FEN " in text or "[Event " in text:
            return text, encoding
    raise PgnError("unable to decode PGN as UTF-8, Big5, or GB18030: " + " | ".join(errors))


def _strip_variations(text: str) -> str:
    previous = None
    cleaned = text
    while cleaned != previous:
        previous = cleaned
        cleaned = re.sub(r"\([^()]*\)", " ", cleaned)
    cleaned = re.sub(r"\{[^{}]*\}", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r";[^\n\r]*", " ", cleaned)
    return cleaned


def _file_from_notation(number: int, color: Color) -> int:
    return 9 - number if color is Color.RED else number - 1


def _direction(move: Move, color: Color) -> str:
    delta = move.to_square.rank - move.from_square.rank
    if delta == 0:
        return "平"
    advancing = delta > 0 if color is Color.RED else delta < 0
    return "进" if advancing else "退"


def _normalized_action(character: str) -> str:
    return {"進": "进", "+": "进", "-": "退", ".": "平"}.get(character, character)


def notation_to_move(state: GameState, notation: str) -> Move:
    token = unicodedata.normalize("NFKC", notation.strip())
    # Import replay validates ordinary move legality independently of the
    # runtime adjudication clock. Historical records may continue through a
    # repeated position before a referee ruling, especially under older rules.
    ordinary_legal_moves = state.ordinary_legal_moves
    if re.fullmatch(r"[a-i][0-9][a-i][0-9]", token, flags=re.IGNORECASE):
        move = Move.from_iccs(token)
        if move not in ordinary_legal_moves:
            raise PgnError(f"ICCS token is illegal in current position: {token}")
        return move
    if len(token) != 4:
        raise PgnError(f"unsupported Xiangqi notation token: {notation!r}")

    color = state.side_to_move
    if token[0] in "前中後后":
        order = token[0]
        piece_char = token[1]
        action_char = token[2]
        destination_char = token[3]
        source_char = None
    else:
        piece_char = token[0]
        source_char = token[1]
        action_char = token[2]
        destination_char = token[3]
        order = None
    if piece_char not in PIECE_KIND or destination_char not in NUMBERS:
        raise PgnError(f"unknown piece or numeral in token: {token}")
    kind = PIECE_KIND[piece_char]
    action = _normalized_action(action_char)
    destination = NUMBERS[destination_char]
    expected_piece = kind if color is Color.RED else kind.lower()

    legal = [
        move
        for move in ordinary_legal_moves
        if state.piece_at(move.from_square) == expected_piece and _direction(move, color) == action
    ]
    if source_char is not None:
        if source_char not in NUMBERS:
            raise PgnError(f"invalid source file in token: {token}")
        source_file = _file_from_notation(NUMBERS[source_char], color)
        legal = [move for move in legal if move.from_square.file == source_file]
    else:
        sources = sorted(
            {move.from_square for move in legal},
            key=lambda square: square.rank,
            reverse=color is Color.RED,
        )
        if order == "前" and sources:
            chosen_source = sources[0]
        elif order in {"後", "后"} and sources:
            chosen_source = sources[-1]
        elif order == "中" and len(sources) % 2 == 1:
            chosen_source = sources[len(sources) // 2]
        else:
            raise PgnError(f"cannot resolve front/middle/rear source in token: {token}")
        legal = [move for move in legal if move.from_square == chosen_source]

    if action == "平" or kind in {"N", "B", "A"}:
        target_file = _file_from_notation(destination, color)
        legal = [move for move in legal if move.to_square.file == target_file]
    else:
        legal = [
            move
            for move in legal
            if abs(move.to_square.rank - move.from_square.rank) == destination
        ]
    if len(legal) != 1:
        candidates = ", ".join(str(move) for move in legal) or "none"
        raise PgnError(f"notation {token!r} resolved to {len(legal)} moves: {candidates}")
    return legal[0]


def parse_pgn(text: str, *, max_ply: int = 10_000) -> ParsedPgn:
    normalized = unicodedata.normalize("NFKC", text).replace("\ufeff", "")
    tags: dict[str, str] = {}
    body_lines: list[str] = []
    for line in normalized.splitlines():
        match = TAG_RE.match(line.strip())
        if match:
            tags[match.group(1)] = match.group(2)
        else:
            body_lines.append(line)
    if tags.get("Game") not in {None, "Chinese Chess"}:
        raise PgnError(f"not a Chinese Chess PGN: {tags.get('Game')!r}")
    result = tags.get("Result", "")
    if result not in RESULTS:
        raise PgnError(f"PGN result must be one of {sorted(RESULTS)}, got {result!r}")
    initial_fen = tags.get("FEN", INITIAL_FEN)
    state = GameState.from_fen(initial_fen, max_ply=max_ply)
    body = _strip_variations("\n".join(body_lines))
    notations = tuple(match.group(0) for match in MOVE_TOKEN_RE.finditer(body))
    if not notations:
        raise PgnError("PGN contains no recognized moves")
    moves: list[Move] = []
    for ply, notation in enumerate(notations, 1):
        try:
            move = notation_to_move(state, notation)
            state = state.apply(move, validate=False)
        except (PgnError, ValueError) as exc:
            raise PgnError(f"failed at ply {ply}, token {notation!r}: {exc}") from exc
        moves.append(move)
    red_general = state.general_square(Color.RED)
    black_general = state.general_square(Color.BLACK)
    terminal_winner: Color | None = None
    terminal_reason: str | None = None
    if red_general is None:
        terminal_winner, terminal_reason = Color.BLACK, "general_captured"
    elif black_general is None:
        terminal_winner, terminal_reason = Color.RED, "general_captured"
    elif not state.ordinary_legal_moves:
        terminal_winner, terminal_reason = (
            state.side_to_move.opponent,
            ("checkmate" if state.is_in_check() else "stalemate"),
        )
    if terminal_winner is not None:
        expected_result = "1-0" if terminal_winner is Color.RED else "0-1"
        if result != expected_result:
            raise PgnError(
                "terminal position conflicts with declared result: "
                f"engine={expected_result} ({terminal_reason}), PGN={result}"
            )
    return ParsedPgn(
        tags=tags,
        initial_fen=initial_fen,
        result=result,
        notations=notations,
        moves=tuple(moves),
        final_state=state,
    )
