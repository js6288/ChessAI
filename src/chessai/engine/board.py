"""Immutable reference implementation of Xiangqi rules.

The move generator implements the ordinary movement and king-safety rules in
full. Repetition adjudication follows a deterministic computer-rule profile:
threefold repetition is a draw unless exactly one side gave check on every one
of its moves in the repeated segment, in which case that side loses. Direct,
same-target chases are classified as a second priority. The rule profile is
versioned because WXF edge cases are considerably more nuanced than Western
chess repetition.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property

from chessai.engine.move import Move, Square

INITIAL_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
PIECES = frozenset("KABNRCPkabnrcp")
EMPTY = "."
NO_CAPTURE_DRAW_PLIES = 120
DEFAULT_MAX_PLY = 300


class Color(StrEnum):
    RED = "red"
    BLACK = "black"

    @property
    def opponent(self) -> Color:
        return Color.BLACK if self is Color.RED else Color.RED

    @property
    def fen(self) -> str:
        return "w" if self is Color.RED else "b"

    @classmethod
    def from_fen(cls, value: str) -> Color:
        if value == "w":
            return cls.RED
        if value == "b":
            return cls.BLACK
        raise ValueError(f"invalid FEN side-to-move: {value!r}")


class GameStatus(StrEnum):
    ONGOING = "ongoing"
    RED_WIN = "red_win"
    BLACK_WIN = "black_win"
    DRAW = "draw"


@dataclass(frozen=True, slots=True)
class Outcome:
    status: GameStatus
    winner: Color | None = None
    reason: str | None = None

    @property
    def terminal(self) -> bool:
        return self.status is not GameStatus.ONGOING


@dataclass(frozen=True, slots=True)
class MoveRecord:
    move: Move
    mover: Color
    captured_piece: str | None
    gave_check: bool
    chased_targets: tuple[str, ...] = ()


def _piece_color(piece: str) -> Color:
    return Color.RED if piece.isupper() else Color.BLACK


def _belongs_to(piece: str, color: Color) -> bool:
    return piece != EMPTY and _piece_color(piece) is color


def _inside(file: int, rank: int) -> bool:
    return 0 <= file < 9 and 0 <= rank < 10


def _inside_palace(square: Square, color: Color) -> bool:
    if not 3 <= square.file <= 5:
        return False
    return 0 <= square.rank <= 2 if color is Color.RED else 7 <= square.rank <= 9


@dataclass(frozen=True)
class GameState:
    """An immutable Xiangqi position with enough history for adjudication."""

    board: tuple[str, ...]
    side_to_move: Color = Color.RED
    halfmove_clock: int = 0
    fullmove_number: int = 1
    ply: int = 0
    position_history: tuple[str, ...] = field(default_factory=tuple)
    board_history: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    move_records: tuple[MoveRecord, ...] = field(default_factory=tuple)
    max_ply: int = DEFAULT_MAX_PLY

    def __post_init__(self) -> None:
        if len(self.board) != 90:
            raise ValueError(f"a Xiangqi board needs 90 squares, got {len(self.board)}")
        invalid = set(self.board) - PIECES - {EMPTY}
        if invalid:
            raise ValueError(f"invalid board pieces: {sorted(invalid)!r}")
        if self.halfmove_clock < 0 or self.fullmove_number < 1 or self.ply < 0:
            raise ValueError("FEN clocks and ply must be non-negative")
        if self.max_ply <= 0:
            raise ValueError("max_ply must be positive")

        key = self._make_position_key(self.board, self.side_to_move)
        if not self.position_history:
            object.__setattr__(self, "position_history", (key,))
        elif self.position_history[-1] != key:
            raise ValueError("position history does not end at the current position")
        if not self.board_history:
            object.__setattr__(self, "board_history", (self.board,))
        elif self.board_history[-1] != self.board:
            raise ValueError("board history does not end at the current board")

    @classmethod
    def initial(cls, *, max_ply: int = DEFAULT_MAX_PLY) -> GameState:
        return cls.from_fen(INITIAL_FEN, max_ply=max_ply)

    @classmethod
    def from_fen(cls, fen: str, *, max_ply: int = DEFAULT_MAX_PLY) -> GameState:
        parts = fen.strip().split()
        if len(parts) < 2:
            raise ValueError("FEN must contain board and side-to-move")
        rows = parts[0].split("/")
        if len(rows) != 10:
            raise ValueError(f"Xiangqi FEN must contain 10 ranks, got {len(rows)}")

        board = [EMPTY] * 90
        for fen_row, rank in zip(rows, range(9, -1, -1), strict=True):
            file = 0
            for token in fen_row:
                if token.isdigit():
                    file += int(token)
                elif token in PIECES:
                    if file >= 9:
                        raise ValueError(f"too many squares in FEN rank {rank}")
                    board[rank * 9 + file] = token
                    file += 1
                else:
                    raise ValueError(f"invalid FEN token: {token!r}")
            if file != 9:
                raise ValueError(f"FEN rank {rank} expands to {file} files, expected 9")

        halfmove = int(parts[4]) if len(parts) >= 5 else 0
        fullmove = int(parts[5]) if len(parts) >= 6 else 1
        side = Color.from_fen(parts[1])
        return cls(
            board=tuple(board),
            side_to_move=side,
            halfmove_clock=halfmove,
            fullmove_number=fullmove,
            max_ply=max_ply,
        )

    def to_fen(self) -> str:
        rows: list[str] = []
        for rank in range(9, -1, -1):
            empty_count = 0
            output: list[str] = []
            for file in range(9):
                piece = self.board[rank * 9 + file]
                if piece == EMPTY:
                    empty_count += 1
                else:
                    if empty_count:
                        output.append(str(empty_count))
                        empty_count = 0
                    output.append(piece)
            if empty_count:
                output.append(str(empty_count))
            rows.append("".join(output))
        return (
            f"{'/'.join(rows)} {self.side_to_move.fen} - - "
            f"{self.halfmove_clock} {self.fullmove_number}"
        )

    @staticmethod
    def _make_position_key(board: tuple[str, ...], side: Color) -> str:
        return "".join(board) + ("w" if side is Color.RED else "b")

    @property
    def position_key(self) -> str:
        return self.position_history[-1]

    def piece_at(self, square: Square) -> str | None:
        piece = self.board[square.index]
        return None if piece == EMPTY else piece

    def general_square(self, color: Color) -> Square | None:
        target = "K" if color is Color.RED else "k"
        try:
            return Square.from_index(self.board.index(target))
        except ValueError:
            return None

    def is_in_check(self, color: Color | None = None) -> bool:
        checked_color = self.side_to_move if color is None else color
        general = self.general_square(checked_color)
        if general is None:
            return True
        return self.is_square_attacked(general, checked_color.opponent)

    def is_square_attacked(self, target: Square, by_color: Color) -> bool:
        for index, piece in enumerate(self.board):
            if piece == EMPTY or not _belongs_to(piece, by_color):
                continue
            origin = Square.from_index(index)
            if self._piece_attacks(origin, target, piece):
                return True
        return False

    def _piece_attacks(self, origin: Square, target: Square, piece: str) -> bool:
        color = _piece_color(piece)
        kind = piece.upper()
        df = target.file - origin.file
        dr = target.rank - origin.rank

        if kind == "R":
            return self._line_blockers(origin, target) == 0
        if kind == "C":
            return self._line_blockers(origin, target) == 1
        if kind == "N":
            if (abs(df), abs(dr)) not in {(1, 2), (2, 1)}:
                return False
            leg = (
                Square(origin.file, origin.rank + (1 if dr > 0 else -1))
                if abs(dr) == 2
                else Square(origin.file + (1 if df > 0 else -1), origin.rank)
            )
            return self.board[leg.index] == EMPTY
        if kind == "B":
            if abs(df) != 2 or abs(dr) != 2:
                return False
            if color is Color.RED and target.rank > 4:
                return False
            if color is Color.BLACK and target.rank < 5:
                return False
            eye = Square(origin.file + df // 2, origin.rank + dr // 2)
            return self.board[eye.index] == EMPTY
        if kind == "A":
            return abs(df) == 1 and abs(dr) == 1 and _inside_palace(target, color)
        if kind == "K":
            if abs(df) + abs(dr) == 1 and _inside_palace(target, color):
                return True
            opposing_general = "k" if color is Color.RED else "K"
            return (
                df == 0
                and self.board[target.index] == opposing_general
                and self._line_blockers(origin, target) == 0
            )
        if kind == "P":
            forward = 1 if color is Color.RED else -1
            if df == 0 and dr == forward:
                return True
            crossed = origin.rank >= 5 if color is Color.RED else origin.rank <= 4
            return crossed and dr == 0 and abs(df) == 1
        raise AssertionError(f"unknown piece: {piece}")

    def _line_blockers(self, origin: Square, target: Square) -> int:
        if origin.file != target.file and origin.rank != target.rank:
            return -1
        df = 0 if origin.file == target.file else (1 if target.file > origin.file else -1)
        dr = 0 if origin.rank == target.rank else (1 if target.rank > origin.rank else -1)
        file, rank = origin.file + df, origin.rank + dr
        blockers = 0
        while (file, rank) != (target.file, target.rank):
            if self.board[rank * 9 + file] != EMPTY:
                blockers += 1
            file += df
            rank += dr
        return blockers

    def pseudo_legal_moves(self, color: Color | None = None) -> tuple[Move, ...]:
        mover = self.side_to_move if color is None else color
        moves: list[Move] = []
        for index, piece in enumerate(self.board):
            if piece == EMPTY or not _belongs_to(piece, mover):
                continue
            moves.extend(self._moves_for_piece(Square.from_index(index), piece))
        return tuple(moves)

    def _moves_for_piece(self, origin: Square, piece: str) -> Iterator[Move]:
        color = _piece_color(piece)
        kind = piece.upper()

        if kind in {"R", "C"}:
            for df, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                file, rank = origin.file + df, origin.rank + dr
                screened = False
                while _inside(file, rank):
                    target = Square(file, rank)
                    occupant = self.board[target.index]
                    if kind == "R":
                        if occupant == EMPTY:
                            yield Move(origin, target)
                        else:
                            if not _belongs_to(occupant, color):
                                yield Move(origin, target)
                            break
                    elif not screened:
                        if occupant == EMPTY:
                            yield Move(origin, target)
                        else:
                            screened = True
                    elif occupant != EMPTY:
                        if not _belongs_to(occupant, color):
                            yield Move(origin, target)
                        break
                    file += df
                    rank += dr
            return

        offsets: tuple[tuple[int, int], ...]
        if kind == "N":
            offsets = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
        elif kind == "B":
            offsets = ((2, 2), (2, -2), (-2, -2), (-2, 2))
        elif kind == "A":
            offsets = ((1, 1), (1, -1), (-1, -1), (-1, 1))
        elif kind == "K":
            offsets = ((1, 0), (0, 1), (-1, 0), (0, -1))
        elif kind == "P":
            forward = 1 if color is Color.RED else -1
            crossed = origin.rank >= 5 if color is Color.RED else origin.rank <= 4
            offsets = ((0, forward), (1, 0), (-1, 0)) if crossed else ((0, forward),)
        else:
            raise AssertionError(f"unknown piece: {piece}")

        for df, dr in offsets:
            file, rank = origin.file + df, origin.rank + dr
            if not _inside(file, rank):
                continue
            target = Square(file, rank)
            occupant = self.board[target.index]
            if occupant != EMPTY and _belongs_to(occupant, color):
                continue
            if self._piece_attacks(origin, target, piece):
                yield Move(origin, target)

        if kind == "K":
            opposing_general = self.general_square(color.opponent)
            if (
                opposing_general is not None
                and opposing_general.file == origin.file
                and self._line_blockers(origin, opposing_general) == 0
            ):
                yield Move(origin, opposing_general)

    def reference_legal_moves(self) -> tuple[Move, ...]:
        """Generate legal moves in pure Python, bypassing the optional accelerator."""

        legal: list[Move] = []
        for move in self.pseudo_legal_moves():
            child = self._apply_unchecked(move, record_history=False)
            if child.general_square(self.side_to_move) is not None and not child.is_in_check(
                self.side_to_move
            ):
                legal.append(move)
        return tuple(legal)

    @cached_property
    def ordinary_legal_moves(self) -> tuple[Move, ...]:
        """Generate moves without applying repetition or the configured ply limit."""

        # Import lazily to keep the reference engine independently importable.
        from chessai.native import legal_moves as native_legal_moves

        accelerated = native_legal_moves(self.to_fen())
        if accelerated is not None:
            return tuple(Move.from_iccs(move) for move in accelerated)
        return self.reference_legal_moves()

    @cached_property
    def legal_moves(self) -> tuple[Move, ...]:
        if self._terminal_without_move_generation().terminal:
            return ()
        return self.ordinary_legal_moves

    def apply(self, move: Move | str, *, validate: bool = True) -> GameState:
        parsed = Move.from_iccs(move) if isinstance(move, str) else move
        if validate and parsed not in self.legal_moves:
            raise ValueError(f"illegal move {parsed} in {self.to_fen()}")
        return self._apply_unchecked(parsed, record_history=True)

    def _apply_unchecked(self, move: Move, *, record_history: bool) -> GameState:
        board = list(self.board)
        piece = board[move.from_square.index]
        captured = board[move.to_square.index]
        if piece == EMPTY:
            raise ValueError(f"no piece at {move.from_square}")
        board[move.to_square.index] = piece
        board[move.from_square.index] = EMPTY
        next_side = self.side_to_move.opponent
        new_board = tuple(board)
        key = self._make_position_key(new_board, next_side)

        if not record_history:
            return GameState(
                board=new_board,
                side_to_move=next_side,
                halfmove_clock=0 if captured != EMPTY else self.halfmove_clock + 1,
                fullmove_number=self.fullmove_number
                + (1 if self.side_to_move is Color.BLACK else 0),
                ply=self.ply + 1,
                position_history=(key,),
                board_history=(new_board,),
                max_ply=self.max_ply,
            )

        provisional = GameState(
            board=new_board,
            side_to_move=next_side,
            halfmove_clock=0 if captured != EMPTY else self.halfmove_clock + 1,
            fullmove_number=self.fullmove_number + (1 if self.side_to_move is Color.BLACK else 0),
            ply=self.ply + 1,
            position_history=(*self.position_history, key),
            board_history=(*self.board_history[-7:], new_board),
            move_records=self.move_records,
            max_ply=self.max_ply,
        )
        record = MoveRecord(
            move=move,
            mover=self.side_to_move,
            captured_piece=None if captured == EMPTY else captured,
            gave_check=provisional.is_in_check(next_side),
            chased_targets=provisional._direct_chase_targets(self.side_to_move),
        )
        return GameState(
            board=provisional.board,
            side_to_move=provisional.side_to_move,
            halfmove_clock=provisional.halfmove_clock,
            fullmove_number=provisional.fullmove_number,
            ply=provisional.ply,
            position_history=provisional.position_history,
            board_history=provisional.board_history,
            move_records=(*self.move_records, record),
            max_ply=self.max_ply,
        )

    def _direct_chase_targets(self, attacker: Color) -> tuple[str, ...]:
        """Return stable signatures for directly attacked, undefended chase targets.

        This deliberately conservative classifier excludes generals and soldiers
        and requires the victim to be undefended. It avoids falsely declaring a
        loss for ambiguous exchange invitations.
        """

        targets: list[str] = []
        for index, piece in enumerate(self.board):
            if piece == EMPTY or _piece_color(piece) is attacker or piece.upper() in {"K", "P"}:
                continue
            square = Square.from_index(index)
            if self.is_square_attacked(square, attacker) and not self.is_square_attacked(
                square, attacker.opponent
            ):
                targets.append(f"{piece.upper()}@{square}")
        return tuple(sorted(targets))

    def repetition_count(self) -> int:
        return self.position_history.count(self.position_key)

    def _repetition_outcome(self) -> Outcome:
        occurrences = [
            index for index, key in enumerate(self.position_history) if key == self.position_key
        ]
        if len(occurrences) < 3:
            return Outcome(GameStatus.ONGOING)
        start = occurrences[-3]
        records = self.move_records[start:]
        if not records:
            return Outcome(GameStatus.DRAW, reason="threefold_repetition")

        perpetual_checkers = [
            color
            for color in (Color.RED, Color.BLACK)
            if any(record.mover is color for record in records)
            and all(record.gave_check for record in records if record.mover is color)
        ]
        if len(perpetual_checkers) == 1:
            loser = perpetual_checkers[0]
            winner = loser.opponent
            return Outcome(
                GameStatus.RED_WIN if winner is Color.RED else GameStatus.BLACK_WIN,
                winner=winner,
                reason="perpetual_check_violation",
            )

        perpetual_chasers: list[Color] = []
        for color in (Color.RED, Color.BLACK):
            own_records = [record for record in records if record.mover is color]
            if own_records and all(record.chased_targets for record in own_records):
                common = set(own_records[0].chased_targets)
                for record in own_records[1:]:
                    common.intersection_update(record.chased_targets)
                if common:
                    perpetual_chasers.append(color)
        if len(perpetual_chasers) == 1:
            loser = perpetual_chasers[0]
            winner = loser.opponent
            return Outcome(
                GameStatus.RED_WIN if winner is Color.RED else GameStatus.BLACK_WIN,
                winner=winner,
                reason="perpetual_chase_violation",
            )
        return Outcome(GameStatus.DRAW, reason="threefold_repetition")

    def _terminal_without_move_generation(self) -> Outcome:
        red_general = self.general_square(Color.RED)
        black_general = self.general_square(Color.BLACK)
        if red_general is None:
            return Outcome(GameStatus.BLACK_WIN, Color.BLACK, "general_captured")
        if black_general is None:
            return Outcome(GameStatus.RED_WIN, Color.RED, "general_captured")
        repetition = self._repetition_outcome()
        if repetition.terminal:
            return repetition
        if self.halfmove_clock >= NO_CAPTURE_DRAW_PLIES:
            return Outcome(GameStatus.DRAW, reason="no_capture_limit")
        if self.ply >= self.max_ply:
            return Outcome(GameStatus.DRAW, reason="max_ply_limit")
        return Outcome(GameStatus.ONGOING)

    def outcome(self) -> Outcome:
        terminal = self._terminal_without_move_generation()
        if terminal.terminal:
            return terminal
        if self.legal_moves:
            return terminal
        winner = self.side_to_move.opponent
        return Outcome(
            GameStatus.RED_WIN if winner is Color.RED else GameStatus.BLACK_WIN,
            winner=winner,
            reason="checkmate" if self.is_in_check() else "stalemate",
        )

    def legal_move_strings(self) -> tuple[str, ...]:
        return tuple(str(move) for move in self.legal_moves)

    def play(self, moves: Iterable[Move | str]) -> GameState:
        state = self
        for move in moves:
            state = state.apply(move)
        return state
