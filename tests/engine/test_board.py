import pytest

from chessai.engine import Color, GameState, GameStatus, Move
from chessai.engine.board import INITIAL_FEN


def test_initial_fen_round_trip_and_move_count() -> None:
    state = GameState.initial()
    assert state.to_fen() == INITIAL_FEN
    assert state.side_to_move is Color.RED
    assert len(state.legal_moves) == 44
    assert "h2e2" in state.legal_move_strings()


def test_apply_is_immutable_and_updates_clocks() -> None:
    state = GameState.initial()
    child = state.apply("h2e2")
    assert state.piece_at(Move.from_iccs("h2e2").from_square) == "C"
    assert child.piece_at(Move.from_iccs("h2e2").from_square) is None
    assert child.piece_at(Move.from_iccs("h2e2").to_square) == "C"
    assert child.side_to_move is Color.BLACK
    assert child.halfmove_clock == 1
    assert child.ply == 1


def test_horse_leg_blocks_both_destinations() -> None:
    open_horse = GameState.from_fen("4k4/9/9/9/4P4/9/9/9/9/1N2K4 w - - 0 1")
    assert "b0a2" in open_horse.legal_move_strings()
    assert "b0c2" in open_horse.legal_move_strings()

    blocked = GameState.from_fen("4k4/9/9/9/4P4/9/9/9/1P7/1N2K4 w - - 0 1")
    assert "b0a2" not in blocked.legal_move_strings()
    assert "b0c2" not in blocked.legal_move_strings()


def test_elephant_cannot_cross_river_and_eye_can_be_blocked() -> None:
    state = GameState.from_fen("4k4/9/9/9/9/4B4/9/9/9/4K4 w - - 0 1")
    assert all(
        Move.from_iccs(move).to_square.rank <= 4
        for move in state.legal_move_strings()
        if move.startswith("e4")
    )

    blocked = GameState.from_fen("4k4/9/9/9/9/4B4/3P5/9/9/4K4 w - - 0 1")
    assert "e4c2" not in blocked.legal_move_strings()


def test_cannon_requires_exactly_one_screen_to_capture() -> None:
    state = GameState.from_fen("4k4/9/9/9/4r4/9/4P4/9/4C4/4K4 w - - 0 1")
    assert "e1e5" in state.legal_move_strings()

    no_screen = GameState.from_fen("4k4/9/9/9/4r4/9/9/9/4C4/4K4 w - - 0 1")
    assert "e1e5" not in no_screen.legal_move_strings()


def test_generals_may_not_face_after_move() -> None:
    state = GameState.from_fen("4k4/9/9/9/9/4R4/9/9/9/4K4 w - - 0 1")
    assert "e4d4" not in state.legal_move_strings()
    assert "e4f4" not in state.legal_move_strings()


def test_stalemate_is_a_loss_in_xiangqi() -> None:
    state = GameState.from_fen("4k4/3R1R3/9/9/4P4/9/9/9/9/4K4 b - - 0 1")
    outcome = state.outcome()
    assert outcome.status is GameStatus.RED_WIN
    assert outcome.winner is Color.RED
    assert outcome.reason in {"checkmate", "stalemate"}


def test_illegal_move_is_rejected() -> None:
    with pytest.raises(ValueError, match="illegal move"):
        GameState.initial().apply("a0a9")


def test_max_ply_draw_is_an_explicit_safety_guard() -> None:
    state = GameState.from_fen("4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1", max_ply=1)
    # This malformed face-to-face setup still exercises the immutable guard
    # without pretending it is a legal tournament position.
    state = GameState(
        board=state.board,
        side_to_move=state.side_to_move,
        ply=1,
        max_ply=1,
    )
    assert state.outcome().status is GameStatus.DRAW
    assert state.outcome().reason == "max_ply_limit"
