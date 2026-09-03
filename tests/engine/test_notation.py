from chessai.engine import GameState, Move
from chessai.engine.notation import export_pgn, move_to_chinese


def test_chinese_notation_for_opening_cannons() -> None:
    state = GameState.initial()
    red_move = Move.from_iccs("h2e2")
    assert move_to_chinese(state, red_move) == "炮二平五"

    state = state.apply(red_move)
    black_move = Move.from_iccs("h7e7")
    assert move_to_chinese(state, black_move) == "炮8平5"


def test_pgn_export_replays_moves_and_includes_result() -> None:
    initial = GameState.initial()
    moves = [Move.from_iccs("h2e2"), Move.from_iccs("h7e7")]
    text = export_pgn(initial.to_fen(), moves, result="1/2-1/2")

    assert '[Game "Chinese Chess"]' in text
    assert '[Result "1/2-1/2"]' in text
    assert "1. 炮二平五 炮8平5" in text
