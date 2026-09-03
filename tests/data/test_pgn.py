from chessai.data.pgn import decode_pgn_bytes, notation_to_move, parse_pgn
from chessai.engine import GameState


def test_big5_pgn_decoding_and_replay() -> None:
    text = """[Game \"Chinese Chess\"]
[Event \"測試\"]
[Result \"*\"]
"""
    # Decode detection is tested with a valid event/header while replay and
    # result validation use a complete independent record below.
    decoded, encoding = decode_pgn_bytes(
        text.replace('[Result "*"]', '[Result "1-0"]').encode("big5")
    )
    assert "測試" in decoded
    assert encoding == "big5"


def test_initial_chinese_notation_resolves_by_side() -> None:
    state = GameState.initial()
    assert str(notation_to_move(state, "炮二平五")) == "h2e2"
    state = state.apply("h2e2")
    assert str(notation_to_move(state, "馬8進7")) == "h9g7"


def test_parse_complete_chinese_pgn() -> None:
    text = """[Game \"Chinese Chess\"]
[Event \"Unit test\"]
[Result \"1-0\"]

1. 炮二平五 馬8進7
2. 馬二進三 車9平8
"""
    game = parse_pgn(text)
    assert tuple(str(move) for move in game.moves) == ("h2e2", "h9g7", "h0g2", "i9h9")
    assert game.result == "1-0"


def test_import_replay_uses_ordinary_legality_past_repetition_adjudication() -> None:
    text = """[Game \"Chinese Chess\"]
[Result \"1/2-1/2\"]

1. a0a1 a9a8
2. a1a0 a8a9
3. a0a1 a9a8
4. a1a0 a8a9
5. i3i4
"""
    game = parse_pgn(text)
    repeated_state = GameState.initial().play(game.moves[:8])
    assert repeated_state.outcome().terminal
    assert str(game.moves[-1]) == "i3i4"
