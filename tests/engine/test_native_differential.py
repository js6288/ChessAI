import os
import random

import pytest

from chessai.engine import GameState
from chessai.native import available, legal_move_codes, module


@pytest.mark.skipif(not available(), reason="optional native backend is not built")
def test_native_perft_and_random_reachable_positions_match_reference() -> None:
    native = module()
    initial = GameState.initial()
    assert native.RULE_VERSION == "wxf-2018-computer-v1"
    assert native.perft(initial.to_fen(), 1) == 44
    assert native.perft(initial.to_fen(), 2) == 1_920
    assert native.perft(initial.to_fen(), 3) == 79_666

    clock_terminal = GameState.from_fen(
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 120 61"
    )
    assert clock_terminal.outcome().terminal
    assert clock_terminal.legal_moves == ()
    reference_clock_moves = {str(move) for move in clock_terminal.reference_legal_moves()}
    assert len(reference_clock_moves) == 44
    assert set(native.legal_moves(clock_terminal.to_fen())) == reference_clock_moves

    randomizer = random.Random(20260902)
    state = initial
    target_positions = int(os.getenv("CHESSAI_NATIVE_DIFF_POSITIONS", "250"))
    assert target_positions > 0
    checked = 0
    while checked < target_positions:
        # A bare FEN cannot encode repetition or the configured max-ply
        # guard; those history-aware terminal states remain Python-owned.
        if state.outcome().terminal:
            state = initial
        reference_moves = {str(move) for move in state.reference_legal_moves()}
        assert set(native.legal_moves(state.to_fen())) == reference_moves
        codes = legal_move_codes(state.to_fen())
        assert codes is not None
        compact_moves = {
            f"{'abcdefghi'[code // 90 % 9]}{code // 90 // 9}"
            f"{'abcdefghi'[code % 90 % 9]}{code % 90 // 9}"
            for code in codes
        }
        assert compact_moves == reference_moves
        assert native.position_key(state.to_fen()) == state.position_key
        assert native.is_in_check(state.to_fen()) == state.is_in_check()
        if hasattr(native, "is_in_check_board"):
            assert native.is_in_check_board("".join(state.board), state.side_to_move.value) == (
                state.is_in_check()
            )
        checked += 1
        move = randomizer.choice(state.legal_moves)
        native_fen = native.apply_move(state.to_fen(), str(move))
        state = state.apply(move)
        assert native_fen == state.to_fen()
