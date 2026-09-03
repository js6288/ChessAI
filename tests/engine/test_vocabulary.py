import hashlib

from chessai.engine import Color, GameState, Move
from chessai.engine.vocabulary import (
    action_labels,
    action_vocab_hash,
    action_vocab_payload,
    decode_move,
    encode_move,
)


def test_vocabulary_is_frozen_and_unique() -> None:
    labels = action_labels()
    assert len(labels) == 2086
    assert len(set(labels)) == 2086
    assert len(action_vocab_hash()) == 64
    payload = action_vocab_payload()
    assert b"\r" not in payload
    assert payload.count(b"\n") == 2087
    assert hashlib.sha256(payload).hexdigest() == action_vocab_hash()


def test_move_encoding_round_trip_for_both_canonical_views() -> None:
    for label in ("a0a9", "b0c2", "c0e2", "d0e1"):
        move = Move.from_iccs(label)
        assert decode_move(encode_move(move)) == move
        assert decode_move(encode_move(move, canonical_black=True), canonical_black=True) == move


def test_every_initial_legal_move_has_an_action() -> None:
    state = GameState.initial()
    assert state.side_to_move is Color.RED
    for move in state.legal_moves:
        assert decode_move(encode_move(move)) == move
