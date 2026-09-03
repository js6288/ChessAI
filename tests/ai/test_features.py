import numpy as np

from chessai.ai.features import INPUT_PLANES, encode_batch, encode_state, legal_action_mask
from chessai.engine import GameState


def test_feature_shape_dtype_and_piece_counts() -> None:
    features = encode_state(GameState.initial())
    assert features.shape == (INPUT_PLANES, 10, 9)
    assert features.dtype == np.float32
    assert int(features[:14].sum()) == 32
    assert int(features[14:112].sum()) == 0


def test_history_and_last_move_planes() -> None:
    state = GameState.initial().apply("h2e2")
    features = encode_state(state)
    assert int(features[:112].sum()) == 64
    assert int(features[112].sum()) == 1
    assert int(features[113].sum()) == 1


def test_legal_mask_matches_legal_moves() -> None:
    state = GameState.initial()
    mask = legal_action_mask(state)
    assert mask.shape == (2086,)
    assert mask.dtype == np.bool_
    assert int(mask.sum()) == len(state.legal_moves) == 44


def test_empty_batch_has_stable_shape() -> None:
    assert encode_batch([]).shape == (0, INPUT_PLANES, 10, 9)
