import numpy as np
import pytest

from chessai.ai.search import (
    GumbelSearch,
    complete_qvalues,
    sample_gumbel_top_k,
    stable_softmax,
)
from chessai.engine import GameState


def test_gumbel_top_k_is_unique_and_seeded() -> None:
    logits = np.asarray([0.0, 1.0, 2.0, 3.0])
    first, _ = sample_gumbel_top_k(logits, 3, np.random.default_rng(7))
    second, _ = sample_gumbel_top_k(logits, 3, np.random.default_rng(7))
    assert np.array_equal(first, second)
    assert len(set(first.tolist())) == 3


def test_q_completion_uses_network_value_for_unvisited_actions() -> None:
    completed = complete_qvalues(np.asarray([0.25, -0.5, 0.0]), np.asarray([4, 2, 0]), value=0.75)
    assert completed.tolist() == [0.25, -0.5, 0.75]


def test_stable_softmax_rejects_nan() -> None:
    with pytest.raises(ValueError):
        stable_softmax(np.asarray([0.0, np.nan]))


def test_search_returns_legal_normalized_result() -> None:
    state = GameState.initial()
    result = GumbelSearch(simulations=8, max_considered_actions=8, seed=11).search(state)
    assert result.best_move in state.legal_moves
    assert result.visits == 8
    assert sum(result.root_policy.values()) == pytest.approx(1.0)
    assert result.principal_variation[0] == str(result.best_move)
    assert result.elapsed_ms > 0
