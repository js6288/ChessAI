import numpy as np
import pytest

from chessai.ai.search import (
    GumbelSearch,
    Node,
    complete_qvalues,
    completed_q_policy,
    sample_gumbel_top_k,
    stable_softmax,
)
from chessai.engine import GameState
from chessai.engine.vocabulary import action_labels


def test_gumbel_top_k_is_unique_and_seeded() -> None:
    logits = np.asarray([0.0, 1.0, 2.0, 3.0])
    first, _ = sample_gumbel_top_k(logits, 3, np.random.default_rng(7))
    second, _ = sample_gumbel_top_k(logits, 3, np.random.default_rng(7))
    assert np.array_equal(first, second)
    assert len(set(first.tolist())) == 3


def test_q_completion_uses_network_value_for_unvisited_actions() -> None:
    completed = complete_qvalues(np.asarray([0.25, -0.5, 0.0]), np.asarray([4, 2, 0]), value=0.75)
    assert completed.tolist() == [0.25, -0.5, 0.75]


def test_completed_q_policy_improves_visited_action_and_normalizes() -> None:
    policy = completed_q_policy(
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([0.6, -0.2, 0.0]),
        np.asarray([1, 1, 0]),
        value=0.1,
        scale=3.0,
    )

    assert float(policy.sum()) == pytest.approx(1.0)
    assert policy[0] > policy[2] > policy[1]


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


def test_equal_root_visits_choose_sequential_halving_winner_and_completed_q_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = GameState.initial()
    root_moves = state.legal_moves
    target = root_moves[10]

    class RootMoveEvaluator:
        def evaluate(self, position: GameState) -> tuple[np.ndarray, float]:
            logits = np.zeros(len(action_labels()), dtype=np.float64)
            if position.ply == 0:
                return logits, 0.0
            last = position.move_records[-1].move
            # Values are from the child side-to-move perspective, so the
            # desirable root move must be negative at its child.
            return logits, -0.8 if last == target else 0.2

    def first_actions(
        logits: np.ndarray, k: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        del rng
        return np.arange(k, dtype=np.int64), np.zeros(logits.size, dtype=np.float64)

    monkeypatch.setattr("chessai.ai.search.sample_gumbel_top_k", first_actions)
    result = GumbelSearch(
        RootMoveEvaluator(),
        simulations=16,
        max_considered_actions=16,
        c_visit=2.0,
        c_scale=1.0,
        seed=7,
    ).search(state)

    assert {candidate.visits for candidate in result.candidates} == {1}
    assert result.best_move == target
    assert len(result.root_policy) == len(root_moves)
    assert result.root_policy[str(target)] == max(result.root_policy.values())
    assert sum(result.root_policy.values()) == pytest.approx(1.0)


def test_search_rejects_invalid_completed_q_scales() -> None:
    with pytest.raises(ValueError, match="c_visit"):
        GumbelSearch(c_visit=-1.0)
    with pytest.raises(ValueError, match="c_scale"):
        GumbelSearch(c_scale=0.0)


def test_interior_selection_corrects_an_overvisited_action() -> None:
    state = GameState.initial()
    first, second = state.legal_moves[:2]
    node = Node(
        state=state,
        priors={first: 0.5, second: 0.5},
        network_value=0.0,
        expanded=True,
    )
    node.children[first] = Node(
        state=state.apply(first),
        prior=0.5,
        visit_count=4,
        value_sum=0.0,
    )

    search = GumbelSearch(simulations=8, c_visit=2.0)
    assert search._select_interior(node) == second
