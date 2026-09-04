"""Compute-bounded Gumbel AlphaZero tree search.

The root uses Gumbel-Top-k sampling without replacement and Sequential
Halving. Interior nodes use completed Q-values with a prior-weighted policy
improvement score. The implementation emphasizes determinism and inspectable
results; the cloud actor layer supplies batching around the evaluator.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from chessai.engine import Color, GameState, Move
from chessai.engine.board import GameStatus
from chessai.engine.vocabulary import action_labels, encode_move

FloatArray = npt.NDArray[np.float64]


class Evaluator(Protocol):
    def evaluate(self, state: GameState) -> tuple[npt.NDArray[np.floating], float]:
        """Return full-vocabulary logits and value from side-to-move perspective."""


PIECE_VALUES = {"K": 0.0, "R": 9.0, "C": 4.5, "N": 4.0, "B": 2.0, "A": 2.0, "P": 1.0}


class HeuristicEvaluator:
    """Dependency-free fallback that keeps the GUI playable before training."""

    def evaluate(self, state: GameState) -> tuple[npt.NDArray[np.float64], float]:
        logits = np.full(len(action_labels()), -20.0, dtype=np.float64)
        own_material = 0.0
        enemy_material = 0.0
        for piece in state.board:
            if piece == ".":
                continue
            value = PIECE_VALUES[piece.upper()]
            if (piece.isupper() and state.side_to_move is Color.RED) or (
                piece.islower() and state.side_to_move is Color.BLACK
            ):
                own_material += value
            else:
                enemy_material += value
        canonical_black = state.side_to_move is Color.BLACK
        for move in state.legal_moves:
            score = 0.0
            target = state.piece_at(move.to_square)
            if target is not None:
                score += PIECE_VALUES[target.upper()]
            child = state.apply(move)
            if child.is_in_check():
                score += 1.25
            if child.outcome().winner is state.side_to_move:
                score += 100.0
            logits[encode_move(move, canonical_black=canonical_black)] = score
        value = math.tanh((own_material - enemy_material) / 12.0)
        return logits, value


@dataclass(slots=True)
class Node:
    state: GameState
    prior: float = 1.0
    visit_count: int = 0
    value_sum: float = 0.0
    priors: dict[Move, float] = field(default_factory=dict)
    children: dict[Move, Node] = field(default_factory=dict)
    network_value: float = 0.0
    expanded: bool = False

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def child(self, move: Move) -> Node:
        if move not in self.children:
            self.children[move] = Node(
                self.state.apply(move, validate=False), prior=self.priors[move]
            )
        return self.children[move]


@dataclass(frozen=True, slots=True)
class CandidateStat:
    move: str
    probability: float
    visits: int
    q_value: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    best_move: Move
    root_policy: dict[str, float]
    value: float
    visits: int
    principal_variation: tuple[str, ...]
    elapsed_ms: float
    candidates: tuple[CandidateStat, ...]


def stable_softmax(values: npt.NDArray[np.floating]) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("softmax input must be non-empty and finite")
    shifted = array - float(np.max(array))
    exponentials = np.exp(shifted)
    total = float(exponentials.sum())
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("invalid softmax normalization")
    return exponentials / total


def sample_gumbel_top_k(
    logits: npt.NDArray[np.floating], k: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.int64], FloatArray]:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("logits must be a non-empty vector")
    if not 1 <= k <= values.size:
        raise ValueError(f"k must be in [1, {values.size}], got {k}")
    uniform = np.clip(rng.random(values.size), 1e-12, 1.0 - 1e-12)
    gumbels = -np.log(-np.log(uniform))
    scores = values + gumbels
    indices = np.argpartition(scores, -k)[-k:]
    indices = indices[np.argsort(scores[indices])[::-1]]
    return indices.astype(np.int64), gumbels


def complete_qvalues(
    qvalues: npt.NDArray[np.floating],
    visit_counts: npt.NDArray[np.integer],
    value: float,
) -> FloatArray:
    q = np.asarray(qvalues, dtype=np.float64)
    visits = np.asarray(visit_counts)
    if q.shape != visits.shape:
        raise ValueError("qvalues and visit_counts must have identical shapes")
    completed = q.copy()
    completed[visits == 0] = float(value)
    return completed


def completed_q_policy(
    logits: npt.NDArray[np.floating],
    qvalues: npt.NDArray[np.floating],
    visit_counts: npt.NDArray[np.integer],
    value: float,
    scale: float,
) -> FloatArray:
    """Return the Gumbel AlphaZero improved policy for one node.

    Unvisited actions use the node's network value through Q-value
    completion.  The caller supplies the visit-dependent sigma scale from
    the paper, ``(c_visit + max_b N(b)) * c_scale``.
    """

    policy_logits = np.asarray(logits, dtype=np.float64)
    q = np.asarray(qvalues, dtype=np.float64)
    visits = np.asarray(visit_counts)
    if policy_logits.shape != q.shape or q.shape != visits.shape:
        raise ValueError("logits, qvalues, and visit_counts must have identical shapes")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("completed-Q policy scale must be finite and positive")
    completed = complete_qvalues(q, visits, value)
    return stable_softmax(policy_logits + scale * completed)


class GumbelSearch:
    def __init__(
        self,
        evaluator: Evaluator | None = None,
        *,
        simulations: int = 32,
        max_considered_actions: int = 16,
        c_visit: float = 50.0,
        c_scale: float = 1.0,
        seed: int = 0,
    ) -> None:
        if simulations <= 0:
            raise ValueError("simulations must be positive")
        if max_considered_actions <= 0:
            raise ValueError("max_considered_actions must be positive")
        if not math.isfinite(c_visit) or c_visit <= 0.0:
            raise ValueError("c_visit must be finite and positive")
        if not math.isfinite(c_scale) or c_scale <= 0.0:
            raise ValueError("c_scale must be finite and positive")
        self.evaluator = evaluator or HeuristicEvaluator()
        self.simulations = simulations
        self.max_considered_actions = max_considered_actions
        self.c_visit = c_visit
        self.c_scale = c_scale
        self.rng = np.random.default_rng(seed)

    def search(self, state: GameState, *, temperature: float = 0.0) -> SearchResult:
        started = time.perf_counter()
        if state.outcome().terminal:
            raise ValueError("cannot search a terminal position")
        root = Node(state)
        self._expand(root)
        moves = tuple(root.priors)
        if not moves:
            raise ValueError("position has no legal actions")

        legal_logits = np.asarray(
            [math.log(max(root.priors[move], 1e-300)) for move in moves],
            dtype=np.float64,
        )
        k = min(len(moves), self.max_considered_actions, self.simulations)
        sampled_indices, gumbels = sample_gumbel_top_k(legal_logits, k, self.rng)
        candidates = [moves[int(index)] for index in sampled_indices]
        root_gumbels = {
            move: float(gumbels[int(index)])
            for move, index in zip(candidates, sampled_indices, strict=True)
        }
        root_logits = {
            move: float(legal_logits[int(index)])
            for move, index in zip(candidates, sampled_indices, strict=True)
        }

        remaining_budget = self.simulations
        while len(candidates) > 1 and remaining_budget > 0:
            rounds_left = max(1, math.ceil(math.log2(len(candidates))))
            visits_each = max(1, remaining_budget // (len(candidates) * rounds_left))
            for move in candidates:
                for _ in range(min(visits_each, remaining_budget)):
                    self._simulate_forced_root(root, move)
                    remaining_budget -= 1
                    if remaining_budget == 0:
                        break
                if remaining_budget == 0:
                    break
            ranked = sorted(
                candidates,
                key=lambda move: self._root_rank(root, move, root_logits[move], root_gumbels[move]),
                reverse=True,
            )
            candidates = ranked[: max(1, math.ceil(len(ranked) / 2))]

        chosen = candidates[0]
        while remaining_budget > 0:
            self._simulate_forced_root(root, chosen)
            remaining_budget -= 1

        visited_moves = [move for move in moves if self._child_visits(root, move) > 0]
        if not visited_moves:
            visited_moves = [chosen]
        counts = np.asarray(
            [root.children[move].visit_count for move in visited_moves], dtype=np.float64
        )
        qvalues, all_visits = self._action_statistics(root, moves)
        target_probabilities = completed_q_policy(
            legal_logits,
            qvalues,
            all_visits,
            root.network_value,
            self._q_scale(root),
        )
        if temperature <= 0:
            # Sequential Halving's surviving action is the search result.  A
            # visit-count argmax is incorrect when all sampled root actions
            # receive the same small budget (the common 16-simulation case).
            selected = chosen
        else:
            scaled = np.power(np.maximum(counts, 1e-12), 1.0 / temperature)
            selection_probabilities = scaled / scaled.sum()
            selected = visited_moves[
                int(self.rng.choice(len(visited_moves), p=selection_probabilities))
            ]

        policy = {
            str(move): float(probability)
            for move, probability in zip(moves, target_probabilities, strict=True)
        }
        stats = tuple(
            CandidateStat(
                move=str(move),
                probability=policy[str(move)],
                visits=root.children[move].visit_count,
                q_value=-root.children[move].q_value,
            )
            for move in sorted(
                visited_moves,
                key=lambda item: (root.children[item].visit_count, policy[str(item)]),
                reverse=True,
            )
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return SearchResult(
            best_move=selected,
            root_policy=policy,
            value=root.q_value,
            visits=root.visit_count,
            principal_variation=self._principal_variation(root, selected),
            elapsed_ms=elapsed_ms,
            candidates=stats,
        )

    def _expand(self, node: Node) -> float:
        outcome = node.state.outcome()
        if outcome.terminal:
            node.expanded = True
            if outcome.status is GameStatus.DRAW:
                node.network_value = 0.0
            else:
                node.network_value = 1.0 if outcome.winner is node.state.side_to_move else -1.0
            return node.network_value

        logits, value = self.evaluator.evaluate(node.state)
        logits_array = np.asarray(logits, dtype=np.float64)
        if logits_array.shape != (len(action_labels()),) or not np.all(np.isfinite(logits_array)):
            raise ValueError(f"evaluator logits must be finite and shape {(len(action_labels()),)}")
        value = float(value)
        if not math.isfinite(value) or not -1.0001 <= value <= 1.0001:
            raise ValueError(f"evaluator value outside [-1, 1]: {value}")
        legal_moves = node.state.legal_moves
        if not legal_moves:
            raise ValueError("ongoing node unexpectedly has no legal actions")
        canonical_black = node.state.side_to_move is Color.BLACK
        legal_logits = np.asarray(
            [
                logits_array[encode_move(move, canonical_black=canonical_black)]
                for move in legal_moves
            ]
        )
        priors = stable_softmax(legal_logits)
        node.priors = {move: float(prior) for move, prior in zip(legal_moves, priors, strict=True)}
        node.network_value = value
        node.expanded = True
        return value

    def _simulate_forced_root(self, root: Node, root_move: Move) -> None:
        child = root.child(root_move)
        path = [root, child]
        node = child
        while node.expanded and node.priors:
            move = self._select_interior(node)
            node = node.child(move)
            path.append(node)
        leaf_value = node.network_value if node.expanded else self._expand(node)
        self._backup(path, leaf_value)

    def _select_interior(self, node: Node) -> Move:
        moves = tuple(node.priors)
        qvalues, visits = self._action_statistics(node, moves)
        logits = np.log(
            np.maximum(
                np.asarray([node.priors[move] for move in moves], dtype=np.float64),
                1e-300,
            )
        )
        improved_policy = completed_q_policy(
            logits,
            qvalues,
            visits,
            node.network_value,
            self._q_scale(node),
        )
        # Eq. 14: visit actions whose current share falls furthest below the
        # completed-Q improved policy.  The +1 denominator supplies the next
        # visit being allocated.
        visit_share = visits.astype(np.float64) / (1.0 + float(visits.sum()))
        scores = improved_policy - visit_share
        return moves[int(np.argmax(scores))]

    def _root_rank(self, root: Node, move: Move, logit: float, gumbel: float) -> float:
        child = root.children.get(move)
        q = -child.q_value if child is not None and child.visit_count else root.network_value
        return gumbel + logit + self._q_scale(root) * q

    @staticmethod
    def _child_visits(node: Node, move: Move) -> int:
        child = node.children.get(move)
        return child.visit_count if child is not None else 0

    @classmethod
    def _action_statistics(
        cls, node: Node, moves: tuple[Move, ...]
    ) -> tuple[FloatArray, npt.NDArray[np.int64]]:
        visits = np.asarray([cls._child_visits(node, move) for move in moves], dtype=np.int64)
        qvalues = np.asarray(
            [
                -node.children[move].q_value if visits[index] > 0 else 0.0
                for index, move in enumerate(moves)
            ],
            dtype=np.float64,
        )
        return qvalues, visits

    def _q_scale(self, node: Node) -> float:
        max_visits = max((child.visit_count for child in node.children.values()), default=0)
        return (self.c_visit + max_visits) * self.c_scale

    @staticmethod
    def _backup(path: list[Node], leaf_value: float) -> None:
        value = leaf_value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    @staticmethod
    def _principal_variation(root: Node, selected: Move, max_length: int = 12) -> tuple[str, ...]:
        line: list[str] = [str(selected)]
        node = root.children.get(selected)
        while node is not None and len(line) < max_length and node.children:
            visited = [item for item in node.children.items() if item[1].visit_count > 0]
            if not visited:
                break
            move, node = max(visited, key=lambda item: item[1].visit_count)
            line.append(str(move))
        return tuple(line)
