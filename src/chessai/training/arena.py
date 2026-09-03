"""Lightweight paired-opening evaluation for the playable product profile."""

from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Protocol

from chessai.ai.search import GumbelSearch, HeuristicEvaluator
from chessai.engine import Color, GameState, Move
from chessai.engine.board import PIECES


class Player(Protocol):
    name: str

    def select_move(self, state: GameState) -> Move: ...


class RandomPlayer:
    def __init__(self, seed: int = 0) -> None:
        self.name = "random"
        self._rng = random.Random(seed)

    def select_move(self, state: GameState) -> Move:
        return self._rng.choice(state.legal_moves)


class SearchPlayer:
    def __init__(self, search: GumbelSearch, *, name: str = "gumbel-search") -> None:
        self.name = name
        self.search = search

    def select_move(self, state: GameState) -> Move:
        return self.search.search(state).best_move


MATERIAL = {"K": 0.0, "R": 9.0, "C": 4.5, "N": 4.0, "B": 2.0, "A": 2.0, "P": 1.0}


class AlphaBetaPlayer:
    def __init__(self, depth: int = 3) -> None:
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.depth = depth
        self.name = f"material-alpha-beta-d{depth}"

    @staticmethod
    def _evaluate(state: GameState) -> float:
        outcome = state.outcome()
        if outcome.terminal:
            if outcome.winner is None:
                return 0.0
            return 10_000.0 if outcome.winner is state.side_to_move else -10_000.0
        score = 0.0
        for piece in state.board:
            if piece not in PIECES:
                continue
            value = MATERIAL[piece.upper()]
            own = (piece.isupper() and state.side_to_move is Color.RED) or (
                piece.islower() and state.side_to_move is Color.BLACK
            )
            score += value if own else -value
        return score

    def _negamax(self, state: GameState, depth: int, alpha: float, beta: float) -> float:
        if depth == 0 or state.outcome().terminal:
            return self._evaluate(state)
        best = -math.inf
        moves = sorted(
            state.legal_moves,
            key=lambda move: MATERIAL.get((state.piece_at(move.to_square) or "K").upper(), 0.0),
            reverse=True,
        )
        for move in moves:
            value = -self._negamax(state.apply(move), depth - 1, -beta, -alpha)
            best = max(best, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best

    def select_move(self, state: GameState) -> Move:
        best_move = state.legal_moves[0]
        best_value = -math.inf
        for move in state.legal_moves:
            value = -self._negamax(state.apply(move), self.depth - 1, -math.inf, math.inf)
            if value > best_value:
                best_move, best_value = move, value
        return best_move


@dataclass(frozen=True, slots=True)
class GameResult:
    red: str
    black: str
    winner: str | None
    status: str
    reason: str | None
    plies: int


@dataclass(frozen=True, slots=True)
class ArenaResult:
    candidate: str
    opponent: str
    wins: int
    draws: int
    losses: int
    games: int
    score_rate: float
    elapsed_seconds: float


def quick_opening_fens(count: int = 10, *, seed: int = 20260902) -> list[str]:
    """Build a small deterministic set of legal, quiet opening positions."""

    if count <= 0:
        raise ValueError("opening count must be positive")
    openings: list[str] = []
    seen: set[str] = set()
    opening_index = 0
    while len(openings) < count:
        state = GameState.initial()
        for ply in range(6 + opening_index % 5):
            moves = sorted(state.legal_moves, key=str)
            quiet = [move for move in moves if state.piece_at(move.to_square) is None]
            choices = quiet or moves
            material = f"{seed}:{opening_index}:{ply}:{state.to_fen()}".encode()
            choice = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % len(choices)
            state = state.apply(choices[choice])
            if state.outcome().terminal:
                break
        fen = state.to_fen()
        if not state.outcome().terminal and fen not in seen:
            seen.add(fen)
            openings.append(fen)
        opening_index += 1
        if opening_index > 1_000:  # pragma: no cover - defensive deterministic guard
            raise RuntimeError("failed to construct quick opening positions")
    return openings


def play_game(
    red: Player,
    black: Player,
    *,
    initial_fen: str | None = None,
    max_ply: int = 300,
) -> GameResult:
    state = (
        GameState.from_fen(initial_fen, max_ply=max_ply)
        if initial_fen
        else GameState.initial(max_ply=max_ply)
    )
    while not state.outcome().terminal:
        player = red if state.side_to_move is Color.RED else black
        move = player.select_move(state)
        if move not in state.legal_moves:
            raise ValueError(f"player {player.name} returned illegal move {move}")
        state = state.apply(move)
    outcome = state.outcome()
    return GameResult(
        red=red.name,
        black=black.name,
        winner=outcome.winner.value if outcome.winner else None,
        status=outcome.status.value,
        reason=outcome.reason,
        plies=state.ply,
    )


def run_arena(
    candidate: Player,
    opponent: Player,
    *,
    games: int = 20,
    opening_fens: list[str] | None = None,
    max_ply: int = 300,
) -> tuple[ArenaResult, list[GameResult]]:
    if games <= 0 or games % 2:
        raise ValueError("arena games must be a positive even number")
    openings = opening_fens or quick_opening_fens(min(10, games // 2))
    results: list[GameResult] = []
    wins = draws = losses = 0
    started = time.perf_counter()
    for game_index in range(games):
        opening = openings[(game_index // 2) % len(openings)]
        candidate_red = game_index % 2 == 0
        red, black = (candidate, opponent) if candidate_red else (opponent, candidate)
        game = play_game(red, black, initial_fen=opening, max_ply=max_ply)
        results.append(game)
        if game.winner is None:
            draws += 1
        elif (game.winner == Color.RED.value) == candidate_red:
            wins += 1
        else:
            losses += 1
    return (
        ArenaResult(
            candidate=candidate.name,
            opponent=opponent.name,
            wins=wins,
            draws=draws,
            losses=losses,
            games=games,
            score_rate=(wins + 0.5 * draws) / games,
            elapsed_seconds=time.perf_counter() - started,
        ),
        results,
    )


def default_heuristic_player(simulations: int = 8, seed: int = 0) -> SearchPlayer:
    return SearchPlayer(
        GumbelSearch(HeuristicEvaluator(), simulations=simulations, seed=seed),
        name=f"heuristic-gumbel-n{simulations}",
    )


def arena_result_dict(result: ArenaResult) -> dict[str, object]:
    return asdict(result)
