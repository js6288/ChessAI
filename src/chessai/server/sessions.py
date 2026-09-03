"""Cancellable, stale-write-safe in-memory game sessions."""

from __future__ import annotations

import asyncio
import secrets
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from chessai.ai.search import GumbelSearch, SearchResult
from chessai.engine import Color, GameState, Move, Square
from chessai.engine.notation import BLACK_GLYPHS, RED_GLYPHS, export_pgn, move_to_chinese
from chessai.server.models import ModelRegistry

DIFFICULTY_SIMULATIONS = {
    "beginner": 8,
    "standard": 32,
    "advanced": 128,
    "expert": 256,
}


@dataclass(slots=True)
class GameSession:
    id: str
    human_side: Color
    difficulty: str
    model_id: str
    initial_fen: str
    states: list[GameState]
    moves: list[Move] = field(default_factory=list)
    notations: list[str] = field(default_factory=list)
    resigned_winner: Color | None = None
    generation: int = 0
    ai_task: asyncio.Task[None] | None = None
    last_analysis: dict[str, Any] | None = None
    subscribers: set[asyncio.Queue[dict[str, Any]]] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def state(self) -> GameState:
        return self.states[-1]

    @property
    def ai_side(self) -> Color:
        return self.human_side.opponent

    @property
    def terminal(self) -> bool:
        return self.resigned_winner is not None or self.state.outcome().terminal

    @property
    def ai_thinking(self) -> bool:
        return self.ai_task is not None and not self.ai_task.done()


class SessionManager:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.sessions: dict[str, GameSession] = {}

    async def create(
        self,
        *,
        human_side: str,
        difficulty: str,
        model_id: str,
        initial_fen: str | None,
    ) -> GameSession:
        if difficulty not in DIFFICULTY_SIMULATIONS:
            raise ValueError(f"unknown difficulty: {difficulty}")
        self.registry.evaluator(model_id)
        side = (
            Color(secrets.choice(("red", "black"))) if human_side == "random" else Color(human_side)
        )
        state = GameState.from_fen(initial_fen) if initial_fen else GameState.initial()
        session = GameSession(
            id=uuid4().hex,
            human_side=side,
            difficulty=difficulty,
            model_id=model_id,
            initial_fen=state.to_fen(),
            states=[state],
        )
        self.sessions[session.id] = session
        if state.side_to_move is session.ai_side:
            self.schedule_ai(session)
        return session

    def get(self, game_id: str) -> GameSession:
        try:
            return self.sessions[game_id]
        except KeyError as exc:
            raise KeyError(f"unknown game: {game_id}") from exc

    async def play_human_move(self, session: GameSession, move: Move, expected_ply: int) -> None:
        async with session.lock:
            if session.terminal:
                raise ValueError("game is already over")
            if expected_ply != session.state.ply:
                raise RuntimeError(
                    f"stale ply: request={expected_ply}, current={session.state.ply}"
                )
            if session.state.side_to_move is not session.human_side:
                raise ValueError("it is not the human side's turn")
            if move not in session.state.legal_moves:
                raise ValueError(f"illegal move: {move}")
            notation = move_to_chinese(session.state, move)
            session.moves.append(move)
            session.notations.append(notation)
            session.states.append(session.state.apply(move))
            session.last_analysis = None
            await self.broadcast(
                session, "move_played", actor="human", move=str(move), notation=notation
            )
        if not session.terminal:
            self.schedule_ai(session)

    def schedule_ai(self, session: GameSession) -> None:
        if (
            session.terminal
            or session.state.side_to_move is not session.ai_side
            or session.ai_thinking
        ):
            return
        generation = session.generation
        expected_ply = session.state.ply
        session.ai_task = asyncio.create_task(self._run_ai(session, generation, expected_ply))

    async def _run_ai(self, session: GameSession, generation: int, expected_ply: int) -> None:
        await self.broadcast(
            session,
            "thinking_started",
            ply=expected_ply,
            simulations=DIFFICULTY_SIMULATIONS[session.difficulty],
        )
        try:
            evaluator = self.registry.evaluator(session.model_id)
            search = GumbelSearch(
                evaluator,
                simulations=DIFFICULTY_SIMULATIONS[session.difficulty],
                max_considered_actions=min(16, DIFFICULTY_SIMULATIONS[session.difficulty]),
                seed=expected_ply + generation * 10_000,
            )
            state = session.state
            result = await asyncio.to_thread(search.search, state)
            async with session.lock:
                if (
                    generation != session.generation
                    or expected_ply != session.state.ply
                    or session.terminal
                ):
                    await self.broadcast(session, "search_cancelled", ply=expected_ply)
                    return
                notation = move_to_chinese(session.state, result.best_move)
                session.last_analysis = _analysis_payload(result)
                session.moves.append(result.best_move)
                session.notations.append(notation)
                session.states.append(session.state.apply(result.best_move))
                await self.broadcast(session, "analysis_update", **session.last_analysis)
                await self.broadcast(
                    session,
                    "move_played",
                    actor="ai",
                    move=str(result.best_move),
                    notation=notation,
                )
                if session.terminal:
                    await self.broadcast(
                        session, "game_over", outcome=self.outcome_payload(session)
                    )
        except asyncio.CancelledError:
            await self.broadcast(session, "search_cancelled", ply=expected_ply)
            raise
        except Exception as exc:  # surfaced to the client; session remains recoverable
            await self.broadcast(session, "error", message=str(exc))

    async def cancel_ai(self, session: GameSession) -> None:
        session.generation += 1
        task = session.ai_task
        session.ai_task = None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def undo(self, session: GameSession) -> None:
        await self.cancel_ai(session)
        async with session.lock:
            if not session.moves:
                raise ValueError("no move to undo")
            remove = (
                2
                if session.state.side_to_move is session.human_side and len(session.moves) >= 2
                else 1
            )
            del session.moves[-remove:]
            del session.notations[-remove:]
            del session.states[-remove:]
            session.resigned_winner = None
            session.last_analysis = None
            await self.broadcast(session, "move_played", actor="system", move=None, notation="悔棋")
        if session.state.side_to_move is session.ai_side:
            self.schedule_ai(session)

    async def restart(self, session: GameSession, initial_fen: str | None = None) -> None:
        await self.cancel_ai(session)
        async with session.lock:
            state = GameState.from_fen(initial_fen or session.initial_fen)
            session.initial_fen = state.to_fen()
            session.states = [state]
            session.moves = []
            session.notations = []
            session.resigned_winner = None
            session.last_analysis = None
            await self.broadcast(
                session, "move_played", actor="system", move=None, notation="重新开局"
            )
        if session.state.side_to_move is session.ai_side:
            self.schedule_ai(session)

    async def resign(self, session: GameSession) -> None:
        await self.cancel_ai(session)
        async with session.lock:
            if session.terminal:
                raise ValueError("game is already over")
            session.resigned_winner = session.ai_side
            await self.broadcast(session, "game_over", outcome=self.outcome_payload(session))

    async def broadcast(self, session: GameSession, event: str, **payload: Any) -> None:
        message = {"type": event, "game_id": session.id, "ply": session.state.ply, **payload}
        for queue in tuple(session.subscribers):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                session.subscribers.discard(queue)

    def outcome_payload(self, session: GameSession) -> dict[str, Any]:
        if session.resigned_winner is not None:
            winner = session.resigned_winner
            return {
                "status": "red_win" if winner is Color.RED else "black_win",
                "winner": winner.value,
                "reason": "resignation",
                "terminal": True,
            }
        outcome = session.state.outcome()
        return {
            "status": outcome.status.value,
            "winner": outcome.winner.value if outcome.winner else None,
            "reason": outcome.reason,
            "terminal": outcome.terminal,
        }

    def serialize(self, session: GameSession) -> dict[str, Any]:
        state = session.state
        pieces = []
        for index, piece in enumerate(state.board):
            if piece == ".":
                continue
            square = Square.from_index(index)
            color = Color.RED if piece.isupper() else Color.BLACK
            glyph = (RED_GLYPHS if color is Color.RED else BLACK_GLYPHS)[piece.upper()]
            pieces.append(
                {
                    "square": str(square),
                    "piece": piece,
                    "kind": piece.upper(),
                    "color": color.value,
                    "glyph": glyph,
                }
            )
        history = [
            {
                "ply": index + 1,
                "move": str(move),
                "notation": session.notations[index],
                "color": session.states[index].side_to_move.value,
            }
            for index, move in enumerate(session.moves)
        ]
        return {
            "game_id": session.id,
            "fen": state.to_fen(),
            "initial_fen": session.initial_fen,
            "side_to_move": state.side_to_move.value,
            "human_side": session.human_side.value,
            "difficulty": session.difficulty,
            "model_id": session.model_id,
            "ply": state.ply,
            "pieces": pieces,
            "legal_moves": state.legal_move_strings() if not session.terminal else (),
            "last_move": str(session.moves[-1]) if session.moves else None,
            "history": history,
            "outcome": self.outcome_payload(session),
            "in_check": state.is_in_check() if not session.terminal else False,
            "ai_thinking": session.ai_thinking,
            "analysis": session.last_analysis,
        }

    def pgn(self, session: GameSession) -> str:
        outcome = self.outcome_payload(session)
        result = "*"
        if outcome["status"] == "red_win":
            result = "1-0"
        elif outcome["status"] == "black_win":
            result = "0-1"
        elif outcome["status"] == "draw":
            result = "1/2-1/2"
        return export_pgn(session.initial_fen, session.moves, result=result)


def _analysis_payload(result: SearchResult) -> dict[str, Any]:
    return {
        "best_move": str(result.best_move),
        "value": result.value,
        "win_probability": (result.value + 1.0) / 2.0,
        "visits": result.visits,
        "elapsed_ms": result.elapsed_ms,
        "principal_variation": result.principal_variation,
        "candidates": [
            {
                "move": item.move,
                "probability": item.probability,
                "visits": item.visits,
                "q_value": item.q_value,
            }
            for item in result.candidates[:5]
        ],
    }
