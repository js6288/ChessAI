"""FastAPI application factory for the local Xiangqi workbench."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from chessai import __version__
from chessai.engine import Move
from chessai.server.models import ModelRegistry
from chessai.server.schemas import CreateGameRequest, MoveRequest, RestartRequest
from chessai.server.sessions import SessionManager


def create_app(
    *,
    model_dir: str | Path | None = None,
) -> FastAPI:
    app = FastAPI(title="ChessAI Xiangqi API", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    resolved_model_dir: str | Path = (
        model_dir if model_dir is not None else os.getenv("CHESSAI_MODEL_DIR") or "checkpoints"
    )
    registry = ModelRegistry(resolved_model_dir)
    manager = SessionManager(registry)
    app.state.registry = registry
    app.state.sessions = manager

    @app.get("/api/v1/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "version": __version__, "sessions": len(manager.sessions)}

    @app.get("/api/v1/models")
    async def models() -> dict[str, object]:
        registry.refresh()
        return {"models": registry.list()}

    @app.post("/api/v1/games", status_code=201)
    async def create_game(request: CreateGameRequest) -> dict[str, object]:
        try:
            session = await manager.create(**request.model_dump())
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return manager.serialize(session)

    @app.get("/api/v1/games/{game_id}")
    async def get_game(game_id: str) -> dict[str, object]:
        try:
            return manager.serialize(manager.get(game_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/games/{game_id}/moves")
    async def play_move(game_id: str, request: MoveRequest) -> dict[str, object]:
        try:
            session = manager.get(game_id)
            await manager.play_human_move(
                session, Move.from_iccs(request.move), request.expected_ply
            )
            return manager.serialize(session)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/games/{game_id}/undo")
    async def undo(game_id: str) -> dict[str, object]:
        try:
            session = manager.get(game_id)
            await manager.undo(session)
            return manager.serialize(session)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/games/{game_id}/restart")
    async def restart(game_id: str, request: RestartRequest | None = None) -> dict[str, object]:
        try:
            session = manager.get(game_id)
            await manager.restart(session, request.initial_fen if request else None)
            return manager.serialize(session)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/games/{game_id}/resign")
    async def resign(game_id: str) -> dict[str, object]:
        try:
            session = manager.get(game_id)
            await manager.resign(session)
            return manager.serialize(session)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/games/{game_id}/pgn", response_class=PlainTextResponse)
    async def game_pgn(game_id: str) -> str:
        try:
            return manager.pgn(manager.get(game_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.websocket("/api/v1/games/{game_id}/events")
    async def events(websocket: WebSocket, game_id: str) -> None:
        try:
            session = manager.get(game_id)
        except KeyError:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=64)
        session.subscribers.add(queue)
        await websocket.send_json({"type": "snapshot", "game": manager.serialize(session)})
        try:
            while True:
                await websocket.send_json(await queue.get())
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        finally:
            session.subscribers.discard(queue)

    project_root = Path(__file__).resolve().parents[3]
    web_dist = project_root / "web" / "dist"
    if web_dist.is_dir():
        assets = web_dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str) -> FileResponse:
            if path == "api" or path.startswith("api/"):
                raise HTTPException(status_code=404, detail="API route not found")
            candidate = (web_dist / path).resolve()
            if candidate.is_file() and web_dist.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(web_dist / "index.html")

    return app


app = create_app()
