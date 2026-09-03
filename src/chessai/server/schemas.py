"""Versioned request models for the local HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

HumanSide = Literal["red", "black", "random"]
Difficulty = Literal["beginner", "standard", "advanced", "expert"]


class CreateGameRequest(BaseModel):
    human_side: HumanSide = "red"
    difficulty: Difficulty = "standard"
    model_id: str = "heuristic"
    initial_fen: str | None = None


class MoveRequest(BaseModel):
    move: str = Field(min_length=4, max_length=4)
    expected_ply: int = Field(ge=0)

    @field_validator("move")
    @classmethod
    def normalize_move(cls, value: str) -> str:
        return value.lower()


class RestartRequest(BaseModel):
    initial_fen: str | None = None
