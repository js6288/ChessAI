"""Coordinates and moves in ICCS notation.

Coordinates are stored from Red's point of view: file ``a`` is 0, file ``i``
is 8, Red's home rank is 0, and Black's home rank is 9.
"""

from __future__ import annotations

from dataclasses import dataclass

FILES = "abcdefghi"
RANKS = "0123456789"


@dataclass(frozen=True, slots=True, order=True)
class Square:
    file: int
    rank: int

    def __post_init__(self) -> None:
        if not (0 <= self.file < 9 and 0 <= self.rank < 10):
            raise ValueError(f"square outside Xiangqi board: ({self.file}, {self.rank})")

    @classmethod
    def from_iccs(cls, value: str) -> Square:
        if len(value) != 2 or value[0] not in FILES or value[1] not in RANKS:
            raise ValueError(f"invalid ICCS square: {value!r}")
        return cls(FILES.index(value[0]), int(value[1]))

    @property
    def index(self) -> int:
        return self.rank * 9 + self.file

    @classmethod
    def from_index(cls, index: int) -> Square:
        if not 0 <= index < 90:
            raise ValueError(f"square index outside board: {index}")
        rank, file = divmod(index, 9)
        return cls(file, rank)

    def rotate(self) -> Square:
        """Rotate the board 180 degrees."""

        return Square(8 - self.file, 9 - self.rank)

    def __str__(self) -> str:
        return f"{FILES[self.file]}{self.rank}"


@dataclass(frozen=True, slots=True, order=True)
class Move:
    from_square: Square
    to_square: Square

    def __post_init__(self) -> None:
        if self.from_square == self.to_square:
            raise ValueError("a move must change squares")

    @classmethod
    def from_iccs(cls, value: str) -> Move:
        normalized = value.strip().lower()
        if len(normalized) != 4:
            raise ValueError(f"invalid ICCS move: {value!r}")
        return cls(Square.from_iccs(normalized[:2]), Square.from_iccs(normalized[2:]))

    def rotate(self) -> Move:
        return Move(self.from_square.rotate(), self.to_square.rotate())

    def __str__(self) -> str:
        return f"{self.from_square}{self.to_square}"
