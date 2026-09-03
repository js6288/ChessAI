import pytest

from chessai.engine.move import Move, Square


def test_square_and_move_iccs_round_trip() -> None:
    square = Square.from_iccs("h2")
    assert square == Square(7, 2)
    assert square.index == 25
    assert str(Square.from_index(25)) == "h2"
    assert str(square.rotate()) == "b7"

    move = Move.from_iccs("h2e2")
    assert str(move) == "h2e2"
    assert str(move.rotate()) == "b7e7"


@pytest.mark.parametrize("value", ["", "a", "j0", "a10", "z9"])
def test_invalid_square_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        Square.from_iccs(value)


def test_null_move_rejected() -> None:
    with pytest.raises(ValueError):
        Move.from_iccs("a0a0")
