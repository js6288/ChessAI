import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Board } from "../src/components/Board";
import type { Piece } from "../src/types/api";

const pieces: Piece[] = [
  { square: "h2", piece: "C", kind: "C", color: "red", glyph: "炮" },
  { square: "e9", piece: "k", kind: "K", color: "black", glyph: "将" },
];

describe("Board", () => {
  it("renders accessible pieces and reports square activation", () => {
    const onSquare = vi.fn();
    render(
      <Board
        pieces={pieces}
        legalMoves={["h2e2"]}
        lastMove={null}
        selected={null}
        flipped={false}
        interactiveColor="red"
        onSquare={onSquare}
        onMove={vi.fn()}
      />,
    );

    expect(screen.getByLabelText(/红方炮/)).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("piece-h2"));
    expect(onSquare).toHaveBeenCalledWith("h2");
  });

  it("supports keyboard selection and exposes legal target", () => {
    const onSquare = vi.fn();
    render(
      <Board
        pieces={pieces}
        legalMoves={["h2e2"]}
        lastMove="a0a1"
        selected="h2"
        flipped
        interactiveColor="red"
        onSquare={onSquare}
        onMove={vi.fn()}
      />,
    );

    fireEvent.keyDown(screen.getByTestId("piece-h2"), { key: "Enter" });
    fireEvent.click(screen.getByTestId("square-e2"));
    expect(onSquare).toHaveBeenNthCalledWith(1, "h2");
    expect(onSquare).toHaveBeenNthCalledWith(2, "e2");
  });
});
