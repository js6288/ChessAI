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

  it("keeps the board-position transform while a piece is held for dragging", () => {
    const onMove = vi.fn();
    render(
      <Board
        pieces={pieces}
        legalMoves={["h2e2"]}
        lastMove={null}
        selected={null}
        flipped={false}
        interactiveColor="red"
        onSquare={vi.fn()}
        onMove={onMove}
      />,
    );

    const piece = screen.getByTestId("piece-h2");
    const face = screen.getByTestId("piece-face-h2");
    expect(piece).toHaveAttribute("transform", "translate(620 620)");
    expect(face).toHaveAttribute("transform", "translate(0 0)");

    fireEvent.pointerDown(piece, { pointerId: 1 });

    expect(piece).toHaveClass("is-selected");
    expect(piece).toHaveAttribute("transform", "translate(620 620)");
    expect(face).toHaveAttribute("transform", "translate(0 -3)");

    fireEvent.pointerUp(screen.getByTestId("square-e2"), { pointerId: 1 });
    expect(onMove).toHaveBeenCalledOnce();
    expect(onMove).toHaveBeenCalledWith("h2e2");
  });
});
