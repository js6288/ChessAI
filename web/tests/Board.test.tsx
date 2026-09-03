import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Board } from "../src/components/Board";
import type { Piece } from "../src/types/api";

const pieces: Piece[] = [
  { square: "h2", piece: "C", kind: "C", color: "red", glyph: "炮" },
  { square: "e9", piece: "k", kind: "K", color: "black", glyph: "将" },
];

const ongoing = {
  status: "ongoing" as const,
  winner: null,
  reason: null,
  terminal: false,
};

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
        outcome={ongoing}
        onSquare={onSquare}
        onMove={vi.fn()}
        onRestart={vi.fn()}
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
        outcome={ongoing}
        onSquare={onSquare}
        onMove={vi.fn()}
        onRestart={vi.fn()}
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
        outcome={ongoing}
        onSquare={vi.fn()}
        onMove={onMove}
        onRestart={vi.fn()}
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

  it("presents a readable animated settlement and lets the player review the board", () => {
    const onRestart = vi.fn();
    render(
      <Board
        pieces={pieces}
        legalMoves={[]}
        lastMove="h2e2"
        selected={null}
        flipped={false}
        interactiveColor="red"
        disabled
        outcome={{ status: "red_win", winner: "red", reason: "checkmate", terminal: true }}
        onSquare={vi.fn()}
        onMove={vi.fn()}
        onRestart={onRestart}
      />,
    );

    expect(screen.getByRole("dialog", { name: "红方胜" })).toBeInTheDocument();
    expect(screen.getByText("将死")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "查看棋盘" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "查看结算 · 红方胜" }));
    fireEvent.click(screen.getByRole("button", { name: "再开一局" }));

    expect(onRestart).toHaveBeenCalledOnce();
  });
});
