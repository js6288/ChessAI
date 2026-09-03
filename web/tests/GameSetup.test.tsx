import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { GameSetup } from "../src/components/GameSetup";

describe("GameSetup", () => {
  it("changes side and difficulty and starts a game", async () => {
    const user = userEvent.setup();
    const onHumanSide = vi.fn();
    const onDifficulty = vi.fn();
    const onStart = vi.fn();
    render(
      <GameSetup
        humanSide="red"
        difficulty="standard"
        modelId="heuristic"
        models={[{
          id: "heuristic",
          name: "墨衡 · 启发式演示",
          kind: "heuristic",
          compatible: true,
          error: null,
          compatibility: {},
        }]}
        busy={false}
        onHumanSide={onHumanSide}
        onDifficulty={onDifficulty}
        onModel={vi.fn()}
        onStart={onStart}
      />,
    );

    await user.click(screen.getByRole("button", { name: "执黑" }));
    await user.click(screen.getByRole("button", { name: /试锋/ }));
    await user.click(screen.getByRole("button", { name: /另开一局/ }));
    expect(onHumanSide).toHaveBeenCalledWith("black");
    expect(onDifficulty).toHaveBeenCalledWith("beginner");
    expect(onStart).toHaveBeenCalledOnce();
  });
});
