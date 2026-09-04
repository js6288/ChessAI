import { describe, expect, it } from "vitest";
import { preferredModelId } from "../src/lib/models";
import type { ModelDescriptor } from "../src/types/api";

function model(
  id: string,
  kind: string,
  compatible = true,
): ModelDescriptor {
  return {
    id,
    name: id,
    kind,
    compatible,
    error: compatible ? null : "incompatible",
    compatibility: {},
  };
}

describe("preferredModelId", () => {
  it("prefers an installed policy/value checkpoint over the heuristic", () => {
    expect(preferredModelId([
      model("heuristic", "heuristic"),
      model("chessai-best", "policy-value"),
    ])).toBe("chessai-best");
  });

  it("falls back to the compatible heuristic when checkpoints are unavailable", () => {
    expect(preferredModelId([
      model("heuristic", "heuristic"),
      model("broken", "policy-value", false),
    ])).toBe("heuristic");
  });
});
