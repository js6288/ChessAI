import type { ModelDescriptor } from "../types/api";

/** Prefer an installed policy/value checkpoint while retaining the heuristic fallback. */
export function preferredModelId(models: ModelDescriptor[]): string {
  return models.find((model) => model.compatible && model.kind === "policy-value")?.id
    ?? models.find((model) => model.compatible && model.id !== "heuristic")?.id
    ?? models.find((model) => model.compatible)?.id
    ?? "heuristic";
}
