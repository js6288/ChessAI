import type { Outcome } from "../types/api";

export const OUTCOME_REASON_LABELS: Record<string, string> = {
  checkmate: "将死",
  stalemate: "困毙",
  resignation: "认输",
  repetition: "重复局面",
  perpetual_check: "长将判负",
  perpetual_chase: "长捉判负",
  no_capture: "无吃子和棋",
  max_ply_limit: "达到安全回合上限",
};

export function outcomeReasonLabel(outcome: Outcome, fallback = "规则裁决"): string {
  return OUTCOME_REASON_LABELS[outcome.reason ?? ""] ?? fallback;
}

export function outcomeTitle(outcome: Outcome): string {
  if (!outcome.terminal) return "对局进行中";
  if (outcome.status === "draw") return "和棋";
  return `${outcome.winner === "red" ? "红方" : "黑方"}胜`;
}
