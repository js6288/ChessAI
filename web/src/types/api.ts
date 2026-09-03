export type Color = "red" | "black";
export type HumanSide = Color | "random";
export type Difficulty = "beginner" | "standard" | "advanced" | "expert";

export interface Piece {
  square: string;
  piece: string;
  kind: "K" | "A" | "B" | "N" | "R" | "C" | "P";
  color: Color;
  glyph: string;
}

export interface MoveRecord {
  ply: number;
  move: string;
  notation: string;
  color: Color;
}

export interface Outcome {
  status: "ongoing" | "red_win" | "black_win" | "draw";
  winner: Color | null;
  reason: string | null;
  terminal: boolean;
}

export interface Candidate {
  move: string;
  probability: number;
  visits: number;
  q_value: number;
}

export interface Analysis {
  best_move: string;
  value: number;
  win_probability: number;
  visits: number;
  elapsed_ms: number;
  principal_variation: string[];
  candidates: Candidate[];
}

export interface GameState {
  game_id: string;
  fen: string;
  initial_fen: string;
  side_to_move: Color;
  human_side: Color;
  difficulty: Difficulty;
  model_id: string;
  ply: number;
  pieces: Piece[];
  legal_moves: string[];
  last_move: string | null;
  history: MoveRecord[];
  outcome: Outcome;
  ai_thinking: boolean;
  in_check: boolean;
  analysis: Analysis | null;
}

export interface ModelDescriptor {
  id: string;
  name: string;
  kind: string;
  compatible: boolean;
  error: string | null;
  compatibility: Record<string, unknown>;
}
