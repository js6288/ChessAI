import type {
  Difficulty,
  GameState,
  HumanSide,
  ModelDescriptor,
} from "../types/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: init?.body
      ? { "Content-Type": "application/json", ...init.headers }
      : init?.headers,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      message = payload.detail ?? message;
    } catch {
      // The original HTTP status remains useful when the body is not JSON.
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

export function listModels(): Promise<{ models: ModelDescriptor[] }> {
  return request("/api/v1/models");
}

export function createGame(input: {
  human_side: HumanSide;
  difficulty: Difficulty;
  model_id: string;
  initial_fen?: string;
}): Promise<GameState> {
  return request("/api/v1/games", { method: "POST", body: JSON.stringify(input) });
}

export function getGame(gameId: string): Promise<GameState> {
  return request(`/api/v1/games/${gameId}`);
}

export function playMove(gameId: string, move: string, expectedPly: number): Promise<GameState> {
  return request(`/api/v1/games/${gameId}/moves`, {
    method: "POST",
    body: JSON.stringify({ move, expected_ply: expectedPly }),
  });
}

export function gameAction(
  gameId: string,
  action: "undo" | "resign",
): Promise<GameState> {
  return request(`/api/v1/games/${gameId}/${action}`, { method: "POST" });
}

export function restartGame(gameId: string, initialFen?: string): Promise<GameState> {
  return request(`/api/v1/games/${gameId}/restart`, {
    method: "POST",
    body: JSON.stringify(initialFen ? { initial_fen: initialFen } : {}),
  });
}

export function eventSocket(gameId: string): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return new WebSocket(`${protocol}//${window.location.host}/api/v1/games/${gameId}/events`);
}
