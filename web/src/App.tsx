import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnalysisPanel } from "./components/AnalysisPanel";
import { Board } from "./components/Board";
import { GameSetup } from "./components/GameSetup";
import { GameTools } from "./components/GameTools";
import {
  ApiError,
  createGame,
  eventSocket,
  gameAction,
  getGame,
  listModels,
  playMove,
  restartGame,
} from "./lib/api";
import { outcomeReasonLabel, outcomeTitle } from "./lib/outcome";
import type { Difficulty, GameState, HumanSide, ModelDescriptor } from "./types/api";

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "发生了未预期的错误";
}

function statusCopy(game: GameState | null): { title: string; detail: string; tone: string } {
  if (!game) return { title: "正在备枰", detail: "连接本地推理服务", tone: "quiet" };
  if (game.outcome.terminal) {
    if (game.outcome.status === "draw") {
      return { title: outcomeTitle(game.outcome), detail: outcomeReasonLabel(game.outcome), tone: "draw" };
    }
    return {
      title: outcomeTitle(game.outcome),
      detail: outcomeReasonLabel(game.outcome, "终局"),
      tone: game.outcome.winner ?? "quiet",
    };
  }
  if (game.in_check) return { title: "将军", detail: `${game.side_to_move === "red" ? "红方" : "黑方"}须立即应将`, tone: "warning" };
  if (game.ai_thinking) return { title: "AI 推演中", detail: "搜索可随时由悔棋或重开取消", tone: "thinking" };
  return {
    title: game.side_to_move === game.human_side ? "请落子" : "等待 AI",
    detail: `第 ${Math.floor(game.ply / 2) + 1} 回合 · ${game.side_to_move === "red" ? "红方" : "黑方"}行棋`,
    tone: game.side_to_move,
  };
}

export default function App() {
  const [models, setModels] = useState<ModelDescriptor[]>([]);
  const [game, setGame] = useState<GameState | null>(null);
  const [humanSide, setHumanSide] = useState<HumanSide>("red");
  const [difficulty, setDifficulty] = useState<Difficulty>("standard");
  const [modelId, setModelId] = useState("heuristic");
  const [selected, setSelected] = useState<string | null>(null);
  const [flipped, setFlipped] = useState(false);
  const [analysisEnabled, setAnalysisEnabled] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const initialized = useRef(false);

  const beginGame = useCallback(async (override?: {
    side?: HumanSide;
    strength?: Difficulty;
    model?: string;
  }) => {
    setBusy(true);
    setError(null);
    try {
      const next = await createGame({
        human_side: override?.side ?? humanSide,
        difficulty: override?.strength ?? difficulty,
        model_id: override?.model ?? modelId,
      });
      setGame(next);
      setSelected(null);
      setFlipped(next.human_side === "black");
    } catch (cause) {
      setError(`无法开局：${errorMessage(cause)}`);
    } finally {
      setBusy(false);
    }
  }, [difficulty, humanSide, modelId]);

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    void (async () => {
      try {
        const response = await listModels();
        setModels(response.models);
        const compatible = response.models.find((model) => model.compatible)?.id ?? "heuristic";
        setModelId(compatible);
        await beginGame({ model: compatible });
      } catch (cause) {
        setError(`无法连接 ChessAI 服务：${errorMessage(cause)}`);
      }
    })();
  }, [beginGame]);

  useEffect(() => {
    if (!game?.game_id) return;
    const socket = eventSocket(game.game_id);
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data) as { type: string; game?: GameState };
      if (payload.type === "snapshot" && payload.game) {
        setGame(payload.game);
        return;
      }
      if (payload.type === "thinking_started") {
        setGame((current) => current ? { ...current, ai_thinking: true } : current);
      }
      void getGame(game.game_id).then(setGame).catch(() => undefined);
    };
    return () => socket.close();
  }, [game?.game_id]);

  useEffect(() => {
    if (!game?.ai_thinking) return;
    const timer = window.setInterval(() => {
      void getGame(game.game_id).then(setGame).catch(() => undefined);
    }, 700);
    return () => window.clearInterval(timer);
  }, [game?.ai_thinking, game?.game_id]);

  useEffect(() => setSelected(null), [game?.ply]);

  const submitMove = useCallback(async (move: string) => {
    if (!game || game.ai_thinking || game.outcome.terminal) return;
    setBusy(true);
    setError(null);
    setSelected(null);
    try {
      setGame(await playMove(game.game_id, move, game.ply));
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 409) {
        setGame(await getGame(game.game_id));
        setError("局面已更新，刚才的过期落子没有被执行。");
      } else {
        setError(`落子失败：${errorMessage(cause)}`);
      }
    } finally {
      setBusy(false);
    }
  }, [game]);

  const ownMovableSquares = useMemo(() => {
    if (!game) return new Set<string>();
    return new Set(game.legal_moves.map((move) => move.slice(0, 2)));
  }, [game]);

  const handleSquare = (square: string) => {
    if (!game || busy || game.ai_thinking || game.outcome.terminal || game.side_to_move !== game.human_side) return;
    const piece = game.pieces.find((item) => item.square === square);
    if (selected && game.legal_moves.includes(`${selected}${square}`)) {
      void submitMove(`${selected}${square}`);
      return;
    }
    if (piece?.color === game.human_side && ownMovableSquares.has(square)) {
      setSelected(square);
    } else {
      setSelected(null);
    }
  };

  const runAction = async (action: "undo" | "resign") => {
    if (!game) return;
    setBusy(true);
    setError(null);
    try {
      setGame(await gameAction(game.game_id, action));
      setSelected(null);
    } catch (cause) {
      setError(`${action === "undo" ? "悔棋" : "认输"}失败：${errorMessage(cause)}`);
    } finally {
      setBusy(false);
    }
  };

  const runRestart = async (fen?: string) => {
    if (!game) return;
    setBusy(true);
    setError(null);
    try {
      const next = await restartGame(game.game_id, fen);
      setGame(next);
      setSelected(null);
    } catch (cause) {
      setError(`无法载入局面：${errorMessage(cause)}`);
    } finally {
      setBusy(false);
    }
  };

  const exportPgn = async () => {
    if (!game) return;
    try {
      const response = await fetch(`/api/v1/games/${game.game_id}/pgn`);
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `chessai-${game.game_id.slice(0, 8)}.pgn`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (cause) {
      setError(`PGN 导出失败：${errorMessage(cause)}`);
    }
  };

  const status = statusCopy(game);
  const boardDisabled = busy || !game || game.ai_thinking || game.outcome.terminal || game.side_to_move !== game.human_side;

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="brand" aria-label="墨枰中国象棋 AI">
          <span className="brand-seal">墨</span>
          <span><strong>墨枰</strong><small>GUMBEL ALPHAZERO · XIANGQI</small></span>
        </div>
        <div className="service-status" title="FastAPI 本地服务">
          <i className={game ? "online" : ""} />
          <span>{game ? "本地推理在线" : "等待服务"}</span>
        </div>
      </header>

      <main className="play-page">
          <div className="page-intro">
            <div><span className="eyebrow">宋版棋谱 × 现代训练台</span><h1>人机对弈</h1></div>
            <p>规则由参考引擎裁决，搜索结果带版本契约；每一着，都可以追溯。</p>
          </div>

          {error && <div className="notice error" role="alert"><span>!</span>{error}<button type="button" onClick={() => setError(null)} aria-label="关闭提示">×</button></div>}

          <div className="play-grid">
            <aside className="control-stack">
              <GameSetup
                humanSide={humanSide}
                difficulty={difficulty}
                modelId={modelId}
                models={models}
                busy={busy}
                onHumanSide={setHumanSide}
                onDifficulty={setDifficulty}
                onModel={setModelId}
                onStart={() => void beginGame()}
              />
              <GameTools
                game={game}
                flipped={flipped}
                busy={busy}
                onFlip={() => setFlipped((value) => !value)}
                onUndo={() => void runAction("undo")}
                onRestart={(fen) => void runRestart(fen)}
                onResign={() => void runAction("resign")}
                onExportPgn={() => void exportPgn()}
              />
            </aside>

            <section className="board-column" aria-label="对弈棋盘">
              <div className={`turn-banner tone-${status.tone}`}>
                <span className="turn-mark">{game?.side_to_move === "black" ? "黑" : "红"}</span>
                <div><strong>{status.title}</strong><small>{status.detail}</small></div>
                <span className="ply-mark">PLY {String(game?.ply ?? 0).padStart(3, "0")}</span>
              </div>
              {game ? (
                <Board
                  pieces={game.pieces}
                  legalMoves={game.legal_moves}
                  lastMove={game.last_move}
                  selected={selected}
                  flipped={flipped}
                  interactiveColor={game.human_side}
                  disabled={boardDisabled}
                  inCheck={game.in_check}
                  outcome={game.outcome}
                  onSquare={handleSquare}
                  onMove={(move) => void submitMove(move)}
                  onRestart={() => void runRestart()}
                />
              ) : (
                <div className="board-loading"><span>弈</span><p>正在展开棋盘…</p></div>
              )}
              <div className="board-footnote"><span>ICCS 坐标</span><i />固定 2086 动作词表<i />WXF 2018 规则版本</div>
            </section>

            {game ? (
              <AnalysisPanel
                analysis={game.analysis}
                thinking={game.ai_thinking}
                outcome={game.outcome}
                sideToMove={game.side_to_move}
                history={game.history}
                enabled={analysisEnabled}
                onEnabled={setAnalysisEnabled}
              />
            ) : <aside className="analysis-stack"><div className="paper-panel skeleton-panel" /></aside>}
          </div>
      </main>

      <footer className="site-footer">
        <span>CHESSAI / 0.1.0</span>
        <p>本地优先 · 无账号 · 推理接口仅监听 127.0.0.1</p>
        <span>规则 / 数据 / 模型皆有版本</span>
      </footer>
    </div>
  );
}
