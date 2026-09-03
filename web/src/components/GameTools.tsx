import { useEffect, useState } from "react";
import type { GameState } from "../types/api";

interface GameToolsProps {
  game: GameState | null;
  flipped: boolean;
  busy: boolean;
  onFlip: () => void;
  onUndo: () => void;
  onRestart: (fen?: string) => void;
  onResign: () => void;
  onExportPgn: () => void;
}

export function GameTools(props: GameToolsProps) {
  const [fen, setFen] = useState("");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (props.game?.fen) setFen(props.game.fen);
  }, [props.game?.fen]);

  const copyFen = async () => {
    if (!props.game) return;
    await navigator.clipboard.writeText(props.game.fen);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  };

  return (
    <section className="paper-panel tools-panel" aria-labelledby="tools-title">
      <div className="panel-heading compact">
        <span className="section-number">叁</span>
        <div>
          <h2 id="tools-title">局面工具</h2>
          <p>复盘 · 传谱 · 校验</p>
        </div>
      </div>

      <div className="tool-grid">
        <button type="button" onClick={props.onUndo} disabled={!props.game?.history.length || props.busy}>
          <span aria-hidden="true">↶</span>悔棋
        </button>
        <button type="button" onClick={props.onFlip}>
          <span aria-hidden="true">⇅</span>{props.flipped ? "红在下" : "翻转"}
        </button>
        <button type="button" onClick={() => props.onRestart()} disabled={!props.game}>
          <span aria-hidden="true">↻</span>重开
        </button>
        <button type="button" className="danger-quiet" onClick={props.onResign} disabled={!props.game || props.game.outcome.terminal}>
          <span aria-hidden="true">×</span>认输
        </button>
      </div>

      <label className="fen-field">
        <span>FEN 局面</span>
        <textarea
          rows={3}
          value={fen}
          spellCheck={false}
          onChange={(event) => setFen(event.target.value)}
          placeholder="粘贴中国象棋 FEN…"
        />
      </label>
      <div className="fen-actions">
        <button type="button" onClick={copyFen} disabled={!props.game}>{copied ? "已复制" : "复制当前"}</button>
        <button type="button" onClick={() => props.onRestart(fen.trim())} disabled={!fen.trim()}>载入此局</button>
        <button type="button" onClick={props.onExportPgn} disabled={!props.game}>导出 PGN</button>
      </div>
    </section>
  );
}
