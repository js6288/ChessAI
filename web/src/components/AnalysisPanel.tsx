import type { Analysis, Color, MoveRecord, Outcome } from "../types/api";

const REASON_LABELS: Record<string, string> = {
  checkmate: "将死",
  stalemate: "困毙",
  resignation: "认输",
  repetition: "重复局面",
  perpetual_check: "长将判负",
  perpetual_chase: "长捉判负",
  no_capture: "无吃子和棋",
  max_ply_limit: "达到安全回合上限",
};

function outcomeText(outcome: Outcome): string {
  if (!outcome.terminal) return "对局进行中";
  if (outcome.status === "draw") return `和棋 · ${REASON_LABELS[outcome.reason ?? ""] ?? "规则裁决"}`;
  return `${outcome.winner === "red" ? "红方" : "黑方"}胜 · ${REASON_LABELS[outcome.reason ?? ""] ?? "终局"}`;
}

interface AnalysisPanelProps {
  analysis: Analysis | null;
  thinking: boolean;
  outcome: Outcome;
  sideToMove: Color;
  history: MoveRecord[];
  enabled: boolean;
  onEnabled: (enabled: boolean) => void;
}

export function AnalysisPanel(props: AnalysisPanelProps) {
  const redPercent = props.analysis ? Math.round(props.analysis.win_probability * 100) : 50;
  return (
    <aside className="analysis-stack" aria-label="AI 分析与着法记录">
      <section className="paper-panel analysis-panel">
        <div className="panel-heading compact">
          <span className="section-number">贰</span>
          <div>
            <h2>局势推演</h2>
            <p>{props.thinking ? "树上寻路" : outcomeText(props.outcome)}</p>
          </div>
          <label className="ink-switch">
            <input
              type="checkbox"
              checked={props.enabled}
              onChange={(event) => props.onEnabled(event.target.checked)}
            />
            <span aria-hidden="true" />
            <b>分析</b>
          </label>
        </div>

        {!props.enabled ? (
          <div className="analysis-muted">
            <span>隐</span>
            <p>分析数据已收起，棋局搜索仍在后台正常进行。</p>
          </div>
        ) : (
          <>
            <div className={`thinking-band${props.thinking ? " active" : ""}`} aria-live="polite">
              <span className="thinking-orbit" aria-hidden="true"><i /><i /><i /></span>
              <div>
                <strong>{props.thinking ? "AI 正在推演" : "上一轮分析"}</strong>
                <small>
                  {props.analysis
                    ? `${props.analysis.elapsed_ms.toFixed(0)} ms · ${props.analysis.visits} 次访问`
                    : `${props.sideToMove === "red" ? "红方" : "黑方"}行棋`}
                </small>
              </div>
            </div>

            <div className="evaluation-block">
              <div className="evaluation-labels">
                <span>红方 {redPercent}%</span>
                <span>{props.analysis ? `价值 ${props.analysis.value >= 0 ? "+" : ""}${props.analysis.value.toFixed(3)}` : "尚无搜索"}</span>
              </div>
              <div className="evaluation-track" role="meter" aria-label="红方胜率估计" aria-valuenow={redPercent} aria-valuemin={0} aria-valuemax={100}>
                <span style={{ width: `${redPercent}%` }} />
              </div>
              <div className="metric-disclaimer">网络价值估计，不等同于实战棋力证据</div>
            </div>

            <div className="candidate-list">
              <div className="subhead"><span>候选着</span><span>概率 / 访问 / Q</span></div>
              {props.analysis?.candidates.length ? props.analysis.candidates.map((candidate, index) => (
                <div className="candidate-row" key={candidate.move}>
                  <b>{String(index + 1).padStart(2, "0")}</b>
                  <code>{candidate.move}</code>
                  <span className="candidate-bar"><i style={{ width: `${Math.max(4, candidate.probability * 100)}%` }} /></span>
                  <span>{Math.round(candidate.probability * 100)}% · {candidate.visits} · {candidate.q_value.toFixed(2)}</span>
                </div>
              )) : (
                <p className="empty-copy">落子后，这里会留下搜索的墨迹。</p>
              )}
            </div>

            {props.analysis?.principal_variation.length ? (
              <div className="pv-line">
                <span>主变化</span>
                <code>{props.analysis.principal_variation.join("  ")}</code>
              </div>
            ) : null}
          </>
        )}
      </section>

      <section className="paper-panel history-panel">
        <div className="subhead history-title"><span>着法簿</span><span>{props.history.length} 手</span></div>
        <div className="move-scroll" tabIndex={0}>
          {props.history.length ? Array.from({ length: Math.ceil(props.history.length / 2) }, (_, row) => {
            const red = props.history[row * 2];
            const black = props.history[row * 2 + 1];
            return (
              <div className="move-row" key={row}>
                <span>{String(row + 1).padStart(2, "0")}</span>
                <b className="red-move" title={red.move}>{red.notation}</b>
                <b title={black?.move}>{black?.notation ?? "—"}</b>
              </div>
            );
          }) : <p className="empty-copy">棋局未启，着法簿尚空。</p>}
        </div>
      </section>
    </aside>
  );
}
