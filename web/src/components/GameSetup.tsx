import type { Difficulty, HumanSide, ModelDescriptor } from "../types/api";

const DIFFICULTIES: Array<{ id: Difficulty; label: string; sub: string; sims: number }> = [
  { id: "beginner", label: "试锋", sub: "轻快入门", sims: 8 },
  { id: "standard", label: "砺局", sub: "日常对弈", sims: 32 },
  { id: "advanced", label: "深谋", sub: "认真思考", sims: 128 },
  { id: "expert", label: "穷理", sub: "深度推演", sims: 256 },
];

interface GameSetupProps {
  humanSide: HumanSide;
  difficulty: Difficulty;
  modelId: string;
  models: ModelDescriptor[];
  busy: boolean;
  onHumanSide: (side: HumanSide) => void;
  onDifficulty: (difficulty: Difficulty) => void;
  onModel: (model: string) => void;
  onStart: () => void;
}

export function GameSetup(props: GameSetupProps) {
  return (
    <section className="paper-panel setup-panel" aria-labelledby="setup-title">
      <div className="panel-heading">
        <span className="section-number">壹</span>
        <div>
          <h2 id="setup-title">对局设定</h2>
          <p>择边 · 择力 · 开枰</p>
        </div>
      </div>

      <fieldset className="field-group">
        <legend>执子</legend>
        <div className="segmented three">
          {([
            ["red", "执红"],
            ["black", "执黑"],
            ["random", "随机"],
          ] as const).map(([value, label]) => (
            <button
              type="button"
              key={value}
              className={props.humanSide === value ? "active" : ""}
              aria-pressed={props.humanSide === value}
              onClick={() => props.onHumanSide(value)}
            >
              {label}
            </button>
          ))}
        </div>
      </fieldset>

      <fieldset className="field-group">
        <legend>搜索强度</legend>
        <div className="difficulty-grid">
          {DIFFICULTIES.map((item) => (
            <button
              type="button"
              key={item.id}
              className={`difficulty-card${props.difficulty === item.id ? " active" : ""}`}
              aria-pressed={props.difficulty === item.id}
              onClick={() => props.onDifficulty(item.id)}
            >
              <span className="difficulty-main">
                <strong>{item.label}</strong>
                <em>{item.sims}</em>
              </span>
              <span>{item.sub}</span>
            </button>
          ))}
        </div>
      </fieldset>

      <label className="select-label">
        <span>对弈模型</span>
        <select value={props.modelId} onChange={(event) => props.onModel(event.target.value)}>
          {props.models.map((model) => (
            <option key={model.id} value={model.id} disabled={!model.compatible}>
              {model.name}{model.compatible ? "" : "（不兼容）"}
            </option>
          ))}
        </select>
      </label>

      <button type="button" className="primary-action" disabled={props.busy} onClick={props.onStart}>
        <span className="seal-dot">弈</span>
        {props.busy ? "正在布子…" : "另开一局"}
      </button>
    </section>
  );
}
