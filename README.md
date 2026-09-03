# ChessAI 中国象棋

ChessAI 是一个面向实际人机对弈的中国象棋 AI：使用完整规则引擎、
Gumbel AlphaZero 搜索、棋谱监督预热和有限预算自博弈强化学习，并提供
FastAPI + React 的本地网页棋盘。项目采用单一训练路线，不包含消融实验、
外部引擎对比或研究仪表盘。

当前已清洗并验证完整 CCPD 数据：53,685 条输入得到 27,667 盘去重后的合法
棋谱，其中训练/验证/测试为 24,878 / 1,418 / 1,371 盘。数据集总量不到
200 MB，因此无需裁剪。仓库不附带训练完成的强模型；tiny 产物只证明流程
可运行，不能代表棋力。

## 本地开发

```powershell
uv sync --extra dev --extra train --extra native
uv run chessai doctor
uv run pytest

cd web
pnpm install
pnpm test
pnpm build
```

启动本地对弈服务：

```powershell
uv run chessai serve
```

服务默认只监听 `127.0.0.1:8000`。前端开发时可在 `web/` 运行
`pnpm dev`，Vite 会把 `/api` 代理到 Python 服务。

## 一键训练可玩模型

正式训练使用唯一配置 `configs/playable.yaml`：先对完整训练集监督预热
1 个 epoch，再生成 5 轮、合计约 500,000 个强化学习局面，replay 最多保留
最近 300,000 个局面。

```bash
chessai train playable data/processed/ccpd \
  --output runs/playable \
  --model-dir checkpoints \
  --config configs/playable.yaml
```

中断后使用完全相同的路径和配置恢复：

```bash
chessai train playable data/processed/ccpd \
  --output runs/playable \
  --model-dir checkpoints \
  --config configs/playable.yaml \
  --resume
```

状态清单、数据、replay 和 checkpoint 都会重新校验 SHA-256。未传
`--resume` 时，非空输出目录会被拒绝，防止覆盖既有结果。正式训练前可先跑：

```bash
chessai train playable data/processed/ccpd \
  --output runs/playable-tiny \
  --model-dir checkpoints-tiny \
  --tiny
```

## 目录

- `src/chessai/engine`：不可变规则、FEN/ICCS、重复裁决和动作词表。
- `src/chessai/ai`：117 平面特征、ResNet 策略价值网络和 Gumbel 搜索。
- `src/chessai/data`：CCPD 获取、解析、校验、去重和数据清单。
- `src/chessai/training`：监督训练、自博弈、300K replay、恢复和轻量评测。
- `src/chessai/server`：FastAPI 人机对弈服务与可取消搜索任务。
- `web`：React/TypeScript“宋版棋谱 × 现代训练台”对弈界面。
- `native`：可选 C++20 规则后端；Python 始终是正确性参考实现。

源码使用 MIT 许可证。CCPD 数据保留其 CC BY 4.0 许可和独立署名；数据、
replay、模型权重及运行产物不会进入普通 Git 历史。详细契约见
[架构说明](docs/architecture.md)、[数据来源](docs/data.md)、
[轻量评测](docs/evaluation.md)和[RTX 5090 云端运行说明](docs/cloud-training.md)。
