# 实施状态

本页区分“代码已实现”“本地流程证据”和“云端实际棋力证据”。完成训练命令或
损失下降不能替代对局结果。

| 范围 | 已实现 | 当前验证 | 仍需云端完成 |
| --- | --- | --- | --- |
| 规则 | Python 不可变引擎、C++20 热路径、重复/长将/长捉裁决 | perft 44 / 1,920 / 79,666；10,000 个随机可达局面差分通过 | 继续扩充 WXF 长捉边界棋例 |
| 表示 | 2,086 动作词表、117 平面、版本和散列兼容契约 | 词表、特征和 checkpoint 拒绝测试通过 | 无接口项 |
| 搜索 | Gumbel top-k、Sequential Halving、Q completion、合法 mask、价值回传 | 人工小树和象棋搜索测试通过 | 正式模型的吞吐和响应时间 |
| 数据 | 固定 CCPD 来源、许可、解码、合法重放、去重和整盘切分 | 53,685 条输入；27,667 盘去重接受；573 拒绝；全部 SHA-256 复验通过 | 无必需数据项 |
| 一键训练 | 1 epoch 监督预热、3 × 50K 自博弈、300K replay、20 盘快速评测、best/rollback | 真实 CCPD manifest 上的 CPU tiny 完整闭环和完成后恢复通过 | 正式至少 150,000 个 RL 局面 |
| Self-play 性能 | 48 个 spawn actor、共享内存请求槽、单 GPU 动态批推理、滚动调度、压缩 IPC、`runtime.json` | 进程 tiny、批推理、seed/shard 连续与异常保留测试 | RTX 5090 上 5,000 局面 pilot；相对旧基线至少 5 倍才继续 |
| 恢复与留存 | `playable-run-v2`、v1 备份迁移、原子状态、当前轮归档重启、散列复验、安全清理 | 篡改/缺失、v1 迁移、非空输出、候选接受/拒绝、轮换和越界删除测试通过 | 云端中断恢复演练 |
| 服务 | REST/WebSocket、expected-ply、防过期提交、可取消 AI 搜索、精简模型响应 | API 测试通过；已删除的 `/api/v1/runs` 返回 404 | 长时多会话稳定性 |
| GUI | 单一对弈页、SVG 棋盘、择边/难度/模型、FEN/PGN、AI 分析 | 3 个组件测试、生产构建、桌面/窄屏 4 个 Playwright 场景通过 | 使用正式 best 完成整盘人工验收 |
| 云端门禁 | 12 vCPU、32 GB RAM、25 GB 磁盘、24 GB VRAM、BF16、`sm_120`、native | 本地 doctor 正确判定只适合 smoke；云端配置书面满足资源门槛 | 在实际 RTX 5090 上运行 doctor 与 tiny |

完整准备数据的文件清单 SHA-256 为
`df9cd351b1c6280022a33cc912d41f8a1eb8da633c27980105923b5c818f0164`。
训练/验证/测试分别为 24,878 / 1,418 / 1,371 盘。这些是数据完整性事实，
不是棋力结论。

2026-09-03 本地验证快照：Ruff 格式/检查通过，Mypy 对 36 个源文件通过，
70 个 Python 测试通过，native 另由 Linux CI 执行 10,000 个随机局面差分；前端 3 个组件
测试、TypeScript、Vite 构建及桌面/窄屏 4 个 Playwright 场景通过；sdist 和 wheel
构建通过；真实处理后 CCPD 上的 `train playable --tiny` 及 `--resume` 通过。

当前没有把 tiny checkpoint 描述为可玩正式模型。只有云端完成至少 150,000 个局面，
并在最终 Random 20 盘达到至少 80% 得分、零非法着、零引擎崩溃后，才能标记
`playable_gate_passed=true`。
