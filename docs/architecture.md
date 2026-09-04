# 架构与兼容契约

## 产品数据流

```text
固定版本 CCPD
  -> 解码 / 解析 / 合法重放 / 去重 / 整盘切分
  -> 监督预热 1 epoch
  -> 3 轮 Gumbel 搜索自博弈（至少 50K 局面/轮）
  -> 最近 300K replay 优化 -> candidate
  -> 10 个轻量开局、交换红黑的 20 盘快速评测
  -> best / rollback
  -> FastAPI 模型目录 -> React 对弈 GUI
```

训练完成不等于棋力达标。训练损失用于诊断；最终 20 盘 Random 检查才是本产品
版最基础的可玩性证据，且不代表大师级棋力。

## 规则引擎

`GameState` 不可变；FEN 用于局面交换，ICCS 用于走法交换。Python 后端是正确性
参考，C++20/pybind11 后端加速走法生成、落子、将军检测、局面键和 perft。
`CHESSAI_RULES_BACKEND=auto|native|reference` 选择后端。

普通合法性覆盖全部七类棋子、蹩马腿、塞象眼、不过河、九宫、炮架、将帅照面、
应将、自杀、将死和困毙。版本化裁决器处理重复、连续将军和连续追捉；正式宣称
穷尽 WXF 规则前，仍需继续扩充官方长捉边界棋例。

## 模型契约

输入固定为 `[batch, 117, 10, 9]`，并旋转到当前行棋方位于下方：

- 最近 8 个局面的 14 类棋子平面：112；
- 上一着起点/终点：2；
- 重复状态：2；
- 归一化无吃子计数：1。

默认网络为 10 个 128 通道残差块、2,086 维策略头和 `tanh` 标量价值头。动作
词表有版本和 SHA-256，非法着在 softmax 前屏蔽。checkpoint 加载同时校验：

| 契约 | 当前标识 |
| --- | --- |
| schema | `1` |
| rules | `wxf-2018-computer-v1` |
| features | `history8-117planes-v1` |
| action vocabulary | `iccs-2086-v1` 及 SHA-256 |

## 自博弈、replay 与恢复

`GumbelSearch` 按 [Gumbel AlphaZero](https://openreview.net/pdf?id=bERaNdoegnO)
实现根节点 Gumbel top-k 无放回采样、Sequential Halving、非根
改进策略、Q-value completion 和交替视角价值回传。确定性选着使用 Sequential
Halving 的最终存活动作，而不是访问数平手时的列表顺序；训练策略目标为
`softmax(logits + (c_visit + max_visit) * c_scale * completed_Q)`。基础产品训练前两轮
使用 16 次模拟，第三轮使用 32 次；快速模型对局为 64 次；GUI 四档为
8/32/128/256 次。独立的加强配置使用 8 轮、32–64 次模拟和 128 次 arena 搜索。

自博弈支持按累计 `target_positions` 停止，超出目标不超过一个 actor batch。
正式配置使用 `spawn` 启动 48 个独立 actor 进程，通过每 actor 一个固定共享内存
槽向主进程内唯一的 GPU 模型请求推理；动态批次达到 16 即执行、最大 64，等待
上限为 1 ms。actor 完成一盘后立即领取新任务，不再受 Python GIL 或固定波次中
最慢棋局限制。生成与优化仍交替执行，避免争抢单张 5090。

replay 使用压缩状态和稀疏策略，只训练最近 300,000 个局面。replay 元数据包含
`gumbel-completed-q-v2` 搜索目标版本，旧 raw-visit 目标生成的 shard 会被拒绝。
每轮拥有独立目录；
checkpoint 与 `playable-run-v2` 状态完成原子提交后，才会在当前运行目录内清理
窗口外的旧 replay。状态记录阶段、轮次、种子、局面计数、每个 replay manifest、
快速评测、运行时参数历史和权重散列。恢复时重新验证所有引用，不覆盖既有 shard。
旧 v1 运行可在保留 bootstrap/best 的前提下迁移；一次性的
`--restart-current-selfplay` 只归档并重跑尚未提交的当前 self-play。如果旧训练已
完成至少一轮并进入下一轮 self-play，`--restart-from-first-selfplay` 会先归档全部
旧 RL 轮次和当时的 best/rollback，再从不可变的 `bootstrap/bootstrap-best` 恢复并
重新开始第 1 轮；两个入口都不直接删除旧产物。

## 服务并发

API 默认监听环回地址。每个对局拥有独立状态历史、AI 任务和订阅队列。人类着法
携带 `expected_ply`，过期请求返回 409。悔棋、重开、认输或替换搜索会取消旧
任务，并在提交 AI 着法前再次校验 ply，避免幽灵落子。
