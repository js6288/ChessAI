# 轻量评测协议

产品版只支持三类对手：`random`、内置 material alpha-beta depth-3，以及另一个
兼容 checkpoint。报告包含模型权重散列、胜/和/负、得分率、盘数、搜索预算、
种子和耗时，不计算 Elo 或统计置信区间。

## 每轮 candidate 检查

每轮强化学习后，从 10 个确定性可达浅层局面开始，每个开局交换红黑，共 20 盘。
candidate 与当前 `best` 都使用 64 次搜索模拟：

- 得分率不低于 50%：接受 candidate，原 `best` 成为 `rollback`；
- 得分率低于 50%：保留当前 `best`，删除失败 candidate；
- 出现非法着、模型异常或对局崩溃：直接拒绝 candidate。

这些开局是轻量工程检查，不冻结或发布开局散列，也不构成严谨棋力排名。

## 最终可玩性门禁

3 轮训练结束后，`best` 与 Random 对弈 20 盘。通过条件为：

- 得分率至少 80%；
- 零非法着；
- 零引擎崩溃。

未通过不会删除模型。运行状态仍标为完成，同时
`playable_gate_passed=false`，交付时必须说明“训练完成、可玩性门禁未通过”。

手工评测示例：

```bash
chessai evaluate \
  --checkpoint checkpoints/best \
  --opponent random \
  --games 20 \
  --simulations 64 \
  --device cuda \
  --output artifacts/random-check.json
```

`alpha-beta` 和 `checkpoint` 可用于额外的产品回归检查，但不是本轮正式门禁。
训练损失、策略准确率和吞吐量只能诊断训练过程，不应描述为棋力提升证据。
