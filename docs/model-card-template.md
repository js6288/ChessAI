# 模型卡：`<model-id>`

## 用途与状态

用于本地交互式中国象棋对弈。明确标记为 tiny 冒烟模型、训练完成但门禁未通过，
或可玩性门禁通过；不要将“可玩”描述为大师级。

## 兼容性

- schema version：
- rule version：
- feature version：
- action vocabulary version / SHA-256：
- network configuration：
- weights SHA-256：

## 训练来源

- CCPD source commit 与 prepared manifest SHA-256：
- 监督训练棋谱数、局面数、epoch 和优化参数：
- 自博弈总局面数、每轮 simulations、300K replay 窗口和种子：
- `best` 的父模型和被接受轮次：
- 硬件、PyTorch、CUDA、耗时和峰值显存：

## 轻量评测

记录 candidate 快速对局及最终 Random 20 盘的模型散列、预算、胜/和/负、得分率、
非法着和崩溃数。训练指标单列，不把损失曲线作为棋力证据。

## 局限

记录规则裁决边界、开局/残局覆盖不足、有限训练预算、价值校准情况、是否禁用自动
认输，以及任何未通过的门禁。

## 许可与署名

引用源码许可证、CCPD 数据清单和 CC BY 4.0 署名。
