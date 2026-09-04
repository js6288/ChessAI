# ChessAI RTX 5090 云端训练与模型回传操作手册

本文档用于把 `E:\AI_project\ChessAI` 迁移到云端 RTX 5090 服务器，完成
CCPD 监督预热和至少 150,000 个自博弈局面的产品版训练，再把最佳模型下载回
Windows 本地进行推理和网页人机对弈。

本文所有命令都以当前项目接口为准。正式训练不使用 Docker，也不会在云端重新
下载另一份 PyTorch；应直接复用服务器镜像预装的 PyTorch 2.8.0 + CUDA 12.8。

## 快速导航

- 第 1–2 节：确认数据盘并选择迁移方式；
- 第 3 节：推荐的 GitHub 源码同步 + CCPD 单独上传；
- 第 4 节：不使用 GitHub 的直接文件上传方案；
- 第 5–7 节：Python 环境、native 后端、doctor 和 tiny 门禁；
- 第 8 节：5090 性能 pilot；
- 第 9–11 节：正式训练、旧运行升级与普通恢复；
- 第 12–13 节：观察进度并判断模型是否训练好；
- 第 14–16 节：导出、下载并在本地推理；
- 第 17–18 节：后续同步、备份与故障排查。

## 1. 云端资源和目录要求

已知云端配置如下：

- Ubuntu 22.04；
- Python 3.12；
- PyTorch 2.8.0；
- CUDA 12.8；
- RTX 5090 32 GB × 1；
- Intel Xeon Platinum 8470Q，25 vCPU；
- 90 GB RAM；
- 30 GB 系统盘 `/`；
- 50 GB 高速数据盘 `/root/autodl-tmp`；
- 200 GB 文件存储 `/root/autodl-fs`（容量更大，但读写较慢）。

这套配置满足产品训练的计算门槛。截图已经确认 `/root/autodl-tmp` 是 50 GB 的
高速数据盘，实例关机后数据不会丢失，适合放置高读写频率的 replay 和 checkpoint；
它不会随保存镜像一起保存，因此释放实例前仍要把最终模型下载或备份到
`/root/autodl-fs`。开始训练前，`/root/autodl-tmp` 必须至少剩余 25 GB。
不要把仓库、虚拟环境、数据集、replay 或模型放在 30 GB 系统盘 `/`。

先在云端查看磁盘挂载点：

```bash
df -h
lsblk
```

本项目的云端根目录固定为：

```bash
export CHESSAI_ROOT=/root/autodl-tmp/ChessAI
```

后续所有训练命令都使用这个路径。`/root/autodl-fs` 只作为可选的低频备份位置，
不要把正在读写的 replay 放到该目录。

## 2. 选择迁移方式

推荐使用混合方案：

1. 源代码通过 GitHub 同步；
2. 处理后的 CCPD 数据通过云平台文件上传、`scp` 或 `rsync` 单独传输；
3. `runs/` 和 `checkpoints/` 不上传，首次云端训练时新建；
4. 后续代码更新继续使用 Git，但正式训练过程中不要切换提交或配置。

原因是项目的 `.gitignore` 明确忽略：

```text
data/raw/
data/processed/
runs/
checkpoints/
artifacts/
*.safetensors
*.pt
*.npz
```

所以 GitHub 适合代码版本管理，但不会自动带上 CCPD、replay 和模型。训练实际只
需要 `data/processed/ccpd`，不需要上传原始 CCPD；本地该处理后目录约 50 MB。

如果云端不能访问 GitHub，或暂时不想创建远端仓库，可以使用第 4 节的“直接文件
上传方案”。不要上传本地 `.venv`、`web/node_modules`、`native/build*` 或
Windows `.pyd`，这些内容在 Ubuntu 上不能复用。

## 3. 推荐方案：GitHub 同步代码，单独上传数据

### 3.1 确认本地代码已同步到 GitHub

项目已经发布到 `https://github.com/js6288/ChessAI`。源代码优先通过 GitHub
同步；数据、replay 和 checkpoint 仍按本节说明单独传输。

在 Windows PowerShell 中确认工作区、远端和提交，然后推送最新版本：

```powershell
Set-Location E:\AI_project\ChessAI

git status --short
git remote -v
git log -1 --oneline
git push origin main
```

不要把访问令牌直接写进脚本、README 或命令历史。推送前确认 `data/`、`runs/`、
`checkpoints/`、`.venv/` 和任何密钥没有进入提交：

```powershell
git status --short
git diff --cached --name-only
```

### 3.2 在云端克隆代码

登录云端终端后执行：

```bash
export CHESSAI_ROOT=/root/autodl-tmp/ChessAI
git clone git@github.com:js6288/ChessAI.git "$CHESSAI_ROOT"
cd "$CHESSAI_ROOT"
git rev-parse HEAD
git status --short
```

记下 `git rev-parse HEAD` 输出的提交散列。正式训练期间应保持这一提交不变。

如果云端没有 GitHub SSH key，可使用 HTTPS 克隆，或者在 GitHub 为该服务器配置
只读 deploy key。

### 3.3 打包本地处理后数据

训练不需要上传 `data/raw/ccpd`。在 Windows PowerShell 中只打包处理后数据：

```powershell
Set-Location E:\AI_project\ChessAI

tar -czf ..\ChessAI-ccpd-processed.tar.gz -C .\data\processed ccpd
Get-FileHash -Algorithm SHA256 ..\ChessAI-ccpd-processed.tar.gz
```

保存 PowerShell 输出的 SHA-256。上传后要在云端复核这个压缩包没有损坏。

### 3.4 上传处理后数据

可选以下任一方式。

方式 A：使用云平台网页的“文件上传”功能，把
`E:\AI_project\ChessAI-ccpd-processed.tar.gz` 上传到数据盘的临时目录。

方式 B：如果云平台提供 SSH 地址和端口，在 Windows PowerShell 中使用：

```powershell
scp -P SSH端口 "E:\AI_project\ChessAI-ccpd-processed.tar.gz" `
  云端用户名@云端地址:/root/autodl-tmp/
```

如果 SSH 使用默认 22 端口，可以省略 `-P SSH端口`。

方式 C：Linux/macOS 本地可使用断点续传能力更好的 `rsync`：

```bash
rsync -avP -e "ssh -p SSH端口" ChessAI-ccpd-processed.tar.gz \
  云端用户名@云端地址:/root/autodl-tmp/
```

### 3.5 在云端校验并解压数据

假设压缩包上传到了数据盘根目录：

```bash
export CHESSAI_ROOT=/root/autodl-tmp/ChessAI
cd /root/autodl-tmp

sha256sum ChessAI-ccpd-processed.tar.gz
mkdir -p "$CHESSAI_ROOT/data/processed"
tar -xzf ChessAI-ccpd-processed.tar.gz -C "$CHESSAI_ROOT/data/processed"

ls -lh "$CHESSAI_ROOT/data/processed/ccpd"
```

云端 `sha256sum` 必须与本地 `Get-FileHash` 一致。解压后目录应包含：

```text
data/processed/ccpd/
├── manifest.json
├── file-manifest.jsonl
├── train.jsonl
├── validation.jsonl
└── test.jsonl
```

不要只上传三个 JSONL；`manifest.json` 和 `file-manifest.jsonl` 也参与完整性验证。

## 4. 备用方案：完全通过文件上传迁移

如果不使用 GitHub，可以分别制作“代码压缩包”和“处理后数据压缩包”。

在 Windows PowerShell 中执行：

```powershell
Set-Location E:\AI_project\ChessAI

tar -czf ..\ChessAI-code.tar.gz `
  --exclude=.git `
  --exclude=.venv `
  --exclude=data `
  --exclude=runs `
  --exclude=checkpoints `
  --exclude=artifacts `
  --exclude=dist `
  --exclude=web/node_modules `
  --exclude=web/dist `
  --exclude=native/build* `
  .

tar -czf ..\ChessAI-ccpd-processed.tar.gz -C .\data\processed ccpd

Get-FileHash -Algorithm SHA256 ..\ChessAI-code.tar.gz
Get-FileHash -Algorithm SHA256 ..\ChessAI-ccpd-processed.tar.gz
```

通过云平台网页或 `scp` 上传两个压缩包。在云端执行：

```bash
export CHESSAI_ROOT=/root/autodl-tmp/ChessAI
mkdir -p "$CHESSAI_ROOT"
tar -xzf /上传目录/ChessAI-code.tar.gz -C "$CHESSAI_ROOT"

mkdir -p "$CHESSAI_ROOT/data/processed"
tar -xzf /上传目录/ChessAI-ccpd-processed.tar.gz \
  -C "$CHESSAI_ROOT/data/processed"

cd "$CHESSAI_ROOT"
ls -la
```

这种方式可以训练，但缺少 Git 提交散列，不利于以后复现和更新。建议至少保留本地
代码压缩包 SHA-256，并把它写入训练记录。长期使用仍推荐 GitHub 方案。

## 5. 创建云端 Python 环境

进入项目目录并创建位于数据盘内的虚拟环境：

```bash
export CHESSAI_ROOT=/root/autodl-tmp/ChessAI
cd "$CHESSAI_ROOT"

# 先确认这个解释器就是平台预装 PyTorch 所在的 Python。
python3.12 -c "import torch; print(torch.__version__, torch.version.cuda)"
python3.12 -m venv --system-site-packages .venv
source .venv/bin/activate

which python
python --version
python -m pip install --upgrade uv
```

`--system-site-packages` 很重要，它让虚拟环境复用镜像预装的 CUDA PyTorch。安装
项目时故意不启用 `train` extra，避免依赖解析器下载第二份 PyTorch：

```bash
uv pip install -e ".[dev,native]"
```

如果第一条 `python3.12 -c` 找不到 torch，但云平台当前激活的 `python` 可以导入
torch，说明预装包可能位于平台的 base/Conda 环境。此时用那个解释器创建 venv：

```bash
python -c "import sys, torch; print(sys.executable, torch.__version__, torch.version.cuda)"
python -m venv --system-site-packages .venv
source .venv/bin/activate
```

无论选择哪一个解释器，最终虚拟环境都必须是 Python 3.12，并且能看到预装的
PyTorch 2.8.0 和 CUDA 12.8。

立即确认实际使用的 PyTorch、CUDA 和 GPU：

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("BF16 supported:", torch.cuda.is_bf16_supported())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("compiled arches:", torch.cuda.get_arch_list())
PY
```

预期至少应看到：

```text
torch: 2.8.0...
torch CUDA runtime: 12.8
CUDA available: True
BF16 supported: True
GPU: NVIDIA GeForce RTX 5090
capability: (12, 0)
```

如果安装过程中开始下载多 GB 的 `torch` 或 CUDA wheel，应停止安装并检查虚拟
环境是否确实使用了 `--system-site-packages`。不要让重复 wheel 占满系统盘。

## 6. 编译并验证 native C++ 后端

正式自博弈必须使用 native 后端：

```bash
cd "$CHESSAI_ROOT"
source .venv/bin/activate

cmake -S native -B native/build-linux -G Ninja \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build native/build-linux --config Release
```

构建会把 `_chessai_native*.so` 输出到项目的 `src/`。验证 perft：

```bash
python - <<'PY'
import _chessai_native as native

fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
print("perft-1:", native.perft(fen, 1))
print("perft-2:", native.perft(fen, 2))
print("perft-3:", native.perft(fen, 3))
PY
```

预期结果为 `44`、`1920`、`79666`。

## 7. 正式训练前的两道门禁

### 7.1 doctor 和数据完整性

```bash
cd "$CHESSAI_ROOT"
source .venv/bin/activate
export CHESSAI_RULES_BACKEND=native
mkdir -p artifacts

chessai doctor --workspace "$CHESSAI_ROOT" | tee artifacts/doctor.json
chessai data validate data/processed/ccpd | tee artifacts/ccpd-validate.json
```

`doctor.json` 中必须有：

```json
"ready_for_playable_training": true
```

同时确认：

- `cuda_available` 为 `true`；
- `bf16_supported` 为 `true`；
- `required_device_arch` 为 `sm_120`；
- `device_arch_compatible` 为 `true`；
- `vram_gb` 不低于 24；
- `disk_free_gb` 不低于 25；
- `native_backend` 为 `true`；
- `selected_backend` 为 `native`；
- `inference_probe.ok` 为 `true`；
- `outputs_finite` 为 `true`。

数据验证预期应显示：

- train：24,878 盘；
- validation：1,418 盘；
- test：1,371 盘；
- file manifest SHA-256：
  `df9cd351b1c6280022a33cc912d41f8a1eb8da633c27980105923b5c818f0164`。

任意一项不一致都不要开始正式训练。

### 7.2 tiny 完整闭环

先在云端实际跑通监督、自博弈、RL、模型轮换和最终评测：

```bash
chessai train playable data/processed/ccpd \
  --output runs/playable-tiny \
  --model-dir checkpoints-tiny \
  --tiny
```

完成后检查：

```bash
python -m json.tool runs/playable-tiny/state.json
ls -lh checkpoints-tiny/best
```

tiny 状态应为 `stage: complete`。tiny 只有极少局面和 2 ply 对局，因此
`playable_gate_passed: false` 完全正常；它只证明训练链路可用，不代表棋力。

再验证一次恢复不会重复执行：

```bash
chessai train playable data/processed/ccpd \
  --output runs/playable-tiny \
  --model-dir checkpoints-tiny \
  --tiny \
  --resume
```

## 8. 先运行 5,000 局面性能 pilot

多进程版本必须先用现有监督模型做独立 pilot；该命令使用系统临时目录保存 replay，
结束后只留下 JSON 报告，不会修改 `runs/playable`：

```bash
chessai benchmark selfplay checkpoints/best \
  --positions 5000 \
  --simulations 16 \
  --actors 48 \
  --output artifacts/selfplay-benchmark.json
```

查看关键结果：

```bash
python - <<'PY'
import json
from pathlib import Path

r = json.loads(Path("artifacts/selfplay-benchmark.json").read_text())
print("positions/s:", r["positions_per_second"])
print("games/s:", r["games_per_second"])
print("mean batch:", r["inference"].get("mean_batch_size"))
print("largest batch:", r["inference"].get("largest_batch"))
print("request p95 ms:", r["inference"].get("request_ms_p95"))
print("CUDA peak GiB:", r["cuda_peak_memory_bytes"] / 2**30)
print("failures:", r["failures"])
PY
```

与升级前留存的旧 manifest 在相同 checkpoint、16 simulations 和 max ply 下比较。
新吞吐必须至少达到旧实现的 5 倍，并且 mean batch 至少 16、largest batch 至少 32、
无 worker crash/timeout/NaN/非法着，才启动正式训练。显存占用不是门禁。

## 9. 启动正式 150,000 局面训练

建议在 `tmux` 中运行，避免 SSH 断开导致前台进程退出。若镜像没有 `tmux`，先按
平台允许的方式安装，或使用平台自带的持久终端。

```bash
tmux new -s chessai-train
```

进入 tmux 后执行：

```bash
export CHESSAI_ROOT=/root/autodl-tmp/ChessAI
cd "$CHESSAI_ROOT"
source .venv/bin/activate
export CHESSAI_RULES_BACKEND=native

mkdir -p artifacts runs checkpoints
git rev-parse HEAD 2>/dev/null | tee artifacts/training-code-commit.txt

set -o pipefail
chessai train playable data/processed/ccpd \
  --output runs/playable \
  --model-dir checkpoints \
  --config configs/playable.yaml \
  2>&1 | tee -a artifacts/playable-console.log
```

日志必须放在 `artifacts/`，不能在首次启动前写入 `runs/playable/`。训练命令会拒绝
未带 `--resume` 的非空输出目录，这是防止覆盖数据的安全设计。

按 `Ctrl+B`，再按 `D`，即可退出 tmux 界面但保持训练运行。重新连接后查看：

```bash
tmux ls
tmux attach -t chessai-train
```

正式流程固定为：

1. 完整 24,878 盘训练集监督预热 1 epoch；
2. 完整验证集和测试集评估；
3. 第 1–2 轮各生成至少 50,000 个局面，16 simulations；
4. 第 3 轮生成至少 50,000 个局面，32 simulations；
5. 每轮从最近最多 300,000 个 replay 局面训练 1 epoch；
6. 每轮进行 20 盘 candidate 对 `best` 快速对局；
7. 训练结束后进行 20 盘 `best` 对 Random 的可玩性检查。

## 10. 从旧线程版升级并恢复

### 10.1 旧训练仍在第 1 轮 self-play

当前云端已经完成监督预热、正在旧版第一轮 self-play 时，按以下顺序操作。先在训练
终端按一次 `Ctrl+C` 并等待进程完全退出；旧进程仍运行时不得 `git pull`。

```bash
cd /root/autodl-tmp/ChessAI
cp runs/playable/state.json artifacts/state-before-process-upgrade.json
cp runs/playable/iteration-001/selfplay/manifest.json \
  artifacts/selfplay-thread-baseline.json
git pull --ff-only
source .venv/bin/activate
uv pip install -e ".[dev,native]"

cmake -S native -B native/build-linux -G Ninja \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build native/build-linux --config Release
```

依次重跑 doctor、perft、测试和第 8 节 benchmark。pilot 达标后，只在这一次执行：

```bash
chessai train playable data/processed/ccpd \
  --output runs/playable \
  --model-dir checkpoints \
  --config configs/playable.yaml \
  --resume \
  --restart-current-selfplay \
  2>&1 | tee -a artifacts/playable-console.log
```

它会创建 `runs/playable/state.v1.backup.json`，把旧的部分第一轮原子移动到
`runs/playable/abandoned/`，保留 `bootstrap_positions` 和 `checkpoints/best`，然后按
新预算从第一轮 0 个 RL 局面开始。确认新目录已产生完整 shard 且 `runtime.json`
持续更新后，以后的恢复不得再带 `--restart-current-selfplay`。

### 10.2 旧训练已进入第 2 轮 self-play

如果状态显示 `stage: selfplay`、`iteration: 1` 且
`rl_generated_positions` 已包含旧第 1 轮，则 `--restart-current-selfplay` 只会重启
第 2 轮，不能回到第 1 轮。不要手工改 `state.json`，改用一次性的完整 RL 回退：

```bash
chessai train playable data/processed/ccpd \
  --output runs/playable \
  --model-dir checkpoints \
  --config configs/playable.yaml \
  --resume \
  --restart-from-first-selfplay \
  2>&1 | tee -a artifacts/playable-console.log
```

程序会在重新生成任何局面前完成以下校验和归档：

1. 复验数据 manifest、现有 best/rollback 和所有 retained replay 的 SHA-256；
2. 验证 `runs/playable/bootstrap/bootstrap-best` 是完整且兼容的 checkpoint；
3. 保存原始 v1 状态到 `state.v1.backup.json`；
4. 把旧 `iteration-001`、当前部分 `iteration-002`、重启前状态以及旧 best/rollback
   归档到 `runs/playable/abandoned/restart-from-first-selfplay-XX/`；
5. 从 `bootstrap/bootstrap-best` 恢复 `checkpoints/best`；
6. 把 `iteration` 和 `rl_generated_positions` 归零，只保留已完成的 bootstrap；
7. 按当前配置重新生成第 1 轮至少 50,000 个局面。

该操作不删除旧轮次或旧模型。它只允许在已经完成至少一轮、当前恰好处于下一轮
`selfplay` 且没有活跃 candidate 时使用。执行后所有普通恢复只能带 `--resume`，
不得再次携带 `--restart-from-first-selfplay` 或 `--restart-current-selfplay`。

## 11. 普通中断后恢复训练

如果服务器重启、SSH 断线或进程被中断，先确认数据盘仍挂载，然后使用原来的
数据、输出、模型和配置路径：

```bash
export CHESSAI_ROOT=/root/autodl-tmp/ChessAI
cd "$CHESSAI_ROOT"
source .venv/bin/activate
export CHESSAI_RULES_BACKEND=native

chessai doctor --workspace "$CHESSAI_ROOT"

set -o pipefail
chessai train playable data/processed/ccpd \
  --output runs/playable \
  --model-dir checkpoints \
  --config configs/playable.yaml \
  --resume \
  2>&1 | tee -a artifacts/playable-console.log
```

v2 状态允许修改 actor 数、推理批次、等待/超时和 channels-last 等运行时参数；
模型、规则、特征、随机种子、3×50K 预算和 `[16,16,32]` schedule 等训练语义不能
修改。恢复时不要：

- 改动 `configs/playable.yaml`；
- 移动或重命名 `runs/playable`；
- 手工删除 replay shard；
- 替换 `checkpoints/best` 或 `checkpoints/rollback`；
- 在原运行上切换不兼容的代码提交；
- 因为目录非空而去掉 `--resume`。

恢复会重新验证数据、配置、replay manifest、replay shard 和 checkpoint 散列。
如果提示 hash mismatch，应先查明文件是否传输不完整或被手工修改，不要绕过检查。

## 12. 观察训练进度和机器状态

### 12.1 GPU、CPU 和磁盘

另开一个 SSH 会话执行：

```bash
watch -n 5 nvidia-smi
```

观察更细的 GPU 利用率和显存：

```bash
nvidia-smi dmon -s pucmt -d 5
```

查看 CPU actor 和内存：

```bash
htop
```

查看项目空间：

```bash
cd "$CHESSAI_ROOT"
df -h "$CHESSAI_ROOT"
du -sh data runs checkpoints artifacts .venv 2>/dev/null
```

自博弈和网络更新是交替进行的，GPU 利用率可能阶段性波动。短暂降低是正常的；如果
GPU 长时间为 0%、self-play manifest 的 positions 也不再增长，则需要检查进程和
日志。磁盘低于 25 GB 时应暂停排查，不要删除当前状态清单仍引用的 replay。

### 12.2 查看总状态

```bash
cd "$CHESSAI_ROOT"
python -m json.tool runs/playable/state.json
```

也可以只打印关键字段：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("runs/playable/state.json")
state = json.loads(path.read_text(encoding="utf-8"))
print("stage:", state["stage"])
print("iteration:", state["iteration"])
print("bootstrap positions:", state["bootstrap_positions"])
print("RL generated positions:", state["rl_generated_positions"])
print("completed steps:")
for step in state["completed_steps"]:
    print("  -", step)
print("active checkpoint:", state["active_checkpoint"])
print("playable gate:", state["playable_gate_passed"])
PY
```

状态只会在一个阶段完成并原子提交后推进。在某轮 self-play 尚未完成时，要查看该轮
不断更新的 manifest。例如第一轮：

```bash
python -m json.tool runs/playable/iteration-001/selfplay/manifest.json
python -m json.tool runs/playable/iteration-001/selfplay/runtime.json
tail -f runs/playable/iteration-001/selfplay/metrics.jsonl
```

`runtime.json` 每 10 秒原子更新，可查看进程存活数、在途棋局、即时 positions/s、
mean/p50/p95 batch、请求延迟、特征编码和规则/搜索耗时。后续轮次将 `001` 替换为
`002` 或 `003`。

### 12.3 自动显示当前 self-play 百分比、吞吐和 ETA

下面的命令会根据顶层状态自动定位当前轮次，无需手工把路径中的 `001` 改成
`002` 或 `003`。它显示已原子提交的局面数、50,000 目标、完成百分比、本次进程
即时吞吐、预计剩余时间、Actor 存活数、在途棋局和 GPU 推理 batch：

```bash
cd /root/autodl-tmp/ChessAI

selfplay_progress() {
python - <<'PY'
import json
from pathlib import Path

run = Path("runs/playable")
state = json.loads((run / "state.json").read_text(encoding="utf-8"))

if state["stage"] != "selfplay":
    print(f"当前不是 self-play 阶段: stage={state['stage']!r}")
    print(f"已完成 RL 局面: {state['rl_generated_positions']:,}")
    raise SystemExit(0)

iteration = int(state["iteration"]) + 1
selfplay = run / f"iteration-{iteration:03d}" / "selfplay"
runtime_path = selfplay / "runtime.json"
manifest_path = selfplay / "manifest.json"

runtime = (
    json.loads(runtime_path.read_text(encoding="utf-8"))
    if runtime_path.is_file()
    else {}
)
manifest = (
    json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_path.is_file()
    else {}
)

target = int(state["config"]["positions_per_iteration"])
committed = int(manifest.get("positions", runtime.get("completed_positions", 0)))
invocation = int(runtime.get("completed_positions", 0))
rate = float(runtime.get("positions_per_second", 0.0))
remaining = max(0, target - committed)
eta_seconds = remaining / rate if rate > 0 else None
inference = runtime.get("inference", {})
heartbeat = inference.get("actor_heartbeat_age_seconds", [])

def duration(seconds):
    if seconds is None:
        return "暂不可计算"
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

print(f"当前阶段          : selfplay")
print(f"当前轮次          : {iteration}/3")
print(f"本轮搜索次数      : {state['config']['simulation_schedule'][iteration - 1]}")
print(f"已提交/目标局面   : {committed:,}/{target:,}")
print(f"完成百分比        : {min(100.0, committed / target * 100):.2f}%")
print(f"本次启动完成局面  : {invocation:,}")
print(f"即时吞吐          : {rate:.2f} positions/s")
print(f"预计剩余时间      : {duration(eta_seconds)}")
print(f"已完成棋局        : {runtime.get('completed_games', 0)}")
print(
    "Actor 存活/总数  : "
    f"{inference.get('actors_alive', 0)}/{runtime.get('actors', 0)}"
)
print(f"在途棋局          : {inference.get('inflight_games', 0)}")
print(f"推理 requests     : {inference.get('requests', 0)}")
print(f"推理 batches      : {inference.get('batches', 0)}")
print(f"平均 batch        : {inference.get('mean_batch_size', 0):.2f}")
print(f"p95 batch         : {inference.get('batch_size_p95', 0):.2f}")
print(f"最大 batch        : {inference.get('largest_batch', 0)}")
print(f"请求 p95          : {inference.get('request_ms_p95', 0):.2f} ms")
if heartbeat:
    print(f"最久 Actor 心跳   : {max(heartbeat):.1f} s")
print(f"状态目录          : {selfplay}")
PY
}

selfplay_progress
```

`manifest.json` 的 `positions` 是已经写入完整 replay shard 并原子提交的累计数，最适合
判断可恢复进度；它最多会落后一个正在写入的 shard。`runtime.json` 中的
`completed_positions` 是本次进程启动后的计数，恢复训练后会从 0 重新累计，因此
不能把两者直接相加。ETA 使用当前进程的即时吞吐估算，只作为观察值。

上面的命令同时在当前 shell 中定义了 `selfplay_progress` 函数。若希望每 10 秒自动
刷新，紧接着执行下面的循环；按 `Ctrl+C` 只会结束监控，不会终止另一个终端或
tmux 中的正式训练：

```bash
while true; do
  clear
  date '+%F %T'
  selfplay_progress
  sleep 10
done
```

### 12.4 查看监督和 RL 指标

监督预热：

```bash
tail -n 20 runs/playable/bootstrap/metrics.jsonl
```

某一轮 RL：

```bash
tail -n 20 runs/playable/iteration-001/rl/metrics.jsonl
```

重点检查：

- 没有 `NaN`、`Inf`、CUDA OOM 或 Python traceback；
- loss 是有限数值；
- `positions` 和 `step` 持续增加；
- self-play `positions_per_second` 大于 0；
- 推理统计中的 requests、batches 持续增加；
- checkpoint 能写入且散列校验通过。

loss 下降只能说明优化过程在工作，不能单独证明模型棋力已经训练好。

## 13. 如何判断模型是否训练好

### 11.1 先判断训练是否完整结束

正式命令应正常退出，且以下命令显示 `stage: complete`：

```bash
python -m json.tool runs/playable/state.json
```

完整运行的关键条件是：

- `iteration` 等于 3；
- `rl_generated_positions` 至少为 150,000；
- `candidate_checkpoint` 为 `null`；
- `active_checkpoint` 指向 `checkpoints/best`；
- `completed_steps` 包含 bootstrap、3 轮 selfplay/RL/arena 和 final-evaluation；
- `checkpoints/best/metadata.json` 与 `weights.safetensors` 存在。

由于 actor 按批完成棋局，累计局面数可能略高于 150,000，这是正常的受控超出。

### 11.2 再判断基础可玩性是否通过

查看最终报告：

```bash
python -m json.tool runs/playable/final-evaluation.json
```

以及状态中的：

```json
"playable_gate_passed": true
```

通过门禁需要最终 20 盘对 Random：

- `games` 等于 20；
- `score_rate` 不低于 0.80；
- 没有非法着；
- 没有模型或引擎崩溃；
- 报告中没有 `error` 字段。

如果 `stage` 已经是 `complete`，但 `playable_gate_passed` 为 `false`，含义是
“训练流程完成，但基础可玩性门禁未通过”。模型仍会保留在 `checkpoints/best`，
但不能把它描述为已达到预期棋力。

### 11.3 查看每轮 candidate 是否真正改进

```bash
python - <<'PY'
import json
from pathlib import Path

state = json.loads(Path("runs/playable/state.json").read_text(encoding="utf-8"))
for item in state["evaluations"]:
    summary = item["summary"]
    print(
        "iteration", item["iteration"],
        "accepted", item["accepted"],
        "W/D/L", summary.get("wins"), summary.get("draws"), summary.get("losses"),
        "score", summary.get("score_rate"),
    )
PY
```

candidate 得分率不低于 50% 才会替换 `best`。如果多轮全部被拒绝，最终模型可能
仍接近监督预热模型；即使训练生成了 150,000 个局面，也不能据此声称 RL 有效。

### 11.4 独立重跑一次 Random 检查

训练结束后建议再独立执行：

```bash
chessai evaluate \
  --checkpoint checkpoints/best \
  --opponent random \
  --games 20 \
  --simulations 64 \
  --device cuda \
  --output artifacts/random-check.json
```

可选地对内置 material alpha-beta depth-3 做额外回归：

```bash
chessai evaluate \
  --checkpoint checkpoints/best \
  --opponent alpha-beta \
  --games 20 \
  --simulations 64 \
  --device cuda \
  --output artifacts/alpha-beta-check.json
```

产品版“训练好”的最低定义是规则正确、能稳定完成对局、最终 Random 门禁通过。
它不代表大师级，也不代表已经达到专业象棋引擎水平。

## 14. 导出云端最佳模型

不要直接只下载一个裸 `weights.safetensors`，因为本地还需要模型结构、规则版本、
117 平面版本和 2,086 动作词表散列。使用项目导出命令生成完整推理包：

```bash
cd "$CHESSAI_ROOT"
source .venv/bin/activate

chessai export \
  checkpoints/best \
  artifacts/chessai-best \
  --data-manifest data/processed/ccpd/manifest.json \
  --evaluation-report runs/playable/final-evaluation.json
```

推理包应包含：

```text
artifacts/chessai-best/
├── best.safetensors
├── metadata.json
├── config.json
├── action-vocabulary.txt
├── action-vocabulary.json
├── data-manifest.json
├── evaluation-report.json
├── MODEL_CARD.md
├── LICENSE
└── ATTRIBUTION.md
```

推理不需要优化器和 RNG 状态，因此不要加 `--include-resume-state`。只有准备在另一
台机器继续训练时，才需要同时保存 `training-state.pt`、完整 `runs/playable`、
replay、数据和原始配置。

打包并生成校验文件：

```bash
cd "$CHESSAI_ROOT"
tar -czf artifacts/chessai-best.tar.gz -C artifacts chessai-best
sha256sum artifacts/chessai-best.tar.gz \
  | tee artifacts/chessai-best.tar.gz.sha256

ls -lh artifacts/chessai-best.tar.gz*
```

可以再复制一份到 200 GB 文件存储，作为关闭实例前的冗余备份：

```bash
mkdir -p /root/autodl-fs/ChessAI-backup
cp -a artifacts/chessai-best.tar.gz \
  artifacts/chessai-best.tar.gz.sha256 \
  runs/playable/state.json \
  runs/playable/final-evaluation.json \
  /root/autodl-fs/ChessAI-backup/

ls -lh /root/autodl-fs/ChessAI-backup
```

训练热路径仍保留在 `/root/autodl-tmp`；这里只复制最终小文件，不把 replay 移到
较慢的 `/root/autodl-fs`。在关闭或释放云端实例前，确认模型已经下载到本地，或
至少存在于 `/root/autodl-fs/ChessAI-backup`。

## 15. 把模型下载回 Windows 本地

### 13.1 使用云平台网页下载

在云平台文件管理器中下载：

- `artifacts/chessai-best.tar.gz`；
- `artifacts/chessai-best.tar.gz.sha256`。

建议保存到：

```text
E:\AI_project\ChessAI\artifacts\cloud-download\
```

### 13.2 使用 scp 下载

在本地 Windows PowerShell 中执行：

```powershell
New-Item -ItemType Directory -Force `
  E:\AI_project\ChessAI\artifacts\cloud-download | Out-Null

scp -P SSH端口 `
  云端用户名@云端地址:/root/autodl-tmp/ChessAI/artifacts/chessai-best.tar.gz `
  E:\AI_project\ChessAI\artifacts\cloud-download\

scp -P SSH端口 `
  云端用户名@云端地址:/root/autodl-tmp/ChessAI/artifacts/chessai-best.tar.gz.sha256 `
  E:\AI_project\ChessAI\artifacts\cloud-download\
```

默认 22 端口时可省略 `-P SSH端口`。

### 13.3 在本地校验压缩包

```powershell
Set-Location E:\AI_project\ChessAI\artifacts\cloud-download
Get-Content .\chessai-best.tar.gz.sha256
Get-FileHash -Algorithm SHA256 .\chessai-best.tar.gz
```

两边 SHA-256 必须完全相同。不同则重新下载，不要加载不完整的权重。

## 16. 在本地安装模型并推理

### 14.1 解压到本地 checkpoints

```powershell
Set-Location E:\AI_project\ChessAI
New-Item -ItemType Directory -Force .\checkpoints | Out-Null

tar -xzf .\artifacts\cloud-download\chessai-best.tar.gz `
  -C .\checkpoints
```

解压后应为：

```text
E:\AI_project\ChessAI\checkpoints\chessai-best\
```

验证 checkpoint 兼容性和权重散列：

```powershell
uv run python -c "from chessai.training.checkpoint import load_checkpoint; c=load_checkpoint(r'checkpoints/chessai-best'); print(c.weights_path); print(c.metadata['compatibility'])"
```

如果出现 action vocabulary、feature、rule 或 schema 不兼容，说明本地代码版本与
训练模型不匹配。应检出训练时记录的 Git 提交，而不是关闭兼容检查。

### 14.2 命令行做一次搜索推理

```powershell
@'
from chessai.ai.evaluator import TorchEvaluator
from chessai.ai.search import GumbelSearch
from chessai.engine import GameState
from chessai.training.checkpoint import load_checkpoint

loaded = load_checkpoint("checkpoints/chessai-best", device="cpu")
evaluator = TorchEvaluator(loaded.model, device="cpu", precision="fp32")
search = GumbelSearch(
    evaluator,
    simulations=32,
    max_considered_actions=16,
    seed=20260902,
)
result = search.search(GameState.initial())
print("best move:", result.best_move)
print("value:", result.value)
print("elapsed ms:", result.elapsed_ms)
print("principal variation:", result.principal_variation)
'@ | uv run python -
```

能打印合法 ICCS 着法且没有兼容/散列错误，说明推理包已正确安装。

### 14.3 在本地网页 GUI 中使用模型

如果前端尚未构建：

```powershell
Set-Location E:\AI_project\ChessAI\web
pnpm install
pnpm build
```

启动服务：

```powershell
Set-Location E:\AI_project\ChessAI
$env:CHESSAI_MODEL_DIR = "E:\AI_project\ChessAI\checkpoints"
uv run chessai serve --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

在“对弈模型”下拉框中选择 `chessai-best`，再选择执红、执黑或随机方以及搜索
难度。服务默认只监听 `127.0.0.1`，不会暴露到公网。

当前 GUI 模型注册表默认在 CPU 上加载网络。本地 RTX 3060 Laptop 仍可完成 CPU
推理和标准档对弈，但 128/256 simulations 可能明显更慢；先使用 8 或 32
simulations 验证完整对局。

## 17. 后续代码同步与模型备份

使用 GitHub 方案时，后续正常流程是：

```powershell
# Windows 本地修改、测试、提交并推送
git push
```

```bash
# 云端没有训练进程时更新
cd "$CHESSAI_ROOT"
git status --short
git pull --ff-only
```

不要在正式训练运行中执行 `git pull`。如果更新改变规则、特征、动作词表或配置，
现有 run 可能不能恢复，这是有意的兼容保护。

至少备份以下最终文件：

- `artifacts/chessai-best.tar.gz`；
- `artifacts/chessai-best.tar.gz.sha256`；
- `runs/playable/state.json`；
- `runs/playable/final-evaluation.json`；
- `artifacts/training-code-commit.txt`；
- `configs/playable.yaml`；
- `artifacts/doctor.json`；
- `artifacts/ccpd-validate.json`。

只进行本地推理时，完整的 `chessai-best.tar.gz` 推理包已经足够；不必下载数十万
局面的 replay，也不必下载 `rollback` 或优化器状态。

## 18. 常见问题排查

### doctor 显示 `ready_for_playable_training: false`

按 `recommendations` 逐项处理。重点确认项目实际位于 50 GB 数据盘、PyTorch 能
看到 RTX 5090、BF16 为 true、架构是 `sm_120`，以及 native 后端已经编译。

### 安装时开始下载另一份 PyTorch

停止安装，删除刚创建但尚未使用的虚拟环境后，重新用
`python3.12 -m venv --system-site-packages .venv` 创建。安装项目时使用
`uv pip install -e ".[dev,native]"`，不要在云端预装镜像上使用 `.[train]`。

### 提示输出目录非空

如果这是之前中断的正式运行，使用同样的路径加 `--resume`。如果不是同一次运行，
换一个新的输出目录。不要为了消除错误而删除不确定来源的目录。

### 提示 checkpoint 或 replay hash mismatch

文件已损坏、传输不完整或被修改。先备份现场并比对状态清单，不要修改 JSON 绕过
校验。若是刚上传的数据，重新上传并运行 `chessai data validate`。

### GPU 利用率不持续 100%

训练采用 CPU actor、GPU 批量推理和 GPU 优化交替流水线，不保证始终满载。应结合
self-play positions/s、manifest 是否增长、GPU 显存和 CPU 利用率判断。短暂空闲
正常，长时间为 0 且局面数不增加才是异常。

### `playable_gate_passed` 为 false

训练流程完成了，但20盘 Random 得分率没有达到80%，或评测发生错误。保留
`checkpoints/best` 和所有报告；可以先在 GUI 实际试玩，但不要宣称已达到目标棋力。

## 19. 使用修正版搜索重新训练加强模型

如果旧运行的所有 RL candidate 都被拒绝，而且 `checkpoints/best/metadata.json` 仍然
显示 `training.kind: supervised-bootstrap`，继续给旧运行简单追加轮数并不能解决根因。
旧实现把根节点访问计数作为训练策略目标，并在小预算访问数打平时按合法着顺序选着；
修正版改为 completed-Q 改进策略，并使用 Sequential Halving 的最终排名选着。

因此需要开启一个全新的加强运行。旧 `checkpoints/best` 可以继续用于对弈和留档，
但旧 replay 不得混入新训练，也不要对 `runs/playable` 使用 `--resume`。

### 19.1 更新云端代码并验证环境

先确认旧训练进程已经退出：

```bash
pgrep -af "chessai train playable" || true
```

若仍有旧训练进程，只在原训练终端按一次 `Ctrl+C` 并等待退出。然后更新代码：

```bash
cd /root/autodl-tmp/ChessAI
source .venv/bin/activate

git status --short
git pull --ff-only
uv pip install -e ".[dev,native]"

cmake -S native -B native/build-linux -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(which python)"
cmake --build native/build-linux --config Release

chessai doctor --workspace /root/autodl-tmp/ChessAI
chessai data validate data/processed/ccpd
uv run pytest tests/ai/test_search.py tests/ai/test_replay.py -q
```

`doctor` 必须输出 `ready_for_playable_training: true`。最后一条测试应验证新的
completed-Q 策略目标和 replay 搜索版本保护。

### 19.2 先运行 5,000 局面性能 pilot

可以使用原监督模型做性能基准；pilot 写入 `artifacts`，不会修改正式运行：

```bash
chessai benchmark selfplay checkpoints/best \
  --positions 5000 \
  --simulations 32 \
  --actors 48 \
  --output artifacts/stronger-selfplay-pilot.json

python -m json.tool artifacts/stronger-selfplay-pilot.json
```

检查 `failures` 下的 `worker_crashes`、`timeouts`、`illegal_moves` 和 `nan_or_inf` 均为
0，48 个 actor 正常存活，并确认 positions/s 持续增长。GPU 利用率不必持续 100%；
真正的门槛是没有错误、动态 batch 正常形成且局面吞吐稳定。

### 19.3 启动独立加强训练

加强配置的固定预算为：

- 监督预热 2 epoch；
- 8 轮强化学习，每轮至少 100,000 个新局面，总计至少 800,000；
- simulations 依次为 `32, 32, 48, 48, 64, 64, 64, 64`；
- 每轮使用最近最多 300,000 个 replay 局面训练 2 epoch；
- 每轮 arena 40 盘、128 simulations；
- 最终 Random 门禁 40 盘、128 simulations。

使用新的输出目录和模型目录：

```bash
cd /root/autodl-tmp/ChessAI
source .venv/bin/activate
mkdir -p runs

chessai train playable data/processed/ccpd \
  --output runs/stronger-v2 \
  --model-dir checkpoints-stronger \
  --config configs/stronger.yaml \
  2>&1 | tee runs/stronger-v2-console.log
```

首次启动不要加 `--resume`。SSH 断开或手动中断后，使用完全相同的配置恢复：

```bash
chessai train playable data/processed/ccpd \
  --output runs/stronger-v2 \
  --model-dir checkpoints-stronger \
  --config configs/stronger.yaml \
  --resume \
  2>&1 | tee -a runs/stronger-v2-console.log
```

### 19.4 观察训练进度和判断是否真正变强

另开一个 SSH 终端查看总状态：

```bash
cd /root/autodl-tmp/ChessAI
python - <<'PY'
import json
from pathlib import Path

path = Path("runs/stronger-v2/state.json")
state = json.loads(path.read_text(encoding="utf-8"))
print("stage:", state["stage"])
print("iteration:", state["iteration"], "/", state["config"]["iterations"])
print("RL positions:", state["rl_generated_positions"])
print("active checkpoint:", state["active_checkpoint"])
print("playable gate:", state["playable_gate_passed"])
print("evaluations:")
for item in state["evaluations"]:
    summary = item["summary"]
    print(
        "  iteration", item["iteration"],
        "accepted=", item["accepted"],
        "W/D/L=", summary.get("wins"), summary.get("draws"), summary.get("losses"),
        "score=", summary.get("score_rate"),
    )
PY
```

self-play 阶段查看当前轮实时吞吐：

```bash
python - <<'PY'
import json
from pathlib import Path

state = json.loads(Path("runs/stronger-v2/state.json").read_text(encoding="utf-8"))
iteration = int(state["iteration"]) + 1
root = Path("runs/stronger-v2") / f"iteration-{iteration:03d}" / "selfplay"
for name in ("runtime.json", "manifest.json"):
    path = root / name
    if not path.is_file():
        print(name, "尚未生成")
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(name)
    print("  games:", payload.get("games", payload.get("completed_games")))
    print("  positions:", payload.get("positions", payload.get("completed_positions")))
    print("  positions/s:", payload.get("positions_per_second"))
    print("  inference:", payload.get("inference"))
PY
```

判断模型是否训练好，优先看棋力证据而不是 loss：

1. `evaluations` 中开始出现 `accepted: true`，并且 `active_checkpoint` 散列发生变化；
2. `checkpoints-stronger/best/metadata.json` 的 `training.kind` 不再是
   `supervised-bootstrap`，而是 RL candidate 对应的训练记录；
3. 多轮 arena 得分率能维持在 0.50 以上，而不是偶然只通过一次；
4. 最终 `playable_gate_passed` 为 true，且报告中零非法着、零引擎崩溃；
5. 下载后再以相同 GUI 难度和固定开局与旧模型直接对弈。Random 门禁只证明基础
   可玩性，不等于达到《天天象棋》业余段位。

如果前 2 轮仍然全部被大比分拒绝（例如得分率长期低于 0.40），先停止训练并保留
日志、arena 报告和 checkpoint 排查，不建议盲目烧完 8 轮。

### 19.5 导出并下载加强模型

训练完成后导出新的 best；不要误导出旧 `checkpoints/best`：

```bash
chessai export \
  checkpoints-stronger/best \
  artifacts/chessai-stronger-v2 \
  --data-manifest data/processed/ccpd/manifest.json \
  --evaluation-report runs/stronger-v2/final-evaluation.json

tar -czf artifacts/chessai-stronger-v2.tar.gz \
  -C artifacts chessai-stronger-v2
sha256sum artifacts/chessai-stronger-v2.tar.gz \
  | tee artifacts/chessai-stronger-v2.tar.gz.sha256
```

将 `artifacts/chessai-stronger-v2.tar.gz` 和对应 `.sha256` 下载到 Windows：

```powershell
Set-Location E:\AI_project\ChessAI
New-Item -ItemType Directory -Force .\artifacts\cloud-download | Out-Null

scp -P SSH端口 `
  云端用户名@云端地址:/root/autodl-tmp/ChessAI/artifacts/chessai-stronger-v2.tar.gz `
  .\artifacts\cloud-download\
scp -P SSH端口 `
  云端用户名@云端地址:/root/autodl-tmp/ChessAI/artifacts/chessai-stronger-v2.tar.gz.sha256 `
  .\artifacts\cloud-download\
```

校验、安装并启动 GUI：

```powershell
Get-FileHash -Algorithm SHA256 `
  .\artifacts\cloud-download\chessai-stronger-v2.tar.gz
Get-Content `
  .\artifacts\cloud-download\chessai-stronger-v2.tar.gz.sha256

tar -xzf .\artifacts\cloud-download\chessai-stronger-v2.tar.gz `
  -C .\checkpoints

$env:CHESSAI_MODEL_DIR = "E:\AI_project\ChessAI\checkpoints"
uv run chessai serve --host 127.0.0.1 --port 8000
```

GUI 会优先选择首个兼容的策略价值 checkpoint；如果本地同时存在多个模型，仍应在
“对弈模型”下拉框中确认选择的是 `chessai-stronger-v2`。
