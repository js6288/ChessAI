"""Replay-buffer policy/value optimization for self-play iterations."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from chessai.ai.model import ModelConfig, PolicyValueModel
from chessai.data.manifest import sha256_file
from chessai.training.bootstrap import autocast_context, resolve_device, resolve_precision
from chessai.training.checkpoint import load_checkpoint, restore_training_state, save_checkpoint
from chessai.training.metrics import MetricsWriter
from chessai.training.replay import ReplayDataset

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("install the 'train' extra to run RL optimization") from exc


@dataclass(frozen=True, slots=True)
class RlTrainConfig:
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    replay_capacity: int = 300_000
    epochs: int = 1
    max_steps: int | None = None
    device: str = "auto"
    precision: str = "auto"
    seed: int = 20260902

    @classmethod
    def tiny(cls) -> RlTrainConfig:
        return cls(batch_size=8, replay_capacity=256, max_steps=2, device="cpu")


def run_rl_training(
    replay_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    resume: bool = False,
    config: RlTrainConfig | None = None,
    model_config: ModelConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RlTrainConfig()
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    precision = resolve_precision(device, cfg.precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.reset_peak_memory_stats(device)
    loaded = None
    if checkpoint is None:
        if resume:
            raise ValueError("resume requires a checkpoint")
        model = PolicyValueModel(model_config or ModelConfig())
    else:
        loaded = load_checkpoint(checkpoint, device=device)
        model = loaded.model
    model.to(device).train()
    dataset = ReplayDataset(replay_paths, capacity=cfg.replay_capacity)
    if not dataset:
        raise ValueError("replay dataset is empty")
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    previous_training = loaded.metadata.get("training", {}) if loaded is not None else {}
    if resume and loaded is not None:
        restore_training_state(loaded, optimizer=optimizer)
    value_loss_fn = nn.MSELoss()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = MetricsWriter(output / "metrics.jsonl")
    step = int(previous_training.get("steps", 0)) if resume else 0
    positions = int(previous_training.get("positions", 0)) if resume else 0
    invocation_steps = 0
    invocation_positions = 0
    final_loss = 0.0
    started = time.perf_counter()
    for epoch in range(cfg.epochs):
        for features, target_policy, target_value in loader:
            features = features.to(device, non_blocking=True)
            target_policy = target_policy.to(device, non_blocking=True)
            target_value = target_value.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                logits, predicted_value = model(features)
            log_policy = torch.log_softmax(logits.float(), dim=-1)
            policy_loss = -(target_policy * log_policy).sum(dim=-1).mean()
            value_loss = value_loss_fn(predicted_value.float(), target_value)
            loss = policy_loss + cfg.value_loss_weight * value_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite RL loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            step += 1
            invocation_steps += 1
            positions += int(features.shape[0])
            invocation_positions += int(features.shape[0])
            final_loss = float(loss.item())
            metrics.write(
                "rl_step",
                epoch=epoch,
                step=step,
                positions=positions,
                loss=final_loss,
                policy_loss=float(policy_loss.item()),
                value_loss=float(value_loss.item()),
            )
            if cfg.max_steps is not None and invocation_steps >= cfg.max_steps:
                break
        if cfg.max_steps is not None and invocation_steps >= cfg.max_steps:
            break
    elapsed = time.perf_counter() - started
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    replay_contract = [
        {
            "path": str(Path(path)),
            "sha256": sha256_file(Path(path)),
            "bytes": Path(path).stat().st_size,
        }
        for path in dataset.shard_paths
    ]
    summary = {
        "kind": "gumbel-alphazero-optimization",
        "steps": step,
        "positions": positions,
        "invocation_steps": invocation_steps,
        "invocation_positions": invocation_positions,
        "replay_positions": len(dataset),
        "elapsed_seconds": elapsed,
        "positions_per_second": invocation_positions / elapsed if elapsed else 0.0,
        "final_loss": final_loss,
        "device": str(device),
        "precision": precision,
        "peak_allocated_mb": (
            round(torch.cuda.max_memory_allocated(device) / 2**20, 2)
            if device.type == "cuda"
            else None
        ),
        "resumed": resume,
        "parent_checkpoint": str(checkpoint) if checkpoint is not None else None,
        "replay_shards": replay_contract,
        "config": asdict(cfg),
    }
    metrics.write("rl_complete", **summary)
    save_checkpoint(
        output / "candidate",
        model,
        training=summary,
        metrics={"final_loss": final_loss},
        optimizer=optimizer,
    )
    return summary
