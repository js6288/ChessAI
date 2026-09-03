"""Bounded supervised policy/value bootstrap training."""

from __future__ import annotations

import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from chessai.ai.model import ModelConfig, PolicyValueModel
from chessai.data.manifest import sha256_file
from chessai.training.checkpoint import load_checkpoint, restore_training_state, save_checkpoint
from chessai.training.examples import PHASE_NAMES, JsonlEvaluationDataset, JsonlPositionDataset
from chessai.training.metrics import MetricsWriter

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("install the 'train' extra to run bootstrap training") from exc


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    num_workers: int = 0
    max_games: int | None = None
    max_steps: int | None = None
    device: str = "auto"
    precision: str = "auto"
    seed: int = 20260902

    @classmethod
    def tiny(cls) -> BootstrapConfig:
        return cls(batch_size=8, max_games=2, max_steps=2, device="cpu")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but the installed PyTorch build cannot use it")
    return device


def resolve_precision(device: torch.device, requested: str) -> str:
    if requested not in {"auto", "fp32", "bf16"}:
        raise ValueError("precision must be auto, fp32, or bf16")
    bf16_supported = device.type == "cuda" and bool(torch.cuda.is_bf16_supported())
    if requested == "bf16" and not bf16_supported:
        raise RuntimeError("BF16 training was requested but is unavailable on this device/build")
    if requested == "auto":
        return "bf16" if bf16_supported else "fp32"
    return requested


def autocast_context(device: torch.device, precision: str) -> AbstractContextManager[Any]:
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def evaluate_supervised(
    model: PolicyValueModel,
    dataset_path: str | Path,
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
    num_workers: int = 0,
    max_games: int | None = None,
) -> dict[str, Any]:
    """Measure policy/value quality with the same legal mask used by search."""

    dataset = JsonlEvaluationDataset(dataset_path, max_games=max_games)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    was_training = model.training
    model.eval()
    total = 0
    raw_top1 = 0
    legal_hits = [0, 0, 0]
    squared_error = 0.0
    absolute_error = 0.0
    phase_totals = [0, 0, 0]
    phase_hits = [[0, 0, 0] for _ in PHASE_NAMES]
    phase_squared_error = [0.0, 0.0, 0.0]
    try:
        for features, actions, values, legal_masks, phases in loader:
            features = features.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            legal_masks = legal_masks.to(device, non_blocking=True)
            with autocast_context(device, precision):
                logits, predicted_values = model(features)
            logits = logits.float()
            predicted_values = predicted_values.float()
            if not torch.isfinite(logits).all() or not torch.isfinite(predicted_values).all():
                raise FloatingPointError("non-finite supervised evaluation output")
            masked_logits = logits.masked_fill(~legal_masks, torch.finfo(logits.dtype).min)
            raw_top1_batch = logits.argmax(dim=-1).eq(actions)
            ranked = masked_logits.topk(k=5, dim=-1).indices
            hit_matrix = ranked.eq(actions.unsqueeze(-1))
            top_hits = [hit_matrix[:, :limit].any(dim=-1) for limit in (1, 3, 5)]
            value_error = predicted_values - values

            batch_size_actual = int(features.shape[0])
            total += batch_size_actual
            raw_top1 += int(raw_top1_batch.sum().item())
            for index, hit in enumerate(top_hits):
                legal_hits[index] += int(hit.sum().item())
            squared_error += float(value_error.square().sum().item())
            absolute_error += float(value_error.abs().sum().item())

            phase_cpu = phases.numpy()
            hits_cpu = [hit.cpu().numpy() for hit in top_hits]
            squared_cpu = value_error.square().cpu().numpy()
            for phase_index in range(len(PHASE_NAMES)):
                selected = phase_cpu == phase_index
                count = int(selected.sum())
                phase_totals[phase_index] += count
                if not count:
                    continue
                for top_index, hit in enumerate(hits_cpu):
                    phase_hits[phase_index][top_index] += int(hit[selected].sum())
                phase_squared_error[phase_index] += float(squared_cpu[selected].sum())
    finally:
        if was_training:
            model.train()

    if total == 0:
        raise ValueError(f"supervised evaluation dataset is empty: {dataset_path}")

    def ratio(numerator: float, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    phases_report = {
        name: {
            "positions": phase_totals[index],
            "legal_top1_accuracy": ratio(phase_hits[index][0], phase_totals[index]),
            "legal_top3_accuracy": ratio(phase_hits[index][1], phase_totals[index]),
            "legal_top5_accuracy": ratio(phase_hits[index][2], phase_totals[index]),
            "value_mse": ratio(phase_squared_error[index], phase_totals[index]),
        }
        for index, name in enumerate(PHASE_NAMES)
    }
    return {
        "dataset": str(Path(dataset_path)),
        "positions": total,
        "raw_top1_accuracy": ratio(raw_top1, total),
        "legal_top1_accuracy": ratio(legal_hits[0], total),
        "legal_top3_accuracy": ratio(legal_hits[1], total),
        "legal_top5_accuracy": ratio(legal_hits[2], total),
        "value_mse": ratio(squared_error, total),
        "value_mae": ratio(absolute_error, total),
        "phases": phases_report,
    }


def run_bootstrap(
    train_jsonl: str | Path,
    output_dir: str | Path,
    *,
    config: BootstrapConfig | None = None,
    model_config: ModelConfig | None = None,
    initial_model: PolicyValueModel | None = None,
    checkpoint: str | Path | None = None,
    resume: bool = False,
    validation_jsonl: str | Path | None = None,
    test_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or BootstrapConfig()
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    precision = resolve_precision(device, cfg.precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.reset_peak_memory_stats(device)
    if initial_model is not None and checkpoint is not None:
        raise ValueError("initial_model and checkpoint are mutually exclusive")
    if resume and checkpoint is None:
        raise ValueError("resume requires a checkpoint")
    loaded = load_checkpoint(checkpoint, device=device) if checkpoint is not None else None
    model = (
        loaded.model
        if loaded is not None
        else initial_model or PolicyValueModel(model_config or ModelConfig())
    )
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    previous_training = loaded.metadata.get("training", {}) if loaded is not None else {}
    if resume and loaded is not None:
        restore_training_state(loaded, optimizer=optimizer)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = MetricsWriter(output / "metrics.jsonl")
    global_step = int(previous_training.get("steps", 0)) if resume else 0
    positions = int(previous_training.get("positions", 0)) if resume else 0
    invocation_steps = 0
    invocation_positions = 0
    started = time.perf_counter()
    final_loss = 0.0

    for epoch in range(cfg.epochs):
        dataset = JsonlPositionDataset(train_jsonl, max_games=cfg.max_games)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
        )
        for features, actions, values in loader:
            features = features.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                logits, predicted_values = model(features)
            policy_loss = policy_loss_fn(logits.float(), actions)
            value_loss = value_loss_fn(predicted_values.float(), values)
            loss = policy_loss + cfg.value_loss_weight * value_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite bootstrap loss at step {global_step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            global_step += 1
            invocation_steps += 1
            positions += int(features.shape[0])
            invocation_positions += int(features.shape[0])
            final_loss = float(loss.item())
            accuracy = float((logits.argmax(dim=-1) == actions).float().mean().item())
            metrics.write(
                "bootstrap_step",
                epoch=epoch,
                step=global_step,
                positions=positions,
                loss=final_loss,
                policy_loss=float(policy_loss.item()),
                value_loss=float(value_loss.item()),
                policy_accuracy=accuracy,
            )
            if cfg.max_steps is not None and invocation_steps >= cfg.max_steps:
                break
        if cfg.max_steps is not None and invocation_steps >= cfg.max_steps:
            break

    elapsed = time.perf_counter() - started
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation: dict[str, Any] = {}
    for split, path in (("validation", validation_jsonl), ("test", test_jsonl)):
        if path is None:
            continue
        evaluation[split] = evaluate_supervised(
            model,
            path,
            device=device,
            precision=precision,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            max_games=cfg.max_games,
        )
        metrics.write("bootstrap_evaluation", split=split, **evaluation[split])
    dataset_contract = {
        split: {
            "path": str(Path(path)),
            "sha256": sha256_file(Path(path)),
            "bytes": Path(path).stat().st_size,
        }
        for split, path in (
            ("train", train_jsonl),
            ("validation", validation_jsonl),
            ("test", test_jsonl),
        )
        if path is not None
    }
    summary = {
        "kind": "supervised-bootstrap",
        "steps": global_step,
        "positions": positions,
        "invocation_steps": invocation_steps,
        "invocation_positions": invocation_positions,
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
        "datasets": dataset_contract,
        "evaluation": evaluation,
        "config": asdict(cfg),
    }
    metrics.write("bootstrap_complete", **summary)
    save_checkpoint(
        output / "bootstrap-best",
        model,
        training=summary,
        metrics={"final_loss": final_loss, "evaluation": evaluation},
        optimizer=optimizer,
    )
    return summary
