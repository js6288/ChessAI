"""Safe, version-checked policy/value checkpoints."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file

from chessai.ai.model import ModelConfig, PolicyValueModel
from chessai.compat import (
    ACTION_VOCAB_VERSION,
    FEATURE_VERSION,
    RULE_VERSION,
    SCHEMA_VERSION,
)
from chessai.data.manifest import sha256_file
from chessai.engine.vocabulary import action_vocab_hash

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("install the 'train' extra to use checkpoints") from exc


@dataclass(frozen=True, slots=True)
class Compatibility:
    schema_version: str = SCHEMA_VERSION
    rule_version: str = RULE_VERSION
    feature_version: str = FEATURE_VERSION
    action_vocab_version: str = ACTION_VOCAB_VERSION
    action_vocab_hash: str = ""

    def __post_init__(self) -> None:
        if not self.action_vocab_hash:
            object.__setattr__(self, "action_vocab_hash", action_vocab_hash())

    def validate_current(self) -> None:
        expected = Compatibility()
        mismatches = [
            f"{key}: checkpoint={value!r}, runtime={getattr(expected, key)!r}"
            for key, value in asdict(self).items()
            if value != getattr(expected, key)
        ]
        if mismatches:
            raise ValueError("incompatible checkpoint: " + "; ".join(mismatches))


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    model: PolicyValueModel
    metadata: dict[str, Any]
    weights_path: Path
    training_state_path: Path | None


def _paths(target: Path) -> tuple[Path, Path]:
    if target.suffix == ".safetensors":
        return target, target.with_suffix(".json")
    return target / "weights.safetensors", target / "metadata.json"


def save_checkpoint(
    target: str | Path,
    model: PolicyValueModel,
    *,
    training: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[Path, Path]:
    destination = Path(target)
    weights_path, metadata_path = _paths(destination)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_weights = weights_path.with_name(weights_path.name + ".tmp")
    temporary_metadata = metadata_path.with_name(metadata_path.name + ".tmp")
    save_file(model.state_dict(), temporary_weights)
    weights_hash = sha256_file(temporary_weights)
    metadata: dict[str, Any] = {
        "compatibility": asdict(Compatibility()),
        "created_at": datetime.now(UTC).isoformat(),
        "model": model.metadata(),
        "weights": {"file": weights_path.name, "sha256": weights_hash},
        "training": training or {},
        "metrics": metrics or {},
        "training_state": {
            "file": "training-state.pt",
            "contains_optimizer": optimizer is not None,
            "contains_torch_rng": True,
            "contains_cuda_rng": torch.cuda.is_available(),
        },
    }
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    training_state_path = weights_path.parent / "training-state.pt"
    temporary_training_state = training_state_path.with_name(training_state_path.name + ".tmp")
    state: dict[str, Any] = {
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    torch.save(state, temporary_training_state)
    os.replace(temporary_weights, weights_path)
    os.replace(temporary_training_state, training_state_path)
    # Metadata is the commit marker and is published only after every declared
    # checkpoint payload has reached its final path.
    os.replace(temporary_metadata, metadata_path)
    return weights_path, metadata_path


def load_checkpoint(
    source: str | Path,
    *,
    device: str | torch.device = "cpu",
    validate: bool = True,
) -> LoadedCheckpoint:
    weights_path, metadata_path = _paths(Path(source))
    if metadata_path.is_file() and not weights_path.is_file():
        preview = json.loads(metadata_path.read_text(encoding="utf-8"))
        declared_file = (
            preview.get("weights", {}).get("file") if isinstance(preview, dict) else None
        )
        if isinstance(declared_file, str) and Path(declared_file).name == declared_file:
            weights_path = metadata_path.parent / declared_file
    if not weights_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint needs {weights_path} and {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    declared_weights_hash = metadata.get("weights", {}).get("sha256")
    if declared_weights_hash is not None:
        actual_weights_hash = sha256_file(weights_path)
        if actual_weights_hash != declared_weights_hash:
            raise ValueError(
                "checkpoint weights hash mismatch: "
                f"expected {declared_weights_hash}, got {actual_weights_hash}"
            )
    compatibility = Compatibility(**metadata["compatibility"])
    if validate:
        compatibility.validate_current()
    config = ModelConfig(**metadata["model"]["config"])
    model = PolicyValueModel(config)
    state_dict = load_file(weights_path, device=str(device))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    declared_training_state = metadata.get("training_state", {}).get("file")
    training_state_path = (
        weights_path.parent / str(declared_training_state) if declared_training_state else None
    )
    if training_state_path is not None and not training_state_path.is_file():
        raise FileNotFoundError(f"checkpoint training state is missing: {training_state_path}")
    return LoadedCheckpoint(
        model=model,
        metadata=metadata,
        weights_path=weights_path,
        training_state_path=training_state_path,
    )


def restore_training_state(
    checkpoint: LoadedCheckpoint,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Restore trusted optimizer and RNG state for an interrupted local run."""

    if checkpoint.training_state_path is None:
        raise ValueError("checkpoint does not declare a resumable training state")
    state = torch.load(checkpoint.training_state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("checkpoint training state must be a mapping")
    optimizer_state = state.get("optimizer")
    if optimizer is not None:
        if optimizer_state is None:
            raise ValueError("checkpoint does not contain optimizer state")
        optimizer.load_state_dict(optimizer_state)
    torch_state = state.get("torch_rng_state")
    if not isinstance(torch_state, torch.Tensor):
        raise ValueError("checkpoint does not contain a valid torch RNG state")
    torch.set_rng_state(torch_state)
    cuda_states = state.get("cuda_rng_state_all", [])
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)
    return state
