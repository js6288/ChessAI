"""Compatible model discovery for the local workbench."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chessai.ai.evaluator import TorchEvaluator
from chessai.ai.search import Evaluator, HeuristicEvaluator
from chessai.training.checkpoint import Compatibility, load_checkpoint


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    id: str
    name: str
    kind: str
    compatible: bool
    checkpoint: Path | None = None
    metadata: dict[str, Any] | None = None
    error: str | None = None

    def public(self) -> dict[str, Any]:
        compatibility = (self.metadata or {}).get("compatibility", {})
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "compatible": self.compatible,
            "error": self.error,
            "compatibility": compatibility,
        }


class ModelRegistry:
    def __init__(self, directory: str | Path = "checkpoints", *, device: str = "cpu") -> None:
        self.directory = Path(directory)
        self.device = device
        self._descriptors: dict[str, ModelDescriptor] = {}
        self._evaluators: dict[str, Evaluator] = {"heuristic": HeuristicEvaluator()}
        self._lock = threading.Lock()
        self.refresh()

    def refresh(self) -> list[ModelDescriptor]:
        descriptors = {
            "heuristic": ModelDescriptor(
                id="heuristic",
                name="墨衡 · 启发式演示",
                kind="heuristic",
                compatible=True,
                metadata={
                    "compatibility": {},
                },
            )
        }
        if self.directory.is_dir():
            for metadata_path in sorted(self.directory.rglob("metadata.json")):
                checkpoint = metadata_path.parent
                model_id = checkpoint.relative_to(self.directory).as_posix().replace("/", "--")
                try:
                    import json

                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    Compatibility(**metadata["compatibility"]).validate_current()
                    descriptor = ModelDescriptor(
                        id=model_id,
                        name=metadata.get("training", {}).get("name", checkpoint.name),
                        kind="policy-value",
                        compatible=True,
                        checkpoint=checkpoint,
                        metadata=metadata,
                    )
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    descriptor = ModelDescriptor(
                        id=model_id,
                        name=checkpoint.name,
                        kind="policy-value",
                        compatible=False,
                        checkpoint=checkpoint,
                        error=str(exc),
                    )
                descriptors[model_id] = descriptor
        self._descriptors = descriptors
        return list(descriptors.values())

    def list(self) -> list[dict[str, Any]]:
        return [descriptor.public() for descriptor in self._descriptors.values()]

    def evaluator(self, model_id: str) -> Evaluator:
        descriptor = self._descriptors.get(model_id)
        if descriptor is None:
            self.refresh()
            descriptor = self._descriptors.get(model_id)
        if descriptor is None:
            raise KeyError(f"unknown model: {model_id}")
        if not descriptor.compatible:
            raise ValueError(descriptor.error or f"model {model_id} is incompatible")
        with self._lock:
            cached = self._evaluators.get(model_id)
            if cached is not None:
                return cached
            assert descriptor.checkpoint is not None
            loaded = load_checkpoint(descriptor.checkpoint, device=self.device)
            evaluator = TorchEvaluator(loaded.model, device=self.device)
            self._evaluators[model_id] = evaluator
            return evaluator
