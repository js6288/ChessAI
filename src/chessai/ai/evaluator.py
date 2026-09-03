"""Adapters between PyTorch models and the search evaluator protocol."""

from __future__ import annotations

import queue
import random
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from chessai.ai.features import encode_state
from chessai.ai.model import PolicyValueModel
from chessai.engine import GameState

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("install the 'train' extra to use TorchEvaluator") from exc


def _use_bf16(device: torch.device, precision: str) -> bool:
    if precision not in {"auto", "fp32", "bf16"}:
        raise ValueError("precision must be auto, fp32, or bf16")
    supported = device.type == "cuda" and bool(torch.cuda.is_bf16_supported())
    if precision == "bf16" and not supported:
        raise RuntimeError("BF16 inference was requested but is unavailable on this device/build")
    return supported and precision in {"auto", "bf16"}


class TorchEvaluator:
    def __init__(
        self,
        model: PolicyValueModel,
        device: str | torch.device = "cpu",
        *,
        precision: str = "auto",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.use_bf16 = _use_bf16(self.device, precision)

    def evaluate(self, state: GameState) -> tuple[npt.NDArray[np.float64], float]:
        features = torch.from_numpy(encode_state(state)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            if self.use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, value = self.model(features)
            else:
                logits, value = self.model(features)
        return logits[0].float().cpu().numpy().astype(np.float64), float(value[0].item())


class BatchedTorchEvaluator:
    """Explicit batch interface used by cloud inference workers."""

    def __init__(
        self,
        model: PolicyValueModel,
        device: str | torch.device = "cpu",
        *,
        precision: str = "auto",
        channels_last: bool = False,
        max_batch_size: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.channels_last = channels_last
        if channels_last:
            self.model = self.model.to(  # type: ignore[call-overload]
                memory_format=torch.channels_last
            )
        self.use_bf16 = _use_bf16(self.device, precision)
        self._host_buffer = (
            torch.empty(
                (max_batch_size, 117, 10, 9),
                dtype=torch.float32,
                pin_memory=True,
            )
            if self.device.type == "cuda" and max_batch_size is not None
            else None
        )

    def _forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return cast(tuple[torch.Tensor, torch.Tensor], self.model(batch))
        return cast(tuple[torch.Tensor, torch.Tensor], self.model(batch))

    @torch.inference_mode()
    def evaluate_many(
        self, states: list[GameState]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        if not states:
            return (
                np.empty((0, self.model.config.action_size), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        return self.evaluate_features(np.stack([encode_state(state) for state in states]))

    @torch.inference_mode()
    def evaluate_features(
        self, features: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Evaluate an already encoded contiguous NCHW feature batch."""

        if features.ndim != 4 or features.shape[1:] != (117, 10, 9):
            raise ValueError(f"unexpected feature batch shape: {features.shape}")
        if features.shape[0] == 0:
            return (
                np.empty((0, self.model.config.action_size), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        contiguous = np.ascontiguousarray(features, dtype=np.float32)
        if self._host_buffer is not None:
            if contiguous.shape[0] > self._host_buffer.shape[0]:
                raise ValueError("feature batch exceeds the preallocated pinned buffer")
            host = self._host_buffer[: contiguous.shape[0]]
            host.copy_(torch.from_numpy(contiguous))
        else:
            host = torch.from_numpy(contiguous)
            if self.device.type == "cuda":
                host = host.pin_memory()
        batch = host.to(self.device, non_blocking=self.device.type == "cuda")
        if self.channels_last:
            batch = batch.contiguous(memory_format=torch.channels_last)
        logits, values = self._forward(batch)
        return logits.float().cpu().numpy(), values.float().cpu().numpy()


@dataclass(slots=True)
class _InferenceRequest:
    state: GameState
    future: Future[tuple[npt.NDArray[np.float64], float]]
    enqueued_at: float


class BatchingEvaluator:
    """Synchronous evaluator facade backed by one shared GPU batching thread.

    Concurrent search actors block on individual futures while the inference
    worker combines their leaf requests into one model call. Only the worker
    touches the model, avoiding concurrent CUDA launches from actor threads.
    """

    def __init__(
        self,
        model: PolicyValueModel,
        device: str | torch.device = "cpu",
        *,
        max_batch_size: int = 64,
        max_wait_ms: float = 2.0,
        precision: str = "auto",
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")
        self._backend = BatchedTorchEvaluator(model, device, precision=precision)
        self._max_batch_size = max_batch_size
        self._max_wait_seconds = max_wait_ms / 1000.0
        self._requests: queue.Queue[_InferenceRequest | None] = queue.Queue()
        self._closed = False
        self._state_lock = threading.Lock()
        self._statistics: dict[str, float | int | str] = {
            "requests": 0,
            "batches": 0,
            "largest_batch": 0,
            "model_seconds": 0.0,
            "precision": "bf16" if self._backend.use_bf16 else "fp32",
        }
        self._latency_samples: dict[str, list[float]] = {
            "batch_model_ms": [],
            "request_ms": [],
        }
        self._latency_seen = {name: 0 for name in self._latency_samples}
        self._latency_rng = random.Random(20260902)
        self._thread = threading.Thread(
            target=self._serve,
            name="chessai-batched-inference",
            daemon=True,
        )
        self._thread.start()

    def evaluate(self, state: GameState) -> tuple[npt.NDArray[np.float64], float]:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("batching evaluator is closed")
            future: Future[tuple[npt.NDArray[np.float64], float]] = Future()
            self._requests.put(_InferenceRequest(state, future, time.perf_counter()))
        return future.result()

    def stats(self) -> dict[str, float | int | str]:
        with self._state_lock:
            result = dict(self._statistics)
        batches = int(result["batches"])
        requests = int(result["requests"])
        result["mean_batch_size"] = requests / batches if batches else 0.0
        for name, values in self._latency_samples.items():
            if values:
                result[f"{name}_p50"] = float(np.percentile(values, 50))
                result[f"{name}_p95"] = float(np.percentile(values, 95))
                result[f"{name}_samples"] = len(values)
        return result

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._requests.put(None)
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            raise RuntimeError("batched inference worker did not stop")

    def __enter__(self) -> BatchingEvaluator:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _serve(self) -> None:
        stop_after_batch = False
        while not stop_after_batch:
            first = self._requests.get()
            if first is None:
                break
            requests = [first]
            deadline = time.perf_counter() + self._max_wait_seconds
            while len(requests) < self._max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self._requests.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    stop_after_batch = True
                    break
                requests.append(item)

            started = time.perf_counter()
            try:
                logits, values = self._backend.evaluate_many(
                    [request.state for request in requests]
                )
            except BaseException as exc:  # propagate model/device failures to every actor
                for request in requests:
                    request.future.set_exception(exc)
            else:
                for index, request in enumerate(requests):
                    request.future.set_result(
                        (logits[index].astype(np.float64), float(values[index]))
                    )
            elapsed = time.perf_counter() - started
            finished = time.perf_counter()
            with self._state_lock:
                self._statistics["requests"] = int(self._statistics["requests"]) + len(requests)
                self._statistics["batches"] = int(self._statistics["batches"]) + 1
                self._statistics["largest_batch"] = max(
                    int(self._statistics["largest_batch"]), len(requests)
                )
                self._statistics["model_seconds"] = (
                    float(self._statistics["model_seconds"]) + elapsed
                )
                self._record_latency("batch_model_ms", elapsed * 1000.0)
                for request in requests:
                    self._record_latency("request_ms", (finished - request.enqueued_at) * 1000.0)

    def _record_latency(self, name: str, value: float, capacity: int = 10_000) -> None:
        """Keep a bounded deterministic reservoir for p50/p95 reporting."""

        self._latency_seen[name] += 1
        seen = self._latency_seen[name]
        values = self._latency_samples[name]
        if len(values) < capacity:
            values.append(value)
            return
        replacement = self._latency_rng.randrange(seen)
        if replacement < capacity:
            values[replacement] = value
