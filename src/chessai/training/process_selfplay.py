"""Spawn-based self-play actors with one shared batched model evaluator."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import random
import threading
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np
import numpy.typing as npt

from chessai.ai.evaluator import BatchedTorchEvaluator
from chessai.ai.features import INPUT_PLANES, encode_state
from chessai.ai.model import PolicyValueModel
from chessai.engine import GameState
from chessai.training.replay import PackedReplayBatch, pack_replay_samples

ACTION_SIZE = 2086
FEATURE_SHAPE = (INPUT_PLANES, 10, 9)


@dataclass(frozen=True, slots=True)
class ProcessRuntimeConfig:
    actors: int
    max_batch_size: int
    min_batch_size: int
    wait_ms: float
    timeout_seconds: float
    device: str
    precision: str
    channels_last: bool


@dataclass(frozen=True, slots=True)
class ProcessGameResult:
    game_index: int
    seed: int
    replay: PackedReplayBatch
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _ActorResult:
    actor_id: int
    game_index: int
    seed: int
    replay: PackedReplayBatch | None
    summary: dict[str, Any] | None
    error: str | None


class _SharedMemoryEvaluator:
    """Actor-side synchronous evaluator using one fixed shared-memory slot."""

    def __init__(
        self,
        *,
        actor_id: int,
        input_name: str,
        output_name: str,
        value_name: str,
        sequence_name: str,
        heartbeat_name: str,
        actors: int,
        requests: Any,
        response_event: Any,
        timeout_seconds: float,
    ) -> None:
        self.actor_id = actor_id
        self._input_memory = SharedMemory(name=input_name)
        self._output_memory = SharedMemory(name=output_name)
        self._value_memory = SharedMemory(name=value_name)
        self._sequence_memory = SharedMemory(name=sequence_name)
        self._heartbeat_memory = SharedMemory(name=heartbeat_name)
        self._inputs = np.ndarray(
            (actors, *FEATURE_SHAPE), dtype=np.float32, buffer=self._input_memory.buf
        )
        self._outputs = np.ndarray(
            (actors, ACTION_SIZE), dtype=np.float32, buffer=self._output_memory.buf
        )
        self._values = np.ndarray((actors,), dtype=np.float32, buffer=self._value_memory.buf)
        self._responses = np.ndarray(
            (actors,), dtype=np.int64, buffer=self._sequence_memory.buf
        )
        self._heartbeats = np.ndarray(
            (actors,), dtype=np.float64, buffer=self._heartbeat_memory.buf
        )
        self._requests = requests
        self._event = response_event
        self._timeout_seconds = timeout_seconds
        self._sequence = 0
        self.requests = 0
        self.feature_seconds = 0.0
        self.inference_wait_seconds = 0.0

    def evaluate(self, state: GameState) -> tuple[npt.NDArray[np.float32], float]:
        encoded_at = time.perf_counter()
        self._heartbeats[self.actor_id] = time.time()
        self._inputs[self.actor_id] = encode_state(state)
        self.feature_seconds += time.perf_counter() - encoded_at
        self._sequence += 1
        sequence = self._sequence
        self._event.clear()
        enqueued_at = time.perf_counter()
        self._requests.put((self.actor_id, sequence, enqueued_at))
        if not self._event.wait(self._timeout_seconds):
            raise TimeoutError(
                f"actor {self.actor_id} timed out waiting for inference sequence {sequence}"
            )
        self.inference_wait_seconds += time.perf_counter() - enqueued_at
        response_sequence = int(self._responses[self.actor_id])
        if response_sequence != sequence:
            raise RuntimeError(
                f"actor {self.actor_id} received stale inference response: "
                f"expected={sequence}, got={response_sequence}"
            )
        logits = self._outputs[self.actor_id].copy()
        value = float(self._values[self.actor_id])
        if not np.all(np.isfinite(logits)) or not np.isfinite(value):
            raise ValueError(f"actor {self.actor_id} received NaN or Inf model output")
        self.requests += 1
        self._heartbeats[self.actor_id] = time.time()
        return logits, value

    def close(self) -> None:
        self._input_memory.close()
        self._output_memory.close()
        self._value_memory.close()
        self._sequence_memory.close()
        self._heartbeat_memory.close()


def _actor_main(
    actor_id: int,
    runtime: ProcessRuntimeConfig,
    input_name: str,
    output_name: str,
    value_name: str,
    sequence_name: str,
    heartbeat_name: str,
    requests: Any,
    jobs: Any,
    results: Any,
    response_event: Any,
    simulations: int,
    sample_until_ply: int,
    max_ply: int,
) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass
    evaluator = _SharedMemoryEvaluator(
        actor_id=actor_id,
        input_name=input_name,
        output_name=output_name,
        value_name=value_name,
        sequence_name=sequence_name,
        heartbeat_name=heartbeat_name,
        actors=runtime.actors,
        requests=requests,
        response_event=response_event,
        timeout_seconds=runtime.timeout_seconds,
    )
    try:
        while True:
            job = jobs.get()
            if job is None:
                return
            game_index, seed = job
            cpu_started = time.process_time()
            requests_before = evaluator.requests
            feature_before = evaluator.feature_seconds
            inference_before = evaluator.inference_wait_seconds
            try:
                # Import here so a spawned process does not create a circular module import.
                from chessai.training.selfplay import play_selfplay_game

                samples, summary = play_selfplay_game(
                    evaluator,
                    simulations=simulations,
                    sample_until_ply=sample_until_ply,
                    max_ply=max_ply,
                    seed=seed,
                )
                feature_seconds = evaluator.feature_seconds - feature_before
                inference_wait_seconds = evaluator.inference_wait_seconds - inference_before
                summary.update(
                    {
                        "actor_id": actor_id,
                        "actor_cpu_seconds": time.process_time() - cpu_started,
                        "inference_requests": evaluator.requests - requests_before,
                        "feature_encoding_seconds": feature_seconds,
                        "inference_wait_seconds": inference_wait_seconds,
                        "search_and_rules_seconds": max(
                            0.0,
                            float(summary["elapsed_seconds"])
                            - feature_seconds
                            - inference_wait_seconds,
                        ),
                    }
                )
                results.put(
                    _ActorResult(
                        actor_id,
                        game_index,
                        seed,
                        pack_replay_samples(samples),
                        summary,
                        None,
                    )
                )
            except BaseException:
                results.put(
                    _ActorResult(
                        actor_id,
                        game_index,
                        seed,
                        None,
                        None,
                        traceback.format_exc(),
                    )
                )
                return
    finally:
        evaluator.close()


class _InferenceServer:
    def __init__(
        self,
        model: PolicyValueModel,
        runtime: ProcessRuntimeConfig,
        requests: Any,
        response_events: list[Any],
        inputs: npt.NDArray[np.float32],
        outputs: npt.NDArray[np.float32],
        values: npt.NDArray[np.float32],
        responses: npt.NDArray[np.int64],
    ) -> None:
        self._backend = BatchedTorchEvaluator(
            model,
            runtime.device,
            precision=runtime.precision,
            channels_last=runtime.channels_last,
            max_batch_size=runtime.max_batch_size,
        )
        self._runtime = runtime
        self._requests = requests
        self._events = response_events
        self._inputs = inputs
        self._outputs = outputs
        self._values = values
        self._responses = responses
        self._lock = threading.Lock()
        self._statistics: dict[str, float | int | str] = {
            "requests": 0,
            "batches": 0,
            "largest_batch": 0,
            "model_seconds": 0.0,
            "precision": "bf16" if self._backend.use_bf16 else "fp32",
        }
        self._samples: dict[str, list[float]] = {
            "batch_size": [],
            "batch_model_ms": [],
            "queue_wait_ms": [],
            "request_ms": [],
        }
        self.failure: BaseException | None = None
        self._thread = threading.Thread(target=self._serve, name="process-gpu-inference", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            while True:
                first = self._requests.get()
                if first is None:
                    return
                batch = [first]
                deadline = time.perf_counter() + self._runtime.wait_ms / 1000.0
                while len(batch) < self._runtime.max_batch_size:
                    if len(batch) >= self._runtime.min_batch_size:
                        break
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        request = self._requests.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if request is None:
                        # Shutdown is only sent after actors have stopped requesting.
                        return
                    batch.append(request)
                # Once the minimum is ready, drain an already queued burst without
                # adding latency. This lets 48 actors form batches above 16 while
                # preserving the 1 ms upper wait bound.
                while len(batch) < self._runtime.max_batch_size:
                    try:
                        request = self._requests.get_nowait()
                    except queue.Empty:
                        break
                    if request is None:
                        return
                    batch.append(request)
                actor_ids = [int(item[0]) for item in batch]
                features = np.ascontiguousarray(self._inputs[actor_ids])
                model_started = time.perf_counter()
                logits, values = self._backend.evaluate_features(features)
                model_finished = time.perf_counter()
                if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(values)):
                    raise ValueError("GPU inference produced NaN or Inf")
                for index, (actor_id, sequence, _enqueued_at) in enumerate(batch):
                    self._outputs[actor_id] = logits[index]
                    self._values[actor_id] = values[index]
                    self._responses[actor_id] = sequence
                    self._events[actor_id].set()
                finished = time.perf_counter()
                with self._lock:
                    self._statistics["requests"] = int(self._statistics["requests"]) + len(batch)
                    self._statistics["batches"] = int(self._statistics["batches"]) + 1
                    self._statistics["largest_batch"] = max(
                        int(self._statistics["largest_batch"]), len(batch)
                    )
                    self._statistics["model_seconds"] = float(
                        self._statistics["model_seconds"]
                    ) + (model_finished - model_started)
                    self._record("batch_size", float(len(batch)))
                    self._record("batch_model_ms", (model_finished - model_started) * 1000.0)
                    for _actor_id, _sequence, enqueued_at in batch:
                        self._record("queue_wait_ms", (model_started - enqueued_at) * 1000.0)
                        self._record("request_ms", (finished - enqueued_at) * 1000.0)
        except BaseException as exc:
            self.failure = exc
            for event in self._events:
                event.set()

    def _record(self, name: str, value: float, capacity: int = 20_000) -> None:
        values = self._samples[name]
        if len(values) < capacity:
            values.append(value)
            return
        index = random.randrange(int(self._statistics["requests"]) + 1)
        if index < capacity:
            values[index] = value

    def stats(self) -> dict[str, float | int | str]:
        with self._lock:
            result = dict(self._statistics)
            samples = {name: list(values) for name, values in self._samples.items()}
        batches = int(result["batches"])
        requests = int(result["requests"])
        result["mean_batch_size"] = requests / batches if batches else 0.0
        for name, values in samples.items():
            if values:
                result[f"{name}_p50"] = float(np.percentile(values, 50))
                result[f"{name}_p95"] = float(np.percentile(values, 95))
                result[f"{name}_samples"] = len(values)
        return result

    def close(self) -> None:
        self._requests.put(None)
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            raise RuntimeError("process inference server did not stop")
        if self.failure is not None:
            raise RuntimeError(f"process inference server failed: {self.failure}") from self.failure


class ProcessSelfPlayExecutor:
    """Roll games continuously across spawn actors and yield them in index order."""

    def __init__(
        self,
        model: PolicyValueModel,
        runtime: ProcessRuntimeConfig,
        *,
        simulations: int,
        sample_until_ply: int,
        max_ply: int,
        seed: int,
        first_game_index: int,
    ) -> None:
        if runtime.min_batch_size > runtime.max_batch_size:
            raise ValueError("minimum inference batch cannot exceed maximum batch")
        self.runtime = runtime
        self.simulations = simulations
        self.sample_until_ply = sample_until_ply
        self.max_ply = max_ply
        self.seed = seed
        self.first_game_index = first_game_index
        self._context = mp.get_context("spawn")
        self._requests = self._context.Queue()
        self._jobs = self._context.Queue()
        self._results = self._context.Queue()
        self._events = [self._context.Event() for _ in range(runtime.actors)]
        self._input_memory, self.inputs = self._allocate(
            (runtime.actors, *FEATURE_SHAPE), np.float32
        )
        self._output_memory, self.outputs = self._allocate(
            (runtime.actors, ACTION_SIZE), np.float32
        )
        self._value_memory, self.values = self._allocate((runtime.actors,), np.float32)
        self._sequence_memory, self.responses = self._allocate((runtime.actors,), np.int64)
        self._heartbeat_memory, self.heartbeats = self._allocate((runtime.actors,), np.float64)
        self.responses.fill(-1)
        self.heartbeats.fill(time.time())
        self._inference = _InferenceServer(
            model,
            runtime,
            self._requests,
            self._events,
            self.inputs,
            self.outputs,
            self.values,
            self.responses,
        )
        self._processes = [
            self._context.Process(
                target=_actor_main,
                name=f"chessai-actor-{actor_id:02d}",
                args=(
                    actor_id,
                    runtime,
                    self._input_memory.name,
                    self._output_memory.name,
                    self._value_memory.name,
                    self._sequence_memory.name,
                    self._heartbeat_memory.name,
                    self._requests,
                    self._jobs,
                    self._results,
                    self._events[actor_id],
                    simulations,
                    sample_until_ply,
                    max_ply,
                ),
            )
            for actor_id in range(runtime.actors)
        ]
        thread_variables = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        previous_thread_values = {name: os.environ.get(name) for name in thread_variables}
        try:
            for name in thread_variables:
                os.environ[name] = "1"
            for process in self._processes:
                process.start()
        finally:
            for name, value in previous_thread_values.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        self._closed = False
        self._assigned_games = 0
        self._completed_games = 0

    @staticmethod
    def _allocate(shape: tuple[int, ...], dtype: npt.DTypeLike) -> tuple[SharedMemory, np.ndarray]:
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        memory = SharedMemory(create=True, size=size)
        array = np.ndarray(shape, dtype=dtype, buffer=memory.buf)
        array.fill(0)
        return memory, array

    def games(
        self,
        *,
        game_limit: int | None,
        target_positions: int | None,
        previous_positions: int,
    ) -> Iterator[ProcessGameResult]:
        next_job = self.first_game_index
        next_yield = self.first_game_index
        inflight = 0
        observed_positions = 0
        observed_games = 0
        yielded_positions = 0
        buffered: dict[int, ProcessGameResult] = {}

        def target_met() -> bool:
            return (
                target_positions is not None
                and previous_positions + yielded_positions >= target_positions
            )

        def may_schedule() -> bool:
            if game_limit is not None:
                return next_job < self.first_game_index + game_limit
            assert target_positions is not None
            if target_met():
                return False
            if observed_games == 0:
                return inflight < self.runtime.actors
            average = observed_positions / observed_games
            projected = previous_positions + observed_positions + inflight * average
            return projected < target_positions

        def schedule() -> None:
            nonlocal next_job, inflight
            self._jobs.put((next_job, self.seed + next_job))
            next_job += 1
            inflight += 1
            self._assigned_games += 1

        while inflight < self.runtime.actors and may_schedule():
            schedule()

        while inflight:
            if self._inference.failure is not None:
                raise RuntimeError(
                    f"GPU inference failed: {self._inference.failure}"
                ) from self._inference.failure
            try:
                actor_result = self._results.get(timeout=1.0)
            except queue.Empty:
                dead = [
                    f"{process.name}(exit={process.exitcode})"
                    for process in self._processes
                    if not process.is_alive()
                ]
                if dead:
                    raise RuntimeError(
                        "self-play actor stopped unexpectedly: " + ", ".join(dead)
                    ) from None
                continue
            inflight -= 1
            self._completed_games += 1
            if actor_result.error is not None:
                raise RuntimeError(
                    f"self-play actor {actor_result.actor_id} failed in game "
                    f"{actor_result.game_index} (seed {actor_result.seed}):\n{actor_result.error}"
                )
            assert actor_result.replay is not None and actor_result.summary is not None
            result = ProcessGameResult(
                actor_result.game_index,
                actor_result.seed,
                actor_result.replay,
                actor_result.summary,
            )
            buffered[result.game_index] = result
            observed_games += 1
            observed_positions += result.replay.positions
            while may_schedule() and inflight < self.runtime.actors:
                schedule()
            while next_yield in buffered:
                ready = buffered.pop(next_yield)
                yielded_positions += ready.replay.positions
                next_yield += 1
                yield ready

        if buffered:
            raise RuntimeError(f"self-play results are not contiguous: {sorted(buffered)}")

    def stats(self) -> dict[str, Any]:
        now = time.time()
        return {
            **self._inference.stats(),
            "actors": self.runtime.actors,
            "actors_alive": sum(process.is_alive() for process in self._processes),
            "assigned_games": self._assigned_games,
            "completed_games": self._completed_games,
            "inflight_games": self._assigned_games - self._completed_games,
            "actor_heartbeat_age_seconds": [
                max(0.0, now - float(value)) for value in self.heartbeats
            ],
            "actor_exit_codes": {process.name: process.exitcode for process in self._processes},
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _process in self._processes:
            self._jobs.put(None)
        for process in self._processes:
            process.join(timeout=10.0)
        for process in self._processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        inference_error: BaseException | None = None
        try:
            self._inference.close()
        except BaseException as exc:
            inference_error = exc
        for memory in (
            self._input_memory,
            self._output_memory,
            self._value_memory,
            self._sequence_memory,
            self._heartbeat_memory,
        ):
            memory.close()
            memory.unlink()
        for channel in (self._requests, self._jobs, self._results):
            channel.close()
            channel.join_thread()
        if inference_error is not None:
            raise inference_error

    def __enter__(self) -> ProcessSelfPlayExecutor:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
