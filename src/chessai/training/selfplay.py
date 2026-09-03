"""Position-budgeted Gumbel self-play with sparse, resumable replay shards."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chessai.ai.evaluator import BatchingEvaluator, TorchEvaluator
from chessai.ai.features import encode_state
from chessai.ai.search import Evaluator, GumbelSearch, HeuristicEvaluator
from chessai.data.manifest import sha256_file, write_json_atomic
from chessai.engine import Color, GameState
from chessai.engine.vocabulary import encode_move
from chessai.training.checkpoint import load_checkpoint
from chessai.training.metrics import MetricsWriter
from chessai.training.replay import ReplaySample, save_replay_shard


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    games: int | None = 1
    target_positions: int | None = None
    simulations: int = 32
    sample_until_ply: int = 30
    max_ply: int = 300
    seed: int = 20260902
    shard_games: int = 16
    device: str = "cpu"
    actors: int = 1
    inference_batch_size: int = 64
    inference_wait_ms: float = 2.0
    precision: str = "auto"

    @classmethod
    def tiny(cls) -> SelfPlayConfig:
        return cls(
            games=1,
            simulations=8,
            sample_until_ply=8,
            max_ply=24,
            shard_games=1,
            device="cpu",
            actors=1,
        )


@dataclass(frozen=True, slots=True)
class PendingSample:
    features: np.ndarray
    action_ids: np.ndarray
    probabilities: np.ndarray
    perspective: Color


def _evaluator(
    checkpoint: str | Path | None,
    config: SelfPlayConfig,
) -> tuple[Evaluator, str, str, BatchingEvaluator | None]:
    if checkpoint is None:
        return HeuristicEvaluator(), "heuristic-v1", "heuristic", None
    loaded = load_checkpoint(checkpoint, device=config.device)
    network_hash = hashlib.sha256(loaded.weights_path.read_bytes()).hexdigest()
    if config.actors > 1:
        batching = BatchingEvaluator(
            loaded.model,
            config.device,
            max_batch_size=config.inference_batch_size,
            max_wait_ms=config.inference_wait_ms,
            precision=config.precision,
        )
        return batching, network_hash, "checkpoint", batching
    return (
        TorchEvaluator(loaded.model, device=config.device, precision=config.precision),
        network_hash,
        "checkpoint",
        None,
    )


def _final_value(winner: Color | None, perspective: Color) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner is perspective else -1.0


def play_selfplay_game(
    evaluator: Evaluator,
    *,
    simulations: int,
    sample_until_ply: int,
    max_ply: int,
    seed: int,
) -> tuple[list[ReplaySample], dict[str, Any]]:
    state = GameState.initial(max_ply=max_ply)
    pending: list[PendingSample] = []
    search = GumbelSearch(
        evaluator,
        simulations=simulations,
        max_considered_actions=min(16, simulations),
        seed=seed,
    )
    started = time.perf_counter()
    while not state.outcome().terminal:
        result = search.search(state, temperature=1.0 if state.ply < sample_until_ply else 0.0)
        action_ids: list[int] = []
        probabilities: list[float] = []
        legal_by_text = {str(move): move for move in state.legal_moves}
        for move_text, probability in result.root_policy.items():
            move = legal_by_text[move_text]
            action_ids.append(encode_move(move, canonical_black=state.side_to_move is Color.BLACK))
            probabilities.append(probability)
        pending.append(
            PendingSample(
                features=encode_state(state),
                action_ids=np.asarray(action_ids, dtype=np.uint16),
                probabilities=np.asarray(probabilities, dtype=np.float32),
                perspective=state.side_to_move,
            )
        )
        state = state.apply(result.best_move)
    outcome = state.outcome()
    samples = [
        ReplaySample(
            features=item.features,
            action_ids=item.action_ids,
            probabilities=item.probabilities,
            value=_final_value(outcome.winner, item.perspective),
        )
        for item in pending
    ]
    return samples, {
        "plies": len(samples),
        "status": outcome.status.value,
        "winner": outcome.winner.value if outcome.winner else None,
        "reason": outcome.reason,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _play_job(
    job: tuple[Evaluator, int, int, int, int],
) -> tuple[list[ReplaySample], dict[str, Any]]:
    evaluator, simulations, sample_until_ply, max_ply, seed = job
    return play_selfplay_game(
        evaluator,
        simulations=simulations,
        sample_until_ply=sample_until_ply,
        max_ply=max_ply,
        seed=seed,
    )


def run_selfplay(
    output_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    config: SelfPlayConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SelfPlayConfig()
    if (cfg.games is None) == (cfg.target_positions is None):
        raise ValueError("exactly one of games or target_positions must be configured")
    if cfg.games is not None and cfg.games <= 0:
        raise ValueError("games must be positive")
    if cfg.target_positions is not None and cfg.target_positions <= 0:
        raise ValueError("target_positions must be positive")
    if cfg.shard_games <= 0 or cfg.actors <= 0:
        raise ValueError("shard_games and actors must be positive")
    if cfg.inference_batch_size <= 0 or cfg.inference_wait_ms < 0:
        raise ValueError("inference batch size must be positive and wait must be non-negative")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = MetricsWriter(output / "metrics.jsonl")
    evaluator, network_hash, network_mode, batching = _evaluator(checkpoint, cfg)
    manifest_path = output / "manifest.json"
    existing_shards = sorted(output.glob("replay-*.npz"))
    previous_games = 0
    previous_positions = 0
    previous_elapsed = 0.0
    shards: list[dict[str, Any]] = []
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("kind") != "selfplay-run":
            raise ValueError(f"unexpected manifest kind in {manifest_path}")
        previous_config = previous.get("base_config", previous.get("config", {}))
        required_matches = (
            "target_positions",
            "simulations",
            "sample_until_ply",
            "max_ply",
            "seed",
        )
        mismatches = [
            f"{key}: existing={previous_config.get(key)!r}, requested={getattr(cfg, key)!r}"
            for key in required_matches
            if previous_config.get(key) != getattr(cfg, key)
        ]
        if previous.get("network_hash") != network_hash:
            mismatches.append(
                f"network_hash: existing={previous.get('network_hash')!r}, requested={network_hash!r}"
            )
        if mismatches:
            raise ValueError("cannot resume incompatible self-play run: " + "; ".join(mismatches))
        previous_games = int(previous.get("games", 0))
        previous_positions = int(previous.get("positions", 0))
        previous_elapsed = float(previous.get("elapsed_seconds", 0.0))
        shards = list(previous.get("shards", []))
        declared_paths = {str(item["path"]) for item in shards}
        actual_paths = {path.name for path in existing_shards}
        if declared_paths != actual_paths:
            raise ValueError(
                "replay manifest/file mismatch; refusing an ambiguous resume: "
                f"declared={sorted(declared_paths)}, actual={sorted(actual_paths)}"
            )
        for shard in shards:
            shard_path = output / str(shard["path"])
            actual_hash = sha256_file(shard_path)
            if actual_hash != shard.get("sha256"):
                raise ValueError(
                    "replay shard hash mismatch: "
                    f"{shard_path.name} expected={shard.get('sha256')}, got={actual_hash}"
                )
    elif existing_shards:
        raise ValueError("replay shards exist without manifest; refusing an ambiguous resume")

    shard_samples: list[ReplaySample] = []
    shard_game_count = 0
    invocation_positions = 0
    completed_games = 0
    started = time.perf_counter()

    def persist_progress() -> dict[str, Any]:
        elapsed = previous_elapsed + time.perf_counter() - started
        cumulative_games = previous_games + completed_games
        cumulative_positions = previous_positions + invocation_positions
        base_config = asdict(cfg)
        base_config["games"] = None
        current = {
            "kind": "selfplay-run",
            "base_config": base_config,
            "last_invocation_config": asdict(cfg),
            "network_hash": network_hash,
            "network_mode": network_mode,
            "games": cumulative_games,
            "positions": cumulative_positions,
            "target_positions": cfg.target_positions,
            "elapsed_seconds": elapsed,
            "positions_per_second": cumulative_positions / elapsed if elapsed else 0.0,
            "resume": {
                "next_game_index": cumulative_games,
                "next_seed": cfg.seed + cumulative_games,
                "next_shard_index": len(shards),
            },
            "inference": batching.stats() if batching is not None else {"mode": "synchronous"},
            "shards": shards,
        }
        write_json_atomic(manifest_path, current)
        return current

    def flush_shard() -> None:
        nonlocal shard_samples, shard_game_count
        if not shard_samples:
            return
        shard_path = output / f"replay-{len(shards):06d}.npz"
        absolute_last_game = previous_games + completed_games - 1
        shard = save_replay_shard(
            shard_path,
            shard_samples,
            network_hash=network_hash,
            simulations=cfg.simulations,
            seed=cfg.seed + absolute_last_game - shard_game_count + 1,
            games=shard_game_count,
        )
        shards.append(shard)
        shard_samples = []
        shard_game_count = 0
        persist_progress()

    def record_game(generated: tuple[list[ReplaySample], dict[str, Any]]) -> None:
        nonlocal shard_game_count, invocation_positions, completed_games
        samples, game_summary = generated
        absolute_game_index = previous_games + completed_games
        shard_samples.extend(samples)
        shard_game_count += 1
        invocation_positions += len(samples)
        completed_games += 1
        metrics.write("selfplay_game", game=absolute_game_index, **game_summary)
        if shard_game_count >= cfg.shard_games:
            flush_shard()

    def needs_more() -> bool:
        if cfg.target_positions is not None:
            return previous_positions + invocation_positions < cfg.target_positions
        assert cfg.games is not None
        return completed_games < cfg.games

    try:
        if cfg.actors > 1:
            with ThreadPoolExecutor(
                max_workers=cfg.actors,
                thread_name_prefix="chessai-selfplay",
            ) as executor:
                while needs_more():
                    batch_size = cfg.actors
                    if cfg.games is not None:
                        batch_size = min(batch_size, cfg.games - completed_games)
                    jobs = [
                        (
                            evaluator,
                            cfg.simulations,
                            cfg.sample_until_ply,
                            cfg.max_ply,
                            cfg.seed + previous_games + completed_games + offset,
                        )
                        for offset in range(batch_size)
                    ]
                    for generated in executor.map(_play_job, jobs):
                        record_game(generated)
        else:
            while needs_more():
                job = (
                    evaluator,
                    cfg.simulations,
                    cfg.sample_until_ply,
                    cfg.max_ply,
                    cfg.seed + previous_games + completed_games,
                )
                record_game(_play_job(job))
        flush_shard()
    finally:
        if batching is not None:
            batching.close()

    invocation_elapsed = time.perf_counter() - started
    manifest = persist_progress()
    metrics.write(
        "selfplay_complete",
        invocation_games=completed_games,
        invocation_positions=invocation_positions,
        invocation_elapsed_seconds=invocation_elapsed,
        **{key: value for key, value in manifest.items() if key != "shards"},
    )
    return manifest
