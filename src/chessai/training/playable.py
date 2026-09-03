"""Single-route, resumable training pipeline for a playable Xiangqi model."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from chessai.ai.evaluator import TorchEvaluator
from chessai.ai.model import ModelConfig
from chessai.ai.search import GumbelSearch
from chessai.data.manifest import sha256_file, write_json_atomic
from chessai.data.prepare import validate_prepared_dataset
from chessai.training.arena import (
    Player,
    RandomPlayer,
    SearchPlayer,
    arena_result_dict,
    quick_opening_fens,
    run_arena,
)
from chessai.training.bootstrap import BootstrapConfig, run_bootstrap
from chessai.training.checkpoint import load_checkpoint
from chessai.training.rl import RlTrainConfig, run_rl_training
from chessai.training.selfplay import SelfPlayConfig, run_selfplay

PLAYABLE_RUN_SCHEMA = "playable-run-v1"


@dataclass(frozen=True, slots=True)
class PlayableConfig:
    bootstrap: BootstrapConfig = field(
        default_factory=lambda: BootstrapConfig(
            epochs=1,
            batch_size=256,
            num_workers=8,
            device="cuda",
            precision="bf16",
        )
    )
    iterations: int = 5
    positions_per_iteration: int = 100_000
    simulation_schedule: tuple[int, ...] = (16, 16, 32, 32, 32)
    selfplay_sample_until_ply: int = 30
    selfplay_max_ply: int = 300
    selfplay_shard_games: int = 16
    selfplay_actors: int = 12
    inference_batch_size: int = 128
    inference_wait_ms: float = 2.0
    selfplay_device: str = "cuda"
    selfplay_precision: str = "bf16"
    rl: RlTrainConfig = field(
        default_factory=lambda: RlTrainConfig(
            batch_size=256,
            replay_capacity=300_000,
            epochs=1,
            device="cuda",
            precision="bf16",
        )
    )
    arena_games: int = 20
    arena_openings: int = 10
    arena_simulations: int = 64
    arena_max_ply: int = 300
    candidate_score_min: float = 0.50
    final_games: int = 20
    final_score_min: float = 0.80
    final_simulations: int = 64
    seed: int = 20260902
    keep_replay_iterations: int = 3
    tiny_model: bool = False

    def validate(self) -> None:
        if self.iterations <= 0 or self.positions_per_iteration <= 0:
            raise ValueError("iterations and positions_per_iteration must be positive")
        if len(self.simulation_schedule) != self.iterations:
            raise ValueError("simulation_schedule length must match iterations")
        if any(value <= 0 for value in self.simulation_schedule):
            raise ValueError("all simulation budgets must be positive")
        if self.arena_games <= 0 or self.arena_games % 2:
            raise ValueError("arena_games must be a positive even number")
        if self.final_games <= 0 or self.final_games % 2:
            raise ValueError("final_games must be a positive even number")
        if self.arena_openings <= 0 or self.keep_replay_iterations <= 0:
            raise ValueError("arena_openings and keep_replay_iterations must be positive")
        if not 0.0 <= self.candidate_score_min <= 1.0:
            raise ValueError("candidate_score_min must be between zero and one")
        if not 0.0 <= self.final_score_min <= 1.0:
            raise ValueError("final_score_min must be between zero and one")

    @classmethod
    def tiny(cls) -> PlayableConfig:
        return cls(
            bootstrap=BootstrapConfig.tiny(),
            iterations=1,
            positions_per_iteration=2,
            simulation_schedule=(1,),
            selfplay_sample_until_ply=1,
            selfplay_max_ply=2,
            selfplay_shard_games=1,
            selfplay_actors=1,
            inference_batch_size=1,
            selfplay_device="cpu",
            selfplay_precision="fp32",
            rl=RlTrainConfig.tiny(),
            arena_games=2,
            arena_openings=1,
            arena_simulations=1,
            arena_max_ply=2,
            final_games=2,
            final_simulations=1,
            keep_replay_iterations=1,
            tiny_model=True,
        )


def playable_config_from_mapping(mapping: dict[str, Any]) -> PlayableConfig:
    """Parse the single supported YAML profile without hiding unknown fields."""

    bootstrap = BootstrapConfig(**cast(dict[str, Any], mapping.get("bootstrap", {})))
    rl = RlTrainConfig(**cast(dict[str, Any], mapping.get("rl", {})))
    selfplay = dict(cast(dict[str, Any], mapping.get("selfplay", {})))
    evaluation = dict(cast(dict[str, Any], mapping.get("evaluation", {})))
    pipeline = dict(cast(dict[str, Any], mapping.get("pipeline", {})))
    schedule = pipeline.pop("simulation_schedule", (16, 16, 32, 32, 32))
    config = PlayableConfig(
        bootstrap=bootstrap,
        rl=rl,
        simulation_schedule=tuple(int(value) for value in schedule),
        iterations=int(pipeline.pop("iterations", 5)),
        positions_per_iteration=int(pipeline.pop("positions_per_iteration", 100_000)),
        seed=int(pipeline.pop("seed", 20260902)),
        keep_replay_iterations=int(pipeline.pop("keep_replay_iterations", 3)),
        selfplay_sample_until_ply=int(selfplay.pop("sample_until_ply", 30)),
        selfplay_max_ply=int(selfplay.pop("max_ply", 300)),
        selfplay_shard_games=int(selfplay.pop("shard_games", 16)),
        selfplay_actors=int(selfplay.pop("actors", 12)),
        inference_batch_size=int(selfplay.pop("inference_batch_size", 128)),
        inference_wait_ms=float(selfplay.pop("inference_wait_ms", 2.0)),
        selfplay_device=str(selfplay.pop("device", "cuda")),
        selfplay_precision=str(selfplay.pop("precision", "bf16")),
        arena_games=int(evaluation.pop("arena_games", 20)),
        arena_openings=int(evaluation.pop("arena_openings", 10)),
        arena_simulations=int(evaluation.pop("arena_simulations", 64)),
        arena_max_ply=int(evaluation.pop("max_ply", 300)),
        candidate_score_min=float(evaluation.pop("candidate_score_min", 0.50)),
        final_games=int(evaluation.pop("final_games", 20)),
        final_score_min=float(evaluation.pop("final_score_min", 0.80)),
        final_simulations=int(evaluation.pop("final_simulations", 64)),
    )
    leftovers = {
        "pipeline": pipeline,
        "selfplay": selfplay,
        "evaluation": evaluation,
    }
    unknown = {section: values for section, values in leftovers.items() if values}
    if unknown:
        raise ValueError(f"unknown playable configuration fields: {unknown}")
    config.validate()
    return config


def _checkpoint_record(path: Path) -> dict[str, str]:
    loaded = load_checkpoint(path, device="cpu")
    return {"path": str(path), "weights_sha256": sha256_file(loaded.weights_path)}


def _safe_remove_tree(path: Path, root: Path) -> None:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        raise ValueError(f"refusing to remove path outside managed root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _install_best(source: Path, model_dir: Path) -> tuple[dict[str, str], dict[str, str] | None]:
    """Copy and validate a candidate before atomically rotating best/rollback."""

    source_record = _checkpoint_record(source)
    best = model_dir / "best"
    rollback = model_dir / "rollback"
    staging = model_dir / ".best.next"
    model_dir.mkdir(parents=True, exist_ok=True)
    if best.is_dir():
        current = _checkpoint_record(best)
        if current["weights_sha256"] == source_record["weights_sha256"]:
            previous = _checkpoint_record(rollback) if rollback.is_dir() else None
            return current, previous
    _safe_remove_tree(staging, model_dir)
    shutil.copytree(source, staging)
    staged = _checkpoint_record(staging)
    if staged["weights_sha256"] != source_record["weights_sha256"]:
        raise ValueError("staged best checkpoint hash does not match the source")
    if best.is_dir():
        _safe_remove_tree(rollback, model_dir)
        os.replace(best, rollback)
    os.replace(staging, best)
    previous = _checkpoint_record(rollback) if rollback.is_dir() else None
    return _checkpoint_record(best), previous


def _checkpoint_player(path: Path, *, device: str, simulations: int, seed: int) -> SearchPlayer:
    loaded = load_checkpoint(path, device=device)
    digest = sha256_file(loaded.weights_path)
    return SearchPlayer(
        GumbelSearch(
            TorchEvaluator(loaded.model, device=device),
            simulations=simulations,
            max_considered_actions=min(16, simulations),
            seed=seed,
        ),
        name=f"checkpoint:{path.name}@{digest[:12]}-n{simulations}",
    )


def _arena_report(
    candidate: Path,
    opponent: Path | RandomPlayer,
    *,
    device: str,
    games: int,
    openings: int,
    simulations: int,
    max_ply: int,
    seed: int,
) -> dict[str, Any]:
    candidate_record = _checkpoint_record(candidate)
    candidate_player = _checkpoint_player(
        candidate, device=device, simulations=simulations, seed=seed
    )
    opponent_player: Player
    if isinstance(opponent, Path):
        opponent_record: dict[str, str] | None = _checkpoint_record(opponent)
        opponent_player = _checkpoint_player(
            opponent, device=device, simulations=simulations, seed=seed + 1
        )
    else:
        opponent_record = None
        opponent_player = opponent
    summary, results = run_arena(
        candidate_player,
        opponent_player,
        games=games,
        opening_fens=quick_opening_fens(openings, seed=seed),
        max_ply=max_ply,
    )
    return {
        "candidate": candidate_record,
        "opponent": opponent_record or {"kind": "random", "seed": seed + 1},
        "simulations": simulations,
        "summary": arena_result_dict(summary),
        "games": [asdict(result) for result in results],
    }


def _prune_replay(state: dict[str, Any], output: Path, keep: int) -> None:
    entries = cast(list[dict[str, Any]], state["replay_iterations"])
    for entry in entries[:-keep]:
        if entry.get("pruned"):
            continue
        replay_dir = Path(str(entry["path"]))
        _safe_remove_tree(replay_dir, output)
        entry["pruned"] = True


def _validate_resume_artifacts(state: dict[str, Any]) -> None:
    """Verify every retained artifact referenced by a resumable run."""

    for checkpoint_field in (
        "active_checkpoint",
        "rollback_checkpoint",
        "candidate_checkpoint",
    ):
        value = state.get(checkpoint_field)
        if value is None:
            continue
        record = cast(dict[str, str], value)
        actual = _checkpoint_record(Path(record["path"]))
        if actual["weights_sha256"] != record["weights_sha256"]:
            raise ValueError(f"{checkpoint_field} checkpoint hash changed")
    for entry in cast(list[dict[str, Any]], state.get("replay_iterations", [])):
        if entry.get("pruned"):
            continue
        manifest = Path(str(entry["path"])) / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"retained replay manifest is missing: {manifest}")
        actual_hash = sha256_file(manifest)
        if actual_hash != entry["manifest_sha256"]:
            raise ValueError(f"retained replay manifest hash changed: {manifest}")


def run_playable_training(
    data_dir: str | Path,
    output_dir: str | Path,
    model_dir: str | Path,
    *,
    config: PlayableConfig | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    cfg = config or PlayableConfig()
    cfg.validate()
    config_payload = json.loads(json.dumps(asdict(cfg)))
    data_root = Path(data_dir).resolve()
    output = Path(output_dir).resolve()
    models = Path(model_dir).resolve()
    state_path = output / "state.json"
    dataset_manifest = validate_prepared_dataset(data_root)
    dataset_files = {
        split: data_root / str(details["path"])
        for split, details in cast(dict[str, dict[str, Any]], dataset_manifest["outputs"]).items()
    }
    for required in ("train", "validation", "test"):
        if required not in dataset_files:
            raise ValueError(f"prepared dataset is missing the {required} split")

    if state_path.is_file():
        if not resume:
            raise FileExistsError("playable output already has state.json; pass --resume")
        loaded_state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_state, dict):
            raise ValueError("playable run state root must be an object")
        state = cast(dict[str, Any], loaded_state)
        if state.get("schema_version") != PLAYABLE_RUN_SCHEMA:
            raise ValueError(f"unsupported playable run state: {state.get('schema_version')!r}")
        if state.get("dataset_manifest_sha256") != sha256_file(data_root / "manifest.json"):
            raise ValueError("prepared dataset manifest changed since the run started")
        if state.get("config") != config_payload:
            raise ValueError("playable configuration changed; resume requires the original profile")
        _validate_resume_artifacts(state)
    else:
        if resume:
            raise FileNotFoundError(f"playable state does not exist: {state_path}")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"playable output is not empty: {output}")
        if (models / "best").exists() or (models / "rollback").exists():
            raise FileExistsError("model directory already contains best or rollback")
        output.mkdir(parents=True, exist_ok=True)
        models.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": PLAYABLE_RUN_SCHEMA,
            "stage": "bootstrap",
            "iteration": 0,
            "seed": cfg.seed,
            "config": config_payload,
            "dataset_manifest": str(data_root / "manifest.json"),
            "dataset_manifest_sha256": sha256_file(data_root / "manifest.json"),
            "bootstrap_positions": 0,
            "rl_generated_positions": 0,
            "active_checkpoint": None,
            "rollback_checkpoint": None,
            "candidate_checkpoint": None,
            "replay_iterations": [],
            "evaluations": [],
            "completed_steps": [],
            "playable_gate_passed": None,
        }
        write_json_atomic(state_path, state)

    if state["stage"] == "complete":
        active = cast(dict[str, str], state["active_checkpoint"])
        if _checkpoint_record(Path(active["path"]))["weights_sha256"] != active["weights_sha256"]:
            raise ValueError("completed run's best checkpoint hash changed")
        return state

    if state["stage"] == "bootstrap":
        bootstrap_dir = output / "bootstrap"
        summary = run_bootstrap(
            dataset_files["train"],
            bootstrap_dir,
            config=cfg.bootstrap,
            model_config=ModelConfig.tiny() if cfg.tiny_model else ModelConfig(),
            validation_jsonl=dataset_files["validation"],
            test_jsonl=dataset_files["test"],
        )
        best, rollback = _install_best(bootstrap_dir / "bootstrap-best", models)
        state.update(
            {
                "stage": "selfplay",
                "bootstrap_positions": int(summary["positions"]),
                "active_checkpoint": best,
                "rollback_checkpoint": rollback,
            }
        )
        cast(list[str], state["completed_steps"]).append("bootstrap")
        write_json_atomic(state_path, state)

    while int(state["iteration"]) < cfg.iterations:
        iteration_index = int(state["iteration"])
        iteration_number = iteration_index + 1
        iteration_dir = output / f"iteration-{iteration_number:03d}"
        replay_dir = iteration_dir / "selfplay"
        rl_dir = iteration_dir / "rl"
        active = cast(dict[str, str], state["active_checkpoint"])
        best_path = Path(active["path"])
        actual_best = _checkpoint_record(best_path)
        if actual_best["weights_sha256"] != active["weights_sha256"]:
            raise ValueError("active checkpoint hash changed during playable training")

        if state["stage"] == "selfplay":
            selfplay_config = SelfPlayConfig(
                games=None,
                target_positions=cfg.positions_per_iteration,
                simulations=cfg.simulation_schedule[iteration_index],
                sample_until_ply=cfg.selfplay_sample_until_ply,
                max_ply=cfg.selfplay_max_ply,
                seed=cfg.seed + iteration_index * 1_000_000,
                shard_games=cfg.selfplay_shard_games,
                device=cfg.selfplay_device,
                actors=cfg.selfplay_actors,
                inference_batch_size=cfg.inference_batch_size,
                inference_wait_ms=cfg.inference_wait_ms,
                precision=cfg.selfplay_precision,
            )
            manifest = run_selfplay(replay_dir, checkpoint=best_path, config=selfplay_config)
            entry = {
                "iteration": iteration_number,
                "path": str(replay_dir),
                "positions": int(manifest["positions"]),
                "manifest_sha256": sha256_file(replay_dir / "manifest.json"),
                "pruned": False,
            }
            entries = cast(list[dict[str, Any]], state["replay_iterations"])
            existing = next(
                (item for item in entries if int(item["iteration"]) == iteration_number), None
            )
            if existing is None:
                entries.append(entry)
            else:
                existing.update(entry)
            state["rl_generated_positions"] = sum(int(item["positions"]) for item in entries)
            state["stage"] = "rl"
            cast(list[str], state["completed_steps"]).append(
                f"iteration-{iteration_number:03d}:selfplay"
            )
            write_json_atomic(state_path, state)

        if state["stage"] == "rl":
            replay_paths = sorted(output.glob("iteration-*/selfplay/replay-*.npz"))
            if not replay_paths:
                raise FileNotFoundError("no retained replay shards are available for RL")
            run_rl_training(
                replay_paths,
                rl_dir,
                checkpoint=best_path,
                config=cfg.rl,
            )
            candidate_path = rl_dir / "candidate"
            state["candidate_checkpoint"] = _checkpoint_record(candidate_path)
            state["stage"] = "arena"
            cast(list[str], state["completed_steps"]).append(f"iteration-{iteration_number:03d}:rl")
            write_json_atomic(state_path, state)

        if state["stage"] == "arena":
            candidate = cast(dict[str, str], state["candidate_checkpoint"])
            candidate_path = Path(candidate["path"])
            if _checkpoint_record(candidate_path)["weights_sha256"] != candidate["weights_sha256"]:
                raise ValueError("candidate checkpoint hash changed before evaluation")
            current_best_hash = _checkpoint_record(models / "best")["weights_sha256"]
            report: dict[str, Any]
            if current_best_hash == candidate["weights_sha256"]:
                accepted = True
                report = {
                    "recovered_after_install": True,
                    "accepted": True,
                    "summary": {"score_rate": 1.0},
                }
                state["active_checkpoint"] = _checkpoint_record(models / "best")
                state["rollback_checkpoint"] = (
                    _checkpoint_record(models / "rollback")
                    if (models / "rollback").is_dir()
                    else None
                )
                write_json_atomic(iteration_dir / "arena.json", report)
            else:
                try:
                    report = _arena_report(
                        candidate_path,
                        models / "best",
                        device=cfg.selfplay_device,
                        games=cfg.arena_games,
                        openings=cfg.arena_openings,
                        simulations=cfg.arena_simulations,
                        max_ply=cfg.arena_max_ply,
                        seed=cfg.seed + iteration_index * 100,
                    )
                    accepted = float(report["summary"]["score_rate"]) >= cfg.candidate_score_min
                except Exception as exc:  # Arena failure rejects, but does not discard the run.
                    accepted = False
                    report = {
                        "error": f"{type(exc).__name__}: {exc}",
                        "summary": {
                            "games": 0,
                            "wins": 0,
                            "draws": 0,
                            "losses": 0,
                            "score_rate": 0.0,
                        },
                    }
                report["accepted"] = accepted
                write_json_atomic(iteration_dir / "arena.json", report)
                if accepted:
                    best, rollback = _install_best(candidate_path, models)
                    state["active_checkpoint"] = best
                    state["rollback_checkpoint"] = rollback
            cast(list[dict[str, Any]], state["evaluations"]).append(
                {
                    "iteration": iteration_number,
                    "accepted": accepted,
                    "report": str(iteration_dir / "arena.json"),
                    "summary": report["summary"],
                }
            )
            state["candidate_checkpoint"] = None
            state["iteration"] = iteration_number
            state["stage"] = "final" if iteration_number == cfg.iterations else "selfplay"
            cast(list[str], state["completed_steps"]).append(
                f"iteration-{iteration_number:03d}:arena"
            )
            write_json_atomic(state_path, state)
            _safe_remove_tree(candidate_path, output)
            _prune_replay(state, output, cfg.keep_replay_iterations)
            write_json_atomic(state_path, state)

    if state["stage"] == "final":
        try:
            report = _arena_report(
                models / "best",
                RandomPlayer(cfg.seed + 99_999),
                device=cfg.selfplay_device,
                games=cfg.final_games,
                openings=min(cfg.arena_openings, cfg.final_games // 2),
                simulations=cfg.final_simulations,
                max_ply=cfg.arena_max_ply,
                seed=cfg.seed + 99_998,
            )
            passed = float(report["summary"]["score_rate"]) >= cfg.final_score_min
        except Exception as exc:  # Preserve the best model and make the failed gate explicit.
            passed = False
            report = {
                "error": f"{type(exc).__name__}: {exc}",
                "summary": {
                    "games": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "score_rate": 0.0,
                },
            }
        report["playable_gate_passed"] = passed
        write_json_atomic(output / "final-evaluation.json", report)
        state["playable_gate_passed"] = passed
        state["final_evaluation"] = str(output / "final-evaluation.json")
        state["stage"] = "complete"
        cast(list[str], state["completed_steps"]).append("final-evaluation")
        state["active_checkpoint"] = _checkpoint_record(models / "best")
        state["rollback_checkpoint"] = (
            _checkpoint_record(models / "rollback") if (models / "rollback").is_dir() else None
        )
        write_json_atomic(state_path, state)
    return state
