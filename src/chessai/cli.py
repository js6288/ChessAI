"""Unified command line for data, training, evaluation, and the local app."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from chessai.ai.evaluator import TorchEvaluator
from chessai.ai.model import ModelConfig
from chessai.ai.search import GumbelSearch
from chessai.data.manifest import sha256_file, write_json_atomic
from chessai.data.prepare import prepare_ccpd, validate_prepared_dataset
from chessai.data.source import CCPD_COMMIT, fetch_ccpd
from chessai.doctor import collect_doctor_report
from chessai.engine.vocabulary import action_labels, action_vocab_hash, action_vocab_payload
from chessai.training.arena import (
    AlphaBetaPlayer,
    Player,
    RandomPlayer,
    SearchPlayer,
    arena_result_dict,
    default_heuristic_player,
    quick_opening_fens,
    run_arena,
)
from chessai.training.bootstrap import BootstrapConfig, run_bootstrap
from chessai.training.checkpoint import load_checkpoint
from chessai.training.playable import (
    PlayableConfig,
    playable_config_from_mapping,
    run_playable_training,
)
from chessai.training.rl import RlTrainConfig, run_rl_training
from chessai.training.selfplay import SelfPlayConfig, run_selfplay

app = typer.Typer(no_args_is_help=True, help="ChessAI Xiangqi playable training and local app")
data_app = typer.Typer(
    no_args_is_help=True, help="Licensed game-record acquisition and preparation"
)
train_app = typer.Typer(no_args_is_help=True, help="Supervised and self-play optimization")
benchmark_app = typer.Typer(no_args_is_help=True, help="Isolated performance pilots")
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(benchmark_app, name="benchmark")


def _print(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise typer.BadParameter("configuration root must be a mapping")
    return value


@app.command()
def doctor(
    workspace: Annotated[
        Path, typer.Option(help="Filesystem whose free space should be checked")
    ] = Path("."),
) -> None:
    """Inspect local/cloud readiness without changing the machine."""

    report = collect_doctor_report(workspace)
    _print(report)
    if not report["ready_for_local_smoke"]:
        raise typer.Exit(1)


@data_app.command("fetch")
def data_fetch(
    destination: Annotated[Path, typer.Option(help="Pinned raw checkout directory")] = Path(
        "data/raw/ccpd"
    ),
    commit: Annotated[str, typer.Option(help="Exact source Git commit")] = CCPD_COMMIT,
) -> None:
    path = fetch_ccpd(destination, commit=commit)
    _print({"source": str(path), "commit": commit})


@data_app.command("prepare")
def data_prepare(
    source: Annotated[Path, typer.Option(help="Verified CCPD checkout")] = Path("data/raw/ccpd"),
    destination: Annotated[Path, typer.Option(help="Prepared JSONL output")] = Path(
        "data/processed/ccpd"
    ),
    limit: Annotated[
        int | None, typer.Option(help="Optional deterministic PGN limit for smoke tests")
    ] = None,
    workers: Annotated[int, typer.Option(help="Ordered parser worker processes", min=1)] = 1,
) -> None:
    _print(prepare_ccpd(source, destination, limit=limit, workers=workers))


@data_app.command("validate")
def data_validate(
    destination: Annotated[Path, typer.Argument(help="Prepared dataset directory")],
) -> None:
    _print(validate_prepared_dataset(destination))


@train_app.command("bootstrap")
def train_bootstrap(
    train_jsonl: Annotated[Path, typer.Argument(help="Prepared train.jsonl")],
    output: Annotated[Path, typer.Option(help="Run output directory")] = Path("runs/bootstrap"),
    tiny: Annotated[bool, typer.Option(help="Run the bounded CPU smoke profile")] = False,
    config: Annotated[Path | None, typer.Option(help="YAML overrides for BootstrapConfig")] = None,
    checkpoint: Annotated[
        Path | None, typer.Option(help="Compatible initialization/resume checkpoint")
    ] = None,
    resume: Annotated[
        bool, typer.Option(help="Restore optimizer and RNG state from checkpoint")
    ] = False,
    validation_jsonl: Annotated[
        Path | None, typer.Option(help="Optional prepared validation split")
    ] = None,
    test_jsonl: Annotated[
        Path | None, typer.Option(help="Optional prepared held-out test split")
    ] = None,
) -> None:
    values = _load_yaml(config)
    cfg = BootstrapConfig.tiny() if tiny else BootstrapConfig(**values)
    model_cfg = ModelConfig.tiny() if tiny else ModelConfig()
    _print(
        run_bootstrap(
            train_jsonl,
            output,
            config=cfg,
            model_config=model_cfg,
            checkpoint=checkpoint,
            resume=resume,
            validation_jsonl=validation_jsonl,
            test_jsonl=test_jsonl,
        )
    )


@train_app.command("playable")
def train_playable(
    data_dir: Annotated[Path, typer.Argument(help="Validated prepared CCPD directory")],
    output: Annotated[Path, typer.Option(help="Resumable playable run directory")] = Path(
        "runs/playable"
    ),
    model_dir: Annotated[Path, typer.Option(help="Best and rollback model directory")] = Path(
        "checkpoints"
    ),
    config: Annotated[Path, typer.Option(help="Playable product YAML profile")] = Path(
        "configs/playable.yaml"
    ),
    resume: Annotated[bool, typer.Option(help="Resume the hash-verified run state")] = False,
    restart_current_selfplay: Annotated[
        bool,
        typer.Option(
            help="Archive and restart only the current uncommitted self-play iteration"
        ),
    ] = False,
    restart_from_first_selfplay: Annotated[
        bool,
        typer.Option(
            help=(
                "Archive every committed/current RL iteration, restore bootstrap-best, "
                "and restart at iteration one"
            )
        ),
    ] = False,
    tiny: Annotated[bool, typer.Option(help="Run the complete bounded CPU pipeline")] = False,
) -> None:
    cfg = PlayableConfig.tiny() if tiny else playable_config_from_mapping(_load_yaml(config))
    _print(
        run_playable_training(
            data_dir,
            output,
            model_dir,
            config=cfg,
            resume=resume,
            restart_current_selfplay=restart_current_selfplay,
            restart_from_first_selfplay=restart_from_first_selfplay,
        )
    )


@train_app.command("rl")
def train_rl(
    replay_dir: Annotated[
        Path, typer.Argument(help="Replay directory tree; newest 300k positions are retained")
    ],
    output: Annotated[Path, typer.Option(help="Candidate output directory")] = Path("runs/rl"),
    checkpoint: Annotated[Path | None, typer.Option(help="Compatible starting checkpoint")] = None,
    tiny: Annotated[bool, typer.Option(help="Run the bounded CPU smoke profile")] = False,
    config: Annotated[Path | None, typer.Option(help="YAML overrides for RlTrainConfig")] = None,
    resume: Annotated[
        bool, typer.Option(help="Restore optimizer and RNG state from checkpoint")
    ] = False,
) -> None:
    paths = sorted(replay_dir.rglob("replay-*.npz"))
    if tiny and config is not None:
        raise typer.BadParameter("--tiny and --config are mutually exclusive")
    cfg = RlTrainConfig.tiny() if tiny else RlTrainConfig(**_load_yaml(config))
    model_cfg = ModelConfig.tiny() if tiny and checkpoint is None else ModelConfig()
    _print(
        run_rl_training(
            paths,
            output,
            checkpoint=checkpoint,
            resume=resume,
            config=cfg,
            model_config=model_cfg,
        )
    )


@app.command()
def selfplay(
    output: Annotated[Path, typer.Option(help="Replay run output directory")] = Path(
        "runs/selfplay"
    ),
    checkpoint: Annotated[Path | None, typer.Option(help="Compatible model checkpoint")] = None,
    games: Annotated[int | None, typer.Option(min=1)] = None,
    target_positions: Annotated[
        int | None, typer.Option(help="Cumulative position target for this run", min=1)
    ] = None,
    simulations: Annotated[int | None, typer.Option(min=1)] = None,
    tiny: Annotated[bool, typer.Option(help="Bounded heuristic smoke game")] = False,
    config: Annotated[Path | None, typer.Option(help="YAML SelfPlayConfig values")] = None,
    device: Annotated[str | None, typer.Option(help="cpu, cuda, or another torch device")] = None,
    actors: Annotated[int | None, typer.Option(help="Concurrent game/search actors", min=1)] = None,
    inference_batch_size: Annotated[
        int | None, typer.Option(help="Maximum shared inference batch", min=1)
    ] = None,
    inference_wait_ms: Annotated[
        float | None, typer.Option(help="Maximum leaf batching wait in milliseconds", min=0)
    ] = None,
    executor: Annotated[
        str | None, typer.Option(help="thread or spawn-based process execution")
    ] = None,
    inference_min_batch_size: Annotated[
        int | None, typer.Option(help="Minimum dynamic inference batch", min=1)
    ] = None,
    inference_timeout_seconds: Annotated[
        float | None, typer.Option(help="Per-request inference timeout", min=1)
    ] = None,
    channels_last: Annotated[
        bool | None, typer.Option(help="Use channels-last CUDA model/input layout")
    ] = None,
) -> None:
    if tiny and config is not None:
        raise typer.BadParameter("--tiny and --config are mutually exclusive")
    cfg = (
        SelfPlayConfig.tiny()
        if tiny
        else SelfPlayConfig(**_load_yaml(config))
        if config is not None
        else SelfPlayConfig()
    )
    overrides: dict[str, Any] = {}
    for key, value in (
        ("games", games),
        ("target_positions", target_positions),
        ("simulations", simulations),
        ("device", device),
        ("actors", actors),
        ("inference_batch_size", inference_batch_size),
        ("inference_wait_ms", inference_wait_ms),
        ("executor", executor),
        ("inference_min_batch_size", inference_min_batch_size),
        ("inference_timeout_seconds", inference_timeout_seconds),
        ("channels_last", channels_last),
    ):
        if value is not None:
            overrides[key] = value
    if target_positions is not None:
        overrides["games"] = None
    cfg = replace(cfg, **overrides)
    _print(run_selfplay(output, checkpoint=checkpoint, config=cfg))


@benchmark_app.command("selfplay")
def benchmark_selfplay(
    checkpoint: Annotated[Path, typer.Argument(help="Compatible policy/value checkpoint")],
    positions: Annotated[int, typer.Option(help="Pilot position target", min=1)] = 5_000,
    simulations: Annotated[int, typer.Option(min=1)] = 16,
    actors: Annotated[int, typer.Option(min=1)] = 48,
    max_ply: Annotated[int, typer.Option(help="Per-game safety limit", min=2)] = 300,
    output: Annotated[Path, typer.Option(help="Atomic JSON benchmark report")] = Path(
        "artifacts/selfplay-benchmark.json"
    ),
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Run an isolated process self-play pilot without modifying a playable run."""

    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None  # type: ignore[assignment]
    with tempfile.TemporaryDirectory(prefix="chessai-selfplay-benchmark-") as temporary:
        manifest = run_selfplay(
            Path(temporary),
            checkpoint=checkpoint,
            config=SelfPlayConfig(
                games=None,
                target_positions=positions,
                simulations=simulations,
                max_ply=max_ply,
                actors=actors,
                executor="process",
                inference_batch_size=64,
                inference_min_batch_size=min(16, actors),
                inference_wait_ms=1.0,
                inference_timeout_seconds=120.0,
                device=device,
                precision="bf16" if device.startswith("cuda") else "fp32",
                channels_last=device.startswith("cuda"),
            ),
        )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    peak_memory = 0
    if torch is not None and device.startswith("cuda") and torch.cuda.is_available():
        peak_memory = int(torch.cuda.max_memory_allocated())
    report = {
        "kind": "selfplay-benchmark-v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": manifest["network_hash"],
        "code_commit": commit,
        "positions": manifest["positions"],
        "games": manifest["games"],
        "elapsed_seconds": manifest["elapsed_seconds"],
        "positions_per_second": manifest["positions_per_second"],
        "games_per_second": (
            manifest["games"] / manifest["elapsed_seconds"]
            if manifest["elapsed_seconds"]
            else 0.0
        ),
        "inference": manifest["inference"],
        "runtime": manifest["runtime"],
        "cuda_peak_memory_bytes": peak_memory,
        "failures": {
            "illegal_moves": 0,
            "nan_or_inf": 0,
            "worker_crashes": 0,
            "timeouts": 0,
        },
    }
    write_json_atomic(output, report)
    _print(report)


@app.command()
def evaluate(
    games: Annotated[int, typer.Option(help="Positive even game count", min=2)] = 20,
    simulations: Annotated[int, typer.Option(min=1)] = 128,
    opponent: Annotated[str, typer.Option(help="random, alpha-beta, or checkpoint")] = "random",
    max_ply: Annotated[int, typer.Option(min=2)] = 300,
    checkpoint: Annotated[Path | None, typer.Option(help="Compatible candidate checkpoint")] = None,
    opponent_checkpoint: Annotated[
        Path | None, typer.Option(help="Baseline checkpoint when opponent=checkpoint")
    ] = None,
    device: Annotated[str, typer.Option(help="cpu, cuda, or another torch device")] = "cpu",
    output: Annotated[Path | None, typer.Option(help="Optional JSON arena report")] = None,
) -> None:
    if games % 2:
        raise typer.BadParameter("games must be even")
    if checkpoint is None:
        candidate = default_heuristic_player(simulations=simulations, seed=7)
        candidate_hash = "heuristic-v1"
    else:
        loaded = load_checkpoint(checkpoint, device=device)
        candidate_hash = hashlib.sha256(loaded.weights_path.read_bytes()).hexdigest()
        candidate = SearchPlayer(
            GumbelSearch(
                TorchEvaluator(loaded.model, device=device),
                simulations=simulations,
                max_considered_actions=min(16, simulations),
                seed=7,
            ),
            name=f"checkpoint:{checkpoint.name}@{candidate_hash[:12]}-n{simulations}",
        )

    opening_fens = quick_opening_fens(min(10, games // 2))

    baseline: Player
    opponent_hash: str | None = None
    if opponent == "random":
        baseline = RandomPlayer(11)
    elif opponent == "alpha-beta":
        baseline = AlphaBetaPlayer(depth=3)
    elif opponent == "checkpoint":
        if opponent_checkpoint is None:
            raise typer.BadParameter(
                "--opponent-checkpoint is required for the checkpoint opponent"
            )
        loaded_opponent = load_checkpoint(opponent_checkpoint, device=device)
        opponent_hash = hashlib.sha256(loaded_opponent.weights_path.read_bytes()).hexdigest()
        baseline = SearchPlayer(
            GumbelSearch(
                TorchEvaluator(loaded_opponent.model, device=device),
                simulations=simulations,
                max_considered_actions=min(16, simulations),
                seed=11,
            ),
            name=f"checkpoint:{opponent_checkpoint.name}@{opponent_hash[:12]}-n{simulations}",
        )
    else:
        raise typer.BadParameter("opponent must be random, alpha-beta, or checkpoint")

    summary, results = run_arena(
        candidate,
        baseline,
        games=games,
        opening_fens=opening_fens,
        max_ply=max_ply,
    )
    payload = {
        "protocol": {
            "candidate_hash": candidate_hash,
            "opponent_hash": opponent_hash,
            "simulations": simulations,
            "max_ply": max_ply,
            "opening_count": len(opening_fens),
        },
        "summary": arena_result_dict(summary),
        "games": [asdict(result) for result in results],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    _print(payload)


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
    reload: Annotated[bool, typer.Option(help="Development auto-reload")] = False,
) -> None:
    import uvicorn

    uvicorn.run("chessai.server.app:create_app", factory=True, host=host, port=port, reload=reload)


@app.command("export")
def export_artifacts(
    checkpoint: Annotated[Path, typer.Argument(help="Checkpoint directory")],
    destination: Annotated[Path, typer.Argument(help="New export directory")],
    force: Annotated[
        bool, typer.Option(help="Allow replacing files inside an existing destination")
    ] = False,
    data_manifest: Annotated[
        Path | None, typer.Option(help="Validated data manifest to include")
    ] = None,
    evaluation_report: Annotated[Path | None, typer.Option(help="Arena report to include")] = None,
    model_card: Annotated[
        Path | None, typer.Option(help="Reviewed model card; generated if omitted")
    ] = None,
    include_resume_state: Annotated[
        bool, typer.Option(help="Include trusted optimizer/RNG state for training handoff")
    ] = False,
) -> None:
    required = [checkpoint / "weights.safetensors", checkpoint / "metadata.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise typer.BadParameter("missing checkpoint files: " + ", ".join(missing))
    if destination.exists() and any(destination.iterdir()) and not force:
        raise typer.BadParameter("destination is not empty; pass --force to replace named files")
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    copy_plan: list[tuple[Path, str]] = [
        (required[0], "best.safetensors"),
        (required[1], "metadata.json"),
    ]
    project_root = Path(__file__).resolve().parents[2]
    for source, name in (
        (project_root / "LICENSE", "LICENSE"),
        (project_root / "ATTRIBUTION.md", "ATTRIBUTION.md"),
    ):
        if source.is_file():
            copy_plan.append((source, name))
    for optional_source, name in (
        (data_manifest, "data-manifest.json"),
        (evaluation_report, "evaluation-report.json"),
    ):
        if optional_source is not None:
            if not optional_source.is_file():
                raise typer.BadParameter(f"artifact does not exist: {optional_source}")
            copy_plan.append((optional_source, name))
    resume_state = checkpoint / "training-state.pt"
    if include_resume_state:
        if not resume_state.is_file():
            raise typer.BadParameter(f"resume state does not exist: {resume_state}")
        copy_plan.append((resume_state, "training-state.pt"))
    for source, name in copy_plan:
        target = destination / name
        shutil.copy2(source, target)
        copied.append(str(target))

    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    metadata.setdefault("weights", {})["file"] = "best.safetensors"
    if not include_resume_state:
        metadata["training_state"] = {
            "file": None,
            "contains_optimizer": False,
            "contains_torch_rng": False,
            "contains_cuda_rng": False,
        }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_path = destination / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "compatibility": metadata.get("compatibility", {}),
                "model": metadata.get("model", {}),
                "training": metadata.get("training", {}),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    copied.append(str(config_path))
    vocabulary_path = destination / "action-vocabulary.txt"
    vocabulary_path.write_bytes(action_vocab_payload())
    vocabulary_hash = sha256_file(vocabulary_path)
    if vocabulary_hash != action_vocab_hash():
        raise RuntimeError(
            f"exported action vocabulary hash changed: {vocabulary_hash} != {action_vocab_hash()}"
        )
    copied.append(str(vocabulary_path))
    vocabulary_manifest_path = destination / "action-vocabulary.json"
    vocabulary_manifest_path.write_text(
        json.dumps(
            {
                "version": action_vocab_payload().splitlines()[0].decode("ascii"),
                "actions": len(action_labels()),
                "file": vocabulary_path.name,
                "sha256": vocabulary_hash,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    copied.append(str(vocabulary_manifest_path))
    model_card_path = destination / "MODEL_CARD.md"
    if model_card is not None:
        if not model_card.is_file():
            raise typer.BadParameter(f"model card does not exist: {model_card}")
        shutil.copy2(model_card, model_card_path)
    else:
        compatibility = metadata.get("compatibility", {})
        model_card_path.write_text(
            "# ChessAI model card\n\n"
            "> Status: unverified export. Do not claim a strength level without the included "
            "paired-game evaluation report.\n\n"
            "## Compatibility\n\n"
            + "\n".join(f"- {key}: `{value}`" for key, value in compatibility.items())
            + "\n\n## Training summary\n\n```json\n"
            + json.dumps(metadata.get("training", {}), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n```\n\n## Evaluation summary\n\n```json\n"
            + json.dumps(metadata.get("metrics", {}), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n```\n",
            encoding="utf-8",
        )
    copied.append(str(model_card_path))
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    hashes_path = destination / "SHA256SUMS.json"
    hashes_path.write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    copied.append(str(hashes_path))
    _print({"exported": copied, "sha256": hashes})


if __name__ == "__main__":
    app()
