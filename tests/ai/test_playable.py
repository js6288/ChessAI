import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from chessai.ai.model import ModelConfig, PolicyValueModel  # noqa: E402
from chessai.data.manifest import sha256_file  # noqa: E402
from chessai.engine import GameState  # noqa: E402
from chessai.training.checkpoint import save_checkpoint  # noqa: E402
from chessai.training.playable import (  # noqa: E402
    LEGACY_PLAYABLE_RUN_SCHEMA,
    PLAYABLE_RUN_SCHEMA,
    PlayableConfig,
    _install_best,
    _prune_replay,
    _safe_remove_tree,
    _validate_resume_artifacts,
    playable_config_from_mapping,
    run_playable_training,
)


def _prepared_dataset(path: Path) -> Path:
    path.mkdir(parents=True)
    record = {
        "game_id": "playable-fixture",
        "initial_fen": GameState.initial().to_fen(),
        "moves": ["h2e2", "h7e7"],
        "result": "1/2-1/2",
    }
    line = json.dumps(record) + "\n"
    outputs = {}
    for split in ("train", "validation", "test"):
        split_path = path / f"{split}.jsonl"
        split_path.write_text(line, encoding="utf-8")
        outputs[split] = {
            "path": split_path.name,
            "sha256": sha256_file(split_path),
            "games": 1,
        }
    (path / "manifest.json").write_text(json.dumps({"outputs": outputs}), encoding="utf-8")
    return path


@pytest.mark.torch
def test_tiny_playable_pipeline_completes_and_resume_is_a_noop(tmp_path: Path) -> None:
    data = _prepared_dataset(tmp_path / "data")
    output = tmp_path / "run"
    models = tmp_path / "models"

    state = run_playable_training(
        data,
        output,
        models,
        config=PlayableConfig.tiny(),
    )

    assert state["schema_version"] == PLAYABLE_RUN_SCHEMA
    assert state["stage"] == "complete"
    assert state["iteration"] == 1
    assert state["rl_generated_positions"] >= 2
    assert (models / "best" / "metadata.json").is_file()
    assert not (output / "iteration-001" / "rl" / "candidate").exists()
    runtime = json.loads(
        (output / "iteration-001" / "selfplay" / "runtime.json").read_text(encoding="utf-8")
    )
    assert runtime["executor"] == "process"
    assert runtime["inference"]["actors"] == 2
    assert runtime["inference"]["inflight_games"] == 0
    state_path = output / "state.json"
    before = state_path.read_bytes()

    resumed = run_playable_training(
        data,
        output,
        models,
        config=PlayableConfig.tiny(),
        resume=True,
    )
    assert resumed == state
    assert state_path.read_bytes() == before


def test_playable_refuses_nonempty_output_without_resume(tmp_path: Path) -> None:
    data = _prepared_dataset(tmp_path / "data")
    output = tmp_path / "run"
    output.mkdir()
    (output / "user-file.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        run_playable_training(data, output, tmp_path / "models", config=PlayableConfig.tiny())
    assert (output / "user-file.txt").read_text(encoding="utf-8") == "preserve"


@pytest.mark.torch
def test_rejected_candidate_keeps_best_without_creating_rollback(tmp_path: Path) -> None:
    data = _prepared_dataset(tmp_path / "data")
    output = tmp_path / "run"
    models = tmp_path / "models"
    config = replace(PlayableConfig.tiny(), candidate_score_min=1.0)

    state = run_playable_training(data, output, models, config=config)

    assert state["stage"] == "complete"
    assert state["evaluations"][0]["accepted"] is False
    assert (models / "best" / "metadata.json").is_file()
    assert not (models / "rollback").exists()
    assert not (output / "iteration-001" / "rl" / "candidate").exists()


@pytest.mark.torch
def test_best_rotation_keeps_only_best_and_rollback(tmp_path: Path) -> None:
    models = tmp_path / "models"
    first = tmp_path / "sources" / "first"
    second = tmp_path / "sources" / "second"
    first_model = PolicyValueModel(ModelConfig.tiny())
    second_model = PolicyValueModel(ModelConfig.tiny())
    with torch.no_grad():
        next(second_model.parameters()).add_(1.0)
    save_checkpoint(first, first_model)
    save_checkpoint(second, second_model)

    best_one, rollback_one = _install_best(first, models)
    best_two, rollback_two = _install_best(second, models)

    assert rollback_one is None
    assert best_one["weights_sha256"] == rollback_two["weights_sha256"]
    assert best_two["weights_sha256"] != rollback_two["weights_sha256"]
    assert {child.name for child in models.iterdir()} == {"best", "rollback"}


def test_replay_cleanup_cannot_escape_the_run_directory(tmp_path: Path) -> None:
    run = tmp_path / "run"
    inside = run / "iteration-01" / "selfplay"
    outside = tmp_path / "user-data"
    inside.mkdir(parents=True)
    outside.mkdir()
    state = {
        "replay_iterations": [
            {"path": str(inside), "pruned": False},
            {"path": str(run / "iteration-02" / "selfplay"), "pruned": False},
        ]
    }

    _prune_replay(state, run, 1)
    assert not inside.exists()
    assert state["replay_iterations"][0]["pruned"] is True
    with pytest.raises(ValueError, match="outside managed root"):
        _safe_remove_tree(outside, run)
    assert outside.is_dir()


@pytest.mark.torch
def test_resume_artifact_validation_rejects_checkpoint_and_replay_changes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best"
    save_checkpoint(checkpoint, PolicyValueModel(ModelConfig.tiny()))
    weights = checkpoint / "weights.safetensors"
    replay = tmp_path / "selfplay"
    replay.mkdir()
    manifest = replay / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    state = {
        "active_checkpoint": {
            "path": str(checkpoint),
            "weights_sha256": sha256_file(weights),
        },
        "rollback_checkpoint": None,
        "candidate_checkpoint": None,
        "replay_iterations": [
            {
                "path": str(replay),
                "manifest_sha256": sha256_file(manifest),
                "pruned": False,
            }
        ],
    }
    _validate_resume_artifacts(state)

    manifest.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="replay manifest hash changed"):
        _validate_resume_artifacts(state)

    state["replay_iterations"][0]["pruned"] = True
    with weights.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match=r"weights hash mismatch|checkpoint hash changed"):
        _validate_resume_artifacts(state)


def test_playable_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown playable configuration"):
        playable_config_from_mapping({"pipeline": {"obsolete_ablation_budget": 1}})


def test_product_training_budget_is_three_times_fifty_thousand() -> None:
    config = PlayableConfig()
    assert config.iterations == 3
    assert config.positions_per_iteration == 50_000
    assert config.simulation_schedule == (16, 16, 32)


@pytest.mark.torch
def test_v1_restart_archives_only_partial_selfplay_and_keeps_bootstrap(tmp_path: Path) -> None:
    data = _prepared_dataset(tmp_path / "data")
    output = tmp_path / "run"
    models = tmp_path / "models"
    best = models / "best"
    save_checkpoint(best, PolicyValueModel(ModelConfig.tiny()))
    partial = output / "iteration-001" / "selfplay"
    partial.mkdir(parents=True)
    partial_manifest = {
        "kind": "selfplay-run",
        "games": 3,
        "positions": 6,
        "positions_per_second": 0.2,
    }
    (partial / "manifest.json").write_text(json.dumps(partial_manifest), encoding="utf-8")
    config = PlayableConfig.tiny()
    old_config = json.loads(json.dumps(asdict(config)))
    old_config.update(
        {
            "iterations": 5,
            "positions_per_iteration": 100_000,
            "simulation_schedule": [16, 16, 32, 32, 32],
        }
    )
    state = {
        "schema_version": LEGACY_PLAYABLE_RUN_SCHEMA,
        "stage": "selfplay",
        "iteration": 0,
        "seed": config.seed,
        "config": old_config,
        "dataset_manifest": str(data / "manifest.json"),
        "dataset_manifest_sha256": sha256_file(data / "manifest.json"),
        "bootstrap_positions": 123,
        "rl_generated_positions": 0,
        "active_checkpoint": {
            "path": str(best),
            "weights_sha256": sha256_file(best / "weights.safetensors"),
        },
        "rollback_checkpoint": None,
        "candidate_checkpoint": None,
        "replay_iterations": [],
        "evaluations": [],
        "completed_steps": ["bootstrap"],
        "playable_gate_passed": None,
    }
    output.mkdir(exist_ok=True)
    (output / "state.json").write_text(json.dumps(state), encoding="utf-8")

    completed = run_playable_training(
        data,
        output,
        models,
        config=config,
        resume=True,
        restart_current_selfplay=True,
    )

    assert completed["schema_version"] == PLAYABLE_RUN_SCHEMA
    assert completed["bootstrap_positions"] == 123
    assert completed["restart_current_selfplay_used"] is True
    assert (output / "state.v1.backup.json").is_file()
    abandoned = completed["abandoned_selfplay"][0]
    assert abandoned["positions"] == 6
    assert Path(abandoned["path"]).is_dir()
    assert completed["config"]["iterations"] == 1
