import json
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from chessai.ai.model import ModelConfig, PolicyValueModel  # noqa: E402
from chessai.data.manifest import sha256_file  # noqa: E402
from chessai.engine import GameState  # noqa: E402
from chessai.training.checkpoint import save_checkpoint  # noqa: E402
from chessai.training.playable import (  # noqa: E402
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
