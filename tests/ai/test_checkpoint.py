import json

import pytest

torch = pytest.importorskip("torch")

from chessai.ai.model import ModelConfig, PolicyValueModel  # noqa: E402
from chessai.training.checkpoint import (  # noqa: E402
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)


@pytest.mark.torch
def test_checkpoint_round_trip_and_compatibility(tmp_path) -> None:
    model = PolicyValueModel(ModelConfig.tiny())
    optimizer = torch.optim.AdamW(model.parameters())
    torch.manual_seed(918)
    save_checkpoint(tmp_path / "model", model, training={"step": 12}, optimizer=optimizer)
    expected_random = torch.rand(4)
    loaded = load_checkpoint(tmp_path / "model")
    assert loaded.model.config == model.config
    assert loaded.metadata["training"]["step"] == 12
    assert len(loaded.metadata["weights"]["sha256"]) == 64
    for expected, actual in zip(model.parameters(), loaded.model.parameters(), strict=True):
        assert torch.equal(expected, actual)
    torch.manual_seed(1)
    restored_optimizer = torch.optim.AdamW(loaded.model.parameters())
    restore_training_state(loaded, optimizer=restored_optimizer)
    assert torch.equal(torch.rand(4), expected_random)

    metadata_path = tmp_path / "model" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["compatibility"]["feature_version"] = "future-format"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible checkpoint"):
        load_checkpoint(tmp_path / "model")


@pytest.mark.torch
def test_checkpoint_rejects_a_declared_weight_hash_mismatch(tmp_path) -> None:
    model = PolicyValueModel(ModelConfig.tiny())
    save_checkpoint(tmp_path / "model", model)
    metadata_path = tmp_path / "model" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["weights"]["sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="weights hash mismatch"):
        load_checkpoint(tmp_path / "model")


@pytest.mark.torch
def test_checkpoint_loader_accepts_a_hash_bound_renamed_export_weight(tmp_path) -> None:
    checkpoint = tmp_path / "export"
    model = PolicyValueModel(ModelConfig.tiny())
    save_checkpoint(checkpoint, model)
    (checkpoint / "weights.safetensors").replace(checkpoint / "best.safetensors")
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["weights"]["file"] = "best.safetensors"
    metadata["training_state"]["file"] = None
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    loaded = load_checkpoint(checkpoint)
    assert loaded.weights_path.name == "best.safetensors"
    assert loaded.training_state_path is None
