import json
from pathlib import Path

from chessai.data.manifest import replace_file_atomic, sha256_file
from chessai.data.prepare import validate_prepared_dataset


def test_validate_prepared_dataset_detects_tampering(tmp_path: Path) -> None:
    outputs = {}
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text("" if split != "train" else '{"game_id":"x"}\n', encoding="utf-8")
        outputs[split] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "games": 1 if split == "train" else 0,
        }
    manifest = {"outputs": outputs}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_prepared_dataset(tmp_path)["outputs"]["train"]["games"] == 1

    (tmp_path / "train.jsonl").write_text("tampered\n", encoding="utf-8")
    try:
        validate_prepared_dataset(tmp_path)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("tampered data was accepted")


def test_atomic_replace_retries_windows_style_sharing_violation(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.tmp"
    destination = tmp_path / "destination.json"
    source.write_text("ready", encoding="utf-8")
    import chessai.data.manifest as manifest_module

    real_replace = manifest_module.os.replace
    calls = 0

    def flaky_replace(from_path, to_path):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError("simulated scanner lock")
        real_replace(from_path, to_path)

    monkeypatch.setattr(manifest_module.os, "replace", flaky_replace)
    monkeypatch.setattr(manifest_module.time, "sleep", lambda _seconds: None)
    replace_file_atomic(source, destination)
    assert calls == 3
    assert destination.read_text(encoding="utf-8") == "ready"
