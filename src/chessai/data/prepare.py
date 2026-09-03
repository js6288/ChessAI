"""Deterministic validation, deduplication, and game-level dataset splitting."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, TextIO, cast

from chessai.data.manifest import (
    base_manifest,
    replace_file_atomic,
    sha256_file,
    sha256_text,
    write_json_atomic,
)
from chessai.data.pgn import PgnError, decode_pgn_bytes, parse_pgn
from chessai.data.source import CCPD_COMMIT, verify_ccpd_checkout


def _game_id(initial_fen: str, moves: tuple[str, ...], result: str) -> str:
    return sha256_text(f"{initial_fen}\n{' '.join(moves)}\n{result}\n")


def _split(game_id: str) -> str:
    bucket = int(game_id[:8], 16) % 1000
    if bucket < 900:
        return "train"
    if bucket < 950:
        return "validation"
    return "test"


def _open_temporary_outputs(destination: Path) -> tuple[dict[str, TextIO], dict[str, Path]]:
    handles: dict[str, TextIO] = {}
    paths: dict[str, Path] = {}
    for split in ("train", "validation", "test"):
        path = destination / f"{split}.jsonl"
        temporary = path.with_name(path.name + ".tmp")
        handles[split] = temporary.open("w", encoding="utf-8", newline="\n")
        paths[split] = path
    return handles, paths


def _process_pgn_file(job: tuple[str, str]) -> dict[str, Any]:
    absolute_path_text, relative_path = job
    path = Path(absolute_path_text)
    source_hash: str | None = None
    source_size: int | None = None
    encoding: str | None = None
    try:
        payload = path.read_bytes()
        source_size = len(payload)
        source_hash = hashlib.sha256(payload).hexdigest()
        text, encoding = decode_pgn_bytes(payload)
        game = parse_pgn(text)
        moves = tuple(str(move) for move in game.moves)
        identifier = _game_id(game.initial_fen, moves, game.result)
        return {
            "status": "parsed",
            "path": relative_path,
            "sha256": source_hash,
            "bytes": source_size,
            "encoding": encoding,
            "game_id": identifier,
            "record": {
                "game_id": identifier,
                "initial_fen": game.initial_fen,
                "moves": moves,
                "result": game.result,
                "source_path": relative_path,
                "source_sha256": source_hash,
                "source_encoding": encoding,
                "tags": game.tags,
            },
        }
    except (OSError, UnicodeError, PgnError, ValueError) as exc:
        return {
            "status": "rejected",
            "path": relative_path,
            "sha256": source_hash,
            "bytes": source_size,
            "encoding": encoding,
            "reason": type(exc).__name__ + ": " + str(exc)[:500],
        }


def prepare_ccpd(
    source: str | Path,
    destination: str | Path,
    *,
    limit: int | None = None,
    workers: int = 1,
    expected_commit: str = CCPD_COMMIT,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    source_manifest = verify_ccpd_checkout(source_path, expected_commit=expected_commit)
    pgn_root = source_path / "Dataset" / "對局" / "大師對局"
    files = sorted(pgn_root.rglob("*.pgn"))
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        files = files[:limit]
    destination_path.mkdir(parents=True, exist_ok=True)
    handles, final_paths = _open_temporary_outputs(destination_path)
    accepted: set[str] = set()
    rejection_reasons: Counter[str] = Counter()
    encodings: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    duplicates = 0
    file_manifest_path = destination_path / "file-manifest.jsonl"
    temporary_file_manifest = file_manifest_path.with_name(file_manifest_path.name + ".tmp")
    file_manifest = temporary_file_manifest.open("w", encoding="utf-8", newline="\n")
    jobs = [(str(path), path.relative_to(source_path).as_posix()) for path in files]
    process_pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    processed_files = (
        process_pool.map(_process_pgn_file, jobs, chunksize=8)
        if process_pool is not None
        else map(_process_pgn_file, jobs)
    )
    try:
        for processed in processed_files:
            if processed["status"] == "parsed":
                identifier = str(processed["game_id"])
                if identifier in accepted:
                    duplicates += 1
                    file_manifest.write(
                        json.dumps(
                            {
                                "path": processed["path"],
                                "sha256": processed["sha256"],
                                "bytes": processed["bytes"],
                                "encoding": processed["encoding"],
                                "status": "duplicate",
                                "game_id": identifier,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    continue
                record = processed["record"]
                accepted.add(identifier)
                split = _split(identifier)
                handles[split].write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                split_counts[split] += 1
                encodings[str(processed["encoding"])] += 1
                file_manifest.write(
                    json.dumps(
                        {
                            "path": processed["path"],
                            "sha256": processed["sha256"],
                            "bytes": processed["bytes"],
                            "encoding": processed["encoding"],
                            "status": "accepted",
                            "game_id": identifier,
                            "split": split,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            else:
                reason = str(processed["reason"])
                rejection_reasons[reason] += 1
                file_manifest.write(
                    json.dumps(
                        {
                            "path": processed["path"],
                            "sha256": processed["sha256"],
                            "bytes": processed["bytes"],
                            "encoding": processed["encoding"],
                            "status": "rejected",
                            "reason": reason,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    finally:
        for handle in handles.values():
            handle.close()
        file_manifest.close()
        if process_pool is not None:
            process_pool.shutdown()

    for _name, final_path in final_paths.items():
        temporary = final_path.with_name(final_path.name + ".tmp")
        replace_file_atomic(temporary, final_path)
    replace_file_atomic(temporary_file_manifest, file_manifest_path)

    manifest = base_manifest(kind="prepared-dataset")
    manifest.update(
        {
            "source": source_manifest,
            "input_files": len(files),
            "preparation": {"workers": workers, "limit": limit},
            "accepted_games": len(accepted),
            "duplicates": duplicates,
            "rejected": sum(rejection_reasons.values()),
            "split_counts": dict(split_counts),
            "encodings": dict(encodings),
            "rejection_reasons": dict(rejection_reasons.most_common()),
            "file_manifest": {
                "path": file_manifest_path.name,
                "sha256": sha256_file(file_manifest_path),
                "files": len(files),
            },
            "outputs": {
                split: {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "games": split_counts[split],
                }
                for split, path in final_paths.items()
            },
        }
    )
    write_json_atomic(destination_path / "manifest.json", manifest)
    return manifest


def validate_prepared_dataset(destination: str | Path) -> dict[str, Any]:
    path = Path(destination).resolve()
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("prepared dataset manifest root must be an object")
    manifest = cast(dict[str, Any], loaded)
    if "file_manifest" in manifest:
        details = manifest["file_manifest"]
        file_manifest_path = path / details["path"]
        actual = sha256_file(file_manifest_path)
        if actual != details["sha256"]:
            raise ValueError(
                f"file manifest hash mismatch: expected {details['sha256']}, got {actual}"
            )
        with file_manifest_path.open("r", encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        if count != details["files"]:
            raise ValueError(
                f"file manifest count mismatch: expected {details['files']}, got {count}"
            )
    for split, details in manifest["outputs"].items():
        output = path / details["path"]
        actual = sha256_file(output)
        if actual != details["sha256"]:
            raise ValueError(f"{split} hash mismatch: expected {details['sha256']}, got {actual}")
        with output.open("r", encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        if count != details["games"]:
            raise ValueError(f"{split} count mismatch: expected {details['games']}, got {count}")
    return manifest
