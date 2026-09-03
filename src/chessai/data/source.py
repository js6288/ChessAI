"""Pinned acquisition for the default CC BY 4.0 Xiangqi record source."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from chessai.data.manifest import base_manifest, sha256_file, write_json_atomic

CCPD_REPOSITORY = "https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset.git"
CCPD_COMMIT = "368a47a947773dd8692c026e286dd19b6277b993"
CCPD_LICENSE = "CC BY 4.0"


def _git(*arguments: str, cwd: Path | None = None) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return process.stdout.strip()


def verify_ccpd_checkout(path: Path, *, expected_commit: str = CCPD_COMMIT) -> dict[str, Any]:
    if not (path / ".git").is_dir():
        raise ValueError(f"CCPD checkout is not a Git repository: {path}")
    actual_commit = _git("rev-parse", "HEAD", cwd=path)
    if actual_commit != expected_commit:
        raise ValueError(f"CCPD revision mismatch: expected {expected_commit}, got {actual_commit}")
    try:
        _git("diff", "--quiet", cwd=path)
    except subprocess.CalledProcessError as exc:
        raise ValueError("CCPD checkout has tracked worktree modifications") from exc
    license_path = path / "LICENSE"
    readme_path = path / "README.md"
    if not license_path.is_file() or not readme_path.is_file():
        raise ValueError("CCPD checkout is missing LICENSE or README.md")
    license_text = license_path.read_text(encoding="utf-8", errors="replace").lower()
    if "creative commons attribution 4.0" not in license_text and "cc by 4.0" not in license_text:
        raise ValueError("CCPD license file does not identify Creative Commons Attribution 4.0")
    pgn_root = path / "Dataset" / "對局" / "大師對局"
    if not pgn_root.is_dir():
        raise ValueError(f"CCPD master-game directory is missing: {pgn_root}")
    pgn_count = sum(1 for _path in pgn_root.rglob("*.pgn"))
    if pgn_count == 0:
        raise ValueError(f"CCPD master-game directory contains no PGN files: {pgn_root}")

    manifest = base_manifest(kind="external-source")
    manifest.update(
        {
            "name": "Chinese Chess Practical Dataset",
            "repository": CCPD_REPOSITORY,
            "commit": actual_commit,
            "license": CCPD_LICENSE,
            "scope": "Dataset/對局/大師對局",
            "pgn_files": pgn_count,
            "git_tree": _git("rev-parse", "HEAD^{tree}", cwd=path),
            "files": {
                "LICENSE": sha256_file(license_path),
                "README.md": sha256_file(readme_path),
            },
        }
    )
    return manifest


def fetch_ccpd(destination: str | Path, *, commit: str = CCPD_COMMIT) -> Path:
    target = Path(destination).resolve()
    if target.exists():
        manifest = verify_ccpd_checkout(target, expected_commit=commit)
        write_json_atomic(target / "source-manifest.json", manifest)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    _git("clone", "--filter=blob:none", "--no-checkout", CCPD_REPOSITORY, str(target))
    try:
        _git("sparse-checkout", "init", "--cone", cwd=target)
        _git(
            "sparse-checkout",
            "set",
            "Dataset/對局/大師對局",
            "LICENSE",
            "README.md",
            cwd=target,
        )
        _git("checkout", "--detach", commit, cwd=target)
        manifest = verify_ccpd_checkout(target, expected_commit=commit)
        write_json_atomic(target / "source-manifest.json", manifest)
    except Exception:
        # Preserve the partial checkout for diagnosis; never delete a broad or
        # unresolved path from an acquisition failure.
        raise
    return target
