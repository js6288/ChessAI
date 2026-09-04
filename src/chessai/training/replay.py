"""Sparse, compressed self-play shards and replay datasets."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from chessai.ai.features import INPUT_PLANES
from chessai.compat import FEATURE_VERSION, RULE_VERSION, SCHEMA_VERSION, SEARCH_VERSION
from chessai.data.manifest import sha256_file
from chessai.engine.vocabulary import action_vocab_hash

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("install the 'train' extra to use replay datasets") from exc


@dataclass(frozen=True, slots=True)
class ReplaySample:
    features: npt.NDArray[np.float32]
    action_ids: npt.NDArray[np.uint16]
    probabilities: npt.NDArray[np.float32]
    value: float


@dataclass(frozen=True, slots=True)
class PackedReplayBatch:
    """Compact, pickle-friendly replay payload used between actor processes."""

    packed_features: np.ndarray
    halfmove: np.ndarray
    policy_offsets: np.ndarray
    action_ids: np.ndarray
    probabilities: np.ndarray
    values: np.ndarray

    @property
    def positions(self) -> int:
        return int(self.values.shape[0])


@dataclass(frozen=True, slots=True)
class ShardMetadata:
    schema_version: str
    rule_version: str
    feature_version: str
    search_version: str
    action_vocab_hash: str
    network_hash: str
    simulations: int
    seed: int
    games: int
    positions: int


def _pack_features(features: npt.NDArray[np.float32]) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 4 or features.shape[1:] != (INPUT_PLANES, 10, 9):
        raise ValueError(f"feature batch has wrong shape: {features.shape}")
    binary = features[:, :116].reshape(features.shape[0], 116, 90) > 0.5
    packed = np.packbits(binary, axis=-1, bitorder="little")
    halfmove = features[:, 116, 0, 0].astype(np.float16)
    return packed, halfmove


def _unpack_features(packed: np.ndarray, halfmove: np.ndarray) -> npt.NDArray[np.float32]:
    binary = np.unpackbits(packed, axis=-1, count=90, bitorder="little").astype(np.float32)
    features = np.zeros((packed.shape[0], INPUT_PLANES, 10, 9), dtype=np.float32)
    features[:, :116] = binary.reshape(packed.shape[0], 116, 10, 9)
    features[:, 116] = halfmove.astype(np.float32)[:, None, None]
    return features


def pack_replay_samples(samples: Sequence[ReplaySample]) -> PackedReplayBatch:
    if not samples:
        raise ValueError("cannot pack empty replay samples")
    features = np.stack([sample.features for sample in samples]).astype(np.float32)
    packed, halfmove = _pack_features(features)
    offsets = [0]
    action_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    for sample in samples:
        if sample.action_ids.size == 0 or sample.action_ids.shape != sample.probabilities.shape:
            raise ValueError("each replay sample needs matching non-empty sparse policy arrays")
        probability_sum = float(sample.probabilities.sum())
        if not np.isfinite(probability_sum) or probability_sum <= 0:
            raise ValueError("replay policy probabilities must have a finite positive sum")
        action_parts.append(sample.action_ids.astype(np.uint16, copy=False))
        probability_parts.append((sample.probabilities / probability_sum).astype(np.float16))
        offsets.append(offsets[-1] + sample.action_ids.size)
    return PackedReplayBatch(
        packed_features=packed,
        halfmove=halfmove,
        policy_offsets=np.asarray(offsets, dtype=np.int64),
        action_ids=np.concatenate(action_parts),
        probabilities=np.concatenate(probability_parts),
        values=np.asarray([sample.value for sample in samples], dtype=np.int8),
    )


def combine_packed_replay(batches: Sequence[PackedReplayBatch]) -> PackedReplayBatch:
    if not batches:
        raise ValueError("cannot combine empty packed replay batches")
    offsets = [0]
    for batch in batches:
        base = offsets[-1]
        offsets.extend((batch.policy_offsets[1:] + base).tolist())
    return PackedReplayBatch(
        packed_features=np.concatenate([batch.packed_features for batch in batches], axis=0),
        halfmove=np.concatenate([batch.halfmove for batch in batches], axis=0),
        policy_offsets=np.asarray(offsets, dtype=np.int64),
        action_ids=np.concatenate([batch.action_ids for batch in batches]),
        probabilities=np.concatenate([batch.probabilities for batch in batches]),
        values=np.concatenate([batch.values for batch in batches]),
    )


def read_replay_metadata(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as archive:
        loaded = json.loads(str(archive["metadata"].item()))
    if not isinstance(loaded, dict):
        raise ValueError(f"replay metadata root must be an object: {path}")
    metadata = cast(dict[str, Any], loaded)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "rule_version": RULE_VERSION,
        "feature_version": FEATURE_VERSION,
        "search_version": SEARCH_VERSION,
        "action_vocab_hash": action_vocab_hash(),
    }
    mismatches = [
        f"{key}: shard={metadata.get(key)!r}, runtime={value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise ValueError("incompatible replay shard: " + "; ".join(mismatches))
    return metadata


def save_replay_shard(
    path: str | Path,
    samples: Sequence[ReplaySample],
    *,
    network_hash: str,
    simulations: int,
    seed: int,
    games: int,
) -> dict[str, Any]:
    return save_packed_replay_shard(
        path,
        pack_replay_samples(samples),
        network_hash=network_hash,
        simulations=simulations,
        seed=seed,
        games=games,
    )


def save_packed_replay_shard(
    path: str | Path,
    packed: PackedReplayBatch,
    *,
    network_hash: str,
    simulations: int,
    seed: int,
    games: int,
) -> dict[str, Any]:
    if packed.positions <= 0:
        raise ValueError("cannot save an empty replay shard")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite replay shard: {target}")
    metadata = ShardMetadata(
        schema_version=SCHEMA_VERSION,
        rule_version=RULE_VERSION,
        feature_version=FEATURE_VERSION,
        search_version=SEARCH_VERSION,
        action_vocab_hash=action_vocab_hash(),
        network_hash=network_hash,
        simulations=simulations,
        seed=seed,
        games=games,
        positions=packed.positions,
    )
    temporary = target.with_name(target.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        packed_features=packed.packed_features,
        halfmove=packed.halfmove,
        policy_offsets=packed.policy_offsets,
        action_ids=packed.action_ids,
        probabilities=packed.probabilities,
        values=packed.values,
        metadata=np.asarray(json.dumps(asdict(metadata), sort_keys=True)),
    )
    if target.exists():  # Defend against a concurrent writer after preparation.
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"refusing to overwrite replay shard: {target}")
    temporary.replace(target)
    return {**asdict(metadata), "path": target.name, "sha256": sha256_file(target)}


class ReplayShard:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as archive:
            self.metadata = json.loads(str(archive["metadata"].item()))
            self._packed_features = archive["packed_features"]
            self._halfmove = archive["halfmove"]
            self._offsets = archive["policy_offsets"].astype(np.int64)
            self._action_ids = archive["action_ids"].astype(np.uint16)
            self._probabilities = archive["probabilities"].astype(np.float32)
            self._values = archive["values"].astype(np.float32)
        read_replay_metadata(self.path)
        if int(self.metadata.get("positions", -1)) != len(self):
            raise ValueError(
                "replay position count mismatch: "
                f"metadata={self.metadata.get('positions')}, arrays={len(self)}"
            )

    def __len__(self) -> int:
        return int(self._values.shape[0])

    def sample(self, index: int) -> ReplaySample:
        start, end = int(self._offsets[index]), int(self._offsets[index + 1])
        return ReplaySample(
            features=_unpack_features(
                self._packed_features[index : index + 1],
                self._halfmove[index : index + 1],
            )[0],
            action_ids=self._action_ids[start:end],
            probabilities=self._probabilities[start:end],
            value=float(self._values[index]),
        )


class ReplayDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, paths: Sequence[str | Path], *, capacity: int = 300_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        selected: list[tuple[Path, int]] = []
        remaining = capacity
        for path in reversed(paths):
            resolved = Path(path)
            metadata = read_replay_metadata(resolved)
            positions = int(metadata.get("positions", 0))
            if positions <= 0:
                raise ValueError(f"replay shard declares no positions: {resolved}")
            take = min(positions, remaining)
            selected.append((resolved, take))
            remaining -= take
            if remaining == 0:
                break

        retained: list[tuple[ReplayShard, int]] = []
        ordered_selected = list(reversed(selected))
        self.shard_paths = tuple(path for path, _take in ordered_selected)
        for path, take in ordered_selected:
            shard = ReplayShard(path)
            if take > len(shard):
                raise ValueError(f"replay metadata exceeds payload length: {path}")
            start = len(shard) - take
            retained.extend((shard, index) for index in range(start, len(shard)))
        self._samples = retained

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shard, sample_index = self._samples[index]
        sample = shard.sample(sample_index)
        policy = torch.zeros(2086, dtype=torch.float32)
        policy[torch.from_numpy(sample.action_ids.astype(np.int64))] = torch.from_numpy(
            sample.probabilities
        )
        return (
            torch.from_numpy(sample.features),
            policy,
            torch.tensor(sample.value, dtype=torch.float32),
        )

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        for index in range(len(self)):
            yield self[index]
