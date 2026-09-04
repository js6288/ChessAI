import json

import numpy as np
import pytest

pytest.importorskip("torch")

from chessai.ai.features import encode_state
from chessai.engine import GameState
from chessai.training.replay import (
    ReplayDataset,
    ReplaySample,
    ReplayShard,
    combine_packed_replay,
    pack_replay_samples,
    read_replay_metadata,
    save_replay_shard,
)
from chessai.training.selfplay import PendingSample, SelfPlayConfig, run_selfplay


def test_sparse_replay_shard_round_trip(tmp_path) -> None:
    state = GameState.initial()
    sample = ReplaySample(
        features=encode_state(state),
        action_ids=np.asarray([2, 9], dtype=np.uint16),
        probabilities=np.asarray([0.25, 0.75], dtype=np.float32),
        value=1.0,
    )
    path = tmp_path / "replay.npz"
    metadata = save_replay_shard(
        path, [sample], network_hash="test", simulations=8, seed=1, games=1
    )
    assert metadata["positions"] == 1
    assert metadata["search_version"] == "gumbel-completed-q-v2"
    assert read_replay_metadata(path)["network_hash"] == "test"
    shard = ReplayShard(path)
    restored = shard.sample(0)
    assert np.array_equal(restored.features, sample.features)
    assert restored.action_ids.tolist() == [2, 9]
    assert restored.probabilities.tolist() == pytest.approx([0.25, 0.75], abs=1e-3)
    dataset = ReplayDataset([path])
    features, policy, value = dataset[0]
    assert features.shape == (117, 10, 9)
    assert policy.sum().item() == pytest.approx(1.0)
    assert value.item() == 1.0

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_replay_shard(path, [sample], network_hash="test", simulations=8, seed=1, games=1)


def test_replay_rejects_a_different_search_target_version(tmp_path, monkeypatch) -> None:
    state = GameState.initial()
    sample = ReplaySample(
        features=encode_state(state),
        action_ids=np.asarray([2], dtype=np.uint16),
        probabilities=np.asarray([1.0], dtype=np.float32),
        value=0.0,
    )
    path = tmp_path / "replay.npz"
    save_replay_shard(path, [sample], network_hash="test", simulations=8, seed=1, games=1)

    monkeypatch.setattr("chessai.training.replay.SEARCH_VERSION", "future-search")
    with pytest.raises(ValueError, match="search_version"):
        read_replay_metadata(path)


def test_packed_actor_payload_combines_without_changing_replay(tmp_path) -> None:
    state = GameState.initial()
    samples = [
        ReplaySample(
            features=encode_state(state),
            action_ids=np.asarray([2, 9], dtype=np.uint16),
            probabilities=np.asarray([0.4, 0.6], dtype=np.float32),
            value=float(value),
        )
        for value in (-1, 1)
    ]
    combined = combine_packed_replay(
        [pack_replay_samples(samples[:1]), pack_replay_samples(samples[1:])]
    )
    assert combined.positions == 2
    assert combined.policy_offsets.tolist() == [0, 2, 4]


def test_replay_capacity_loads_only_the_newest_shards(tmp_path, monkeypatch) -> None:
    state = GameState.initial()
    sample = ReplaySample(
        features=encode_state(state),
        action_ids=np.asarray([2], dtype=np.uint16),
        probabilities=np.asarray([1.0], dtype=np.float32),
        value=0.0,
    )
    paths = []
    for index in range(3):
        path = tmp_path / f"replay-{index}.npz"
        save_replay_shard(
            path,
            [sample, sample],
            network_hash="test",
            simulations=8,
            seed=index,
            games=1,
        )
        paths.append(path)

    import chessai.training.replay as replay_module

    original = replay_module.ReplayShard
    loaded: list[str] = []

    class TrackingReplayShard(original):
        def __init__(self, path):
            loaded.append(path.name)
            super().__init__(path)

    monkeypatch.setattr(replay_module, "ReplayShard", TrackingReplayShard)
    dataset = ReplayDataset(paths, capacity=3)

    assert len(dataset) == 3
    assert loaded == ["replay-1.npz", "replay-2.npz"]


def test_selfplay_resume_appends_shards_and_seeds(tmp_path, monkeypatch) -> None:
    state = GameState.initial()
    observed_seeds: list[int] = []

    def fake_game(evaluator, *, simulations, sample_until_ply, max_ply, seed):
        del evaluator, simulations, sample_until_ply, max_ply
        observed_seeds.append(seed)
        pending = PendingSample(
            features=encode_state(state),
            action_ids=np.asarray([2], dtype=np.uint16),
            probabilities=np.asarray([1.0], dtype=np.float32),
            perspective=state.side_to_move,
        )
        sample = ReplaySample(
            features=pending.features,
            action_ids=pending.action_ids,
            probabilities=pending.probabilities,
            value=0.0,
        )
        return [sample], {"plies": 1, "status": "draw", "winner": None, "reason": "test"}

    monkeypatch.setattr("chessai.training.selfplay.play_selfplay_game", fake_game)
    config = SelfPlayConfig(games=1, simulations=8, max_ply=24, shard_games=1, seed=19)
    first = run_selfplay(tmp_path, config=config)
    second = run_selfplay(tmp_path, config=config)

    assert observed_seeds == [19, 20]
    assert first["games"] == 1
    assert second["games"] == 2
    assert second["resume"] == {
        "next_game_index": 2,
        "next_seed": 21,
        "next_shard_index": 2,
    }
    assert sorted(path.name for path in tmp_path.glob("replay-*.npz")) == [
        "replay-000000.npz",
        "replay-000001.npz",
    ]


def test_selfplay_persists_each_complete_shard_before_a_later_game_fails(
    tmp_path, monkeypatch
) -> None:
    state = GameState.initial()

    def generated_sample():
        return [
            ReplaySample(
                features=encode_state(state),
                action_ids=np.asarray([2], dtype=np.uint16),
                probabilities=np.asarray([1.0], dtype=np.float32),
                value=0.0,
            )
        ], {"plies": 1, "status": "draw", "winner": None, "reason": "test"}

    def fail_second(evaluator, *, simulations, sample_until_ply, max_ply, seed):
        del evaluator, simulations, sample_until_ply, max_ply
        if seed == 20:
            raise RuntimeError("simulated actor failure")
        return generated_sample()

    monkeypatch.setattr("chessai.training.selfplay.play_selfplay_game", fail_second)
    first_config = SelfPlayConfig(games=2, simulations=8, max_ply=24, shard_games=1, seed=19)
    with pytest.raises(RuntimeError, match="actor failure"):
        run_selfplay(tmp_path, config=first_config)

    progress = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert progress["games"] == 1
    assert progress["positions"] == 1
    assert [shard["path"] for shard in progress["shards"]] == ["replay-000000.npz"]

    monkeypatch.setattr(
        "chessai.training.selfplay.play_selfplay_game",
        lambda evaluator, *, simulations, sample_until_ply, max_ply, seed: generated_sample(),
    )
    resumed = run_selfplay(
        tmp_path,
        config=SelfPlayConfig(games=1, simulations=8, max_ply=24, shard_games=1, seed=19),
    )
    assert resumed["games"] == 2
    assert [shard["seed"] for shard in resumed["shards"]] == [19, 20]


def test_selfplay_position_target_stops_after_one_actor_batch_and_resumes_noop(
    tmp_path, monkeypatch
) -> None:
    state = GameState.initial()
    observed_seeds: list[int] = []

    def fake_game(evaluator, *, simulations, sample_until_ply, max_ply, seed):
        del evaluator, simulations, sample_until_ply, max_ply
        observed_seeds.append(seed)
        sample = ReplaySample(
            features=encode_state(state),
            action_ids=np.asarray([2], dtype=np.uint16),
            probabilities=np.asarray([1.0], dtype=np.float32),
            value=0.0,
        )
        return [sample, sample], {
            "plies": 2,
            "status": "draw",
            "winner": None,
            "reason": "test",
        }

    monkeypatch.setattr("chessai.training.selfplay.play_selfplay_game", fake_game)
    config = SelfPlayConfig(
        games=None,
        target_positions=5,
        actors=2,
        simulations=1,
        max_ply=2,
        shard_games=1,
        seed=31,
    )
    first = run_selfplay(tmp_path, config=config)
    assert first["positions"] == 8
    assert first["positions"] - 5 <= config.actors * config.max_ply
    assert sorted(observed_seeds) == [31, 32, 33, 34]

    second = run_selfplay(tmp_path, config=config)
    assert second["positions"] == 8
    assert sorted(observed_seeds) == [31, 32, 33, 34]


def test_selfplay_resume_rejects_a_tampered_shard(tmp_path, monkeypatch) -> None:
    state = GameState.initial()

    def generated_sample():
        sample = ReplaySample(
            features=encode_state(state),
            action_ids=np.asarray([2], dtype=np.uint16),
            probabilities=np.asarray([1.0], dtype=np.float32),
            value=0.0,
        )
        return [sample], {"plies": 1, "status": "draw", "winner": None, "reason": "test"}

    monkeypatch.setattr(
        "chessai.training.selfplay.play_selfplay_game",
        lambda evaluator, *, simulations, sample_until_ply, max_ply, seed: generated_sample(),
    )
    config = SelfPlayConfig(games=1, simulations=1, max_ply=2, shard_games=1)
    run_selfplay(tmp_path, config=config)
    with (tmp_path / "replay-000000.npz").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="shard hash mismatch"):
        run_selfplay(tmp_path, config=config)
