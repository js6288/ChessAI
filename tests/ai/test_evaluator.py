import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from chessai.ai.evaluator import BatchingEvaluator  # noqa: E402
from chessai.ai.model import ModelConfig, PolicyValueModel  # noqa: E402
from chessai.engine import GameState  # noqa: E402
from chessai.training.bootstrap import evaluate_supervised, resolve_precision  # noqa: E402


@pytest.mark.torch
def test_shared_inference_worker_batches_concurrent_actor_requests() -> None:
    evaluator = BatchingEvaluator(
        PolicyValueModel(ModelConfig.tiny()),
        "cpu",
        max_batch_size=4,
        max_wait_ms=100,
    )
    barrier = threading.Barrier(5)

    def evaluate_once() -> tuple[tuple[int, ...], float]:
        barrier.wait()
        logits, value = evaluator.evaluate(GameState.initial())
        return logits.shape, value

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(evaluate_once) for _ in range(4)]
        barrier.wait()
        results = [future.result(timeout=10) for future in futures]
    stats = evaluator.stats()
    evaluator.close()

    assert all(shape == (2086,) for shape, _value in results)
    assert stats["requests"] == 4
    assert stats["largest_batch"] >= 2
    assert stats["precision"] == "fp32"
    assert stats["batch_model_ms_p50"] >= 0
    assert stats["batch_model_ms_p95"] >= stats["batch_model_ms_p50"]
    assert stats["request_ms_p50"] >= stats["batch_model_ms_p50"]
    with pytest.raises(RuntimeError, match="closed"):
        evaluator.evaluate(GameState.initial())


def test_cpu_training_precision_contract() -> None:
    device = torch.device("cpu")
    assert resolve_precision(device, "auto") == "fp32"
    assert resolve_precision(device, "fp32") == "fp32"
    with pytest.raises(RuntimeError, match="BF16"):
        resolve_precision(device, "bf16")


@pytest.mark.torch
def test_supervised_evaluation_reports_legal_topk_and_phase_metrics(tmp_path: Path) -> None:
    dataset = tmp_path / "validation.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "game_id": "evaluation-fixture",
                "initial_fen": "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
                "moves": ["h2e2", "c6c5"],
                "result": "1/2-1/2",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    model = PolicyValueModel(ModelConfig.tiny())
    report = evaluate_supervised(
        model,
        dataset,
        device=torch.device("cpu"),
        precision="fp32",
        batch_size=2,
    )

    assert report["positions"] == 2
    assert 0 <= report["legal_top1_accuracy"] <= report["legal_top3_accuracy"]
    assert report["legal_top3_accuracy"] <= report["legal_top5_accuracy"] <= 1
    assert report["phases"]["opening"]["positions"] == 2
    assert report["phases"]["middlegame"]["positions"] == 0
    assert report["phases"]["endgame"]["positions"] == 0
