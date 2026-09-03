from pathlib import Path

from chessai.doctor import _is_playable_training_ready, collect_doctor_report


def test_doctor_executes_a_real_policy_value_forward(tmp_path: Path) -> None:
    report = collect_doctor_report(tmp_path)

    assert report["engine"]["initial_legal_moves"] == 44
    assert report["torch"]["installed"] is True
    probe = report["torch"]["inference_probe"]
    assert probe["ok"] is True
    assert probe["outputs_finite"] is True
    assert probe["logits_shape"] == [1, 2086]
    assert probe["value_shape"] == [1]
    assert probe["model_parameters"] > 0
    assert probe["elapsed_ms"] >= 0
    assert report["ready_for_local_smoke"] is True


def test_playable_training_readiness_contract_is_reported(tmp_path: Path) -> None:
    report = collect_doctor_report(tmp_path)

    assert report["playable_training_requirements"] == {
        "cpu_count_min": 12,
        "memory_gb_min": 32.0,
        "disk_free_gb_min": 25.0,
        "vram_gb_min": 24.0,
    }
    assert report["memory_gb"] is None or report["memory_gb"] > 0
    assert isinstance(report["ready_for_playable_training"], bool)


def test_playable_readiness_requires_disk_bf16_sm120_and_native() -> None:
    report = {
        "cpu_count": 12,
        "disk_free_gb": 25.0,
        "engine": {"native_backend": True, "selected_backend": "native"},
        "torch": {
            "cuda_available": True,
            "bf16_supported": True,
            "required_device_arch": "sm_120",
            "device_arch_compatible": True,
            "vram_gb": 24.0,
            "inference_probe": {"ok": True},
        },
    }
    assert _is_playable_training_ready(report, 32.0) is True

    report["disk_free_gb"] = 24.99
    assert _is_playable_training_ready(report, 32.0) is False
    report["disk_free_gb"] = 25.0
    report["torch"]["bf16_supported"] = False
    assert _is_playable_training_ready(report, 32.0) is False
    report["torch"]["bf16_supported"] = True
    report["torch"]["required_device_arch"] = "sm_90"
    assert _is_playable_training_ready(report, 32.0) is False
    report["torch"]["required_device_arch"] = "sm_120"
    report["engine"]["native_backend"] = False
    assert _is_playable_training_ready(report, 32.0) is False
