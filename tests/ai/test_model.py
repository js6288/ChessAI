import pytest

torch = pytest.importorskip("torch")

from chessai.ai.model import ModelConfig, PolicyValueModel, masked_policy_logits  # noqa: E402


@pytest.mark.torch
def test_tiny_model_forward_contract() -> None:
    model = PolicyValueModel(ModelConfig.tiny())
    inputs = torch.zeros((2, 117, 10, 9), dtype=torch.float32)
    logits, values = model(inputs)
    assert logits.shape == (2, 2086)
    assert values.shape == (2,)
    assert torch.all(values >= -1.0)
    assert torch.all(values <= 1.0)


@pytest.mark.torch
def test_masked_logits_reject_all_illegal_sample() -> None:
    logits = torch.zeros((1, 2086))
    with pytest.raises(ValueError, match="no legal action"):
        masked_policy_logits(logits, torch.zeros_like(logits, dtype=torch.bool))


@pytest.mark.torch
def test_masked_logits_preserve_legal_entries() -> None:
    logits = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    mask = torch.tensor([[True, False, True, False, False, True]])
    masked = masked_policy_logits(logits, mask)
    assert masked[0, 0] == 0
    assert masked[0, 2] == 2
    assert masked[0, 5] == 5
    assert masked[0, 1] < -1e30
