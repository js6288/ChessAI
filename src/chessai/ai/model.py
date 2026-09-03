"""PyTorch policy/value network and checkpoint-facing model metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

from chessai.ai.features import INPUT_PLANES
from chessai.engine.vocabulary import action_labels

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised by doctor in core-only installs
    raise RuntimeError(
        "PyTorch is required for chessai.ai.model; install the 'train' extra"
    ) from exc


@dataclass(frozen=True, slots=True)
class ModelConfig:
    input_planes: int = INPUT_PLANES
    channels: int = 128
    residual_blocks: int = 10
    policy_channels: int = 32
    value_channels: int = 8
    value_hidden: int = 256
    action_size: int = 2086

    def __post_init__(self) -> None:
        if self.input_planes != INPUT_PLANES:
            raise ValueError(f"input_planes must match feature contract: {INPUT_PLANES}")
        if self.action_size != len(action_labels()):
            raise ValueError(f"action_size must match vocabulary: {len(action_labels())}")
        if (
            min(
                self.channels,
                self.residual_blocks,
                self.policy_channels,
                self.value_channels,
                self.value_hidden,
            )
            <= 0
        ):
            raise ValueError("all model dimensions must be positive")

    @classmethod
    def tiny(cls) -> ModelConfig:
        return cls(
            channels=32, residual_blocks=2, policy_channels=8, value_channels=4, value_hidden=64
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        outputs = self.activation(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return cast(Tensor, self.activation(outputs + residual))


class PolicyValueModel(nn.Module):
    """AlphaZero-style residual trunk with policy and scalar value heads."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.input_planes, cfg.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            *(ResidualBlock(cfg.channels) for _ in range(cfg.residual_blocks))
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(cfg.channels, cfg.policy_channels, 1, bias=False),
            nn.BatchNorm2d(cfg.policy_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(cfg.policy_channels * 10 * 9, cfg.action_size),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(cfg.channels, cfg.value_channels, 1, bias=False),
            nn.BatchNorm2d(cfg.value_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(cfg.value_channels * 10 * 9, cfg.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.value_hidden, 1),
            nn.Tanh(),
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        trunk = self.trunk(self.stem(inputs))
        policy_logits = self.policy_head(trunk)
        value = self.value_mlp(self.value_conv(trunk)).squeeze(-1)
        return policy_logits, value

    @torch.inference_mode()
    def predict(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        was_training = self.training
        self.eval()
        logits, values = self(inputs)
        if was_training:
            self.train()
        return logits, values

    def metadata(self) -> dict[str, Any]:
        return {"architecture": "resnet-policy-value", "config": self.config.to_dict()}


def masked_policy_logits(logits: Tensor, legal_mask: Tensor) -> Tensor:
    if logits.shape != legal_mask.shape:
        raise ValueError(
            f"logits and legal mask shapes differ: {logits.shape} vs {legal_mask.shape}"
        )
    if legal_mask.dtype is not torch.bool:
        raise TypeError("legal_mask must be boolean")
    if torch.any(~torch.any(legal_mask, dim=-1)):
        raise ValueError("at least one sample has no legal action")
    return logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
