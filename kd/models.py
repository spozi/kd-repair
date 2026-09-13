"""Model adapters expose logits and named semantic stages through one contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torchvision import models


@dataclass
class ModelOutput:
    logits: Tensor
    features: dict[str, Tensor] = field(default_factory=dict)


class VisionModel(nn.Module, ABC):
    feature_channels: dict[str, int]

    @abstractmethod
    def forward(self, images: Tensor, *, return_features: bool = False) -> ModelOutput:
        """Return unnormalized class logits and optionally named spatial features."""


class TinyCNN(VisionModel):
    """Small offline models for testing the complete experiment lifecycle."""

    def __init__(self, channels: tuple[int, ...], num_classes: int):
        super().__init__()
        stages = {}
        incoming = 3
        for i, outgoing in enumerate(channels, start=1):
            stages[f"stage{i}"] = nn.Sequential(
                nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(outgoing), nn.ReLU(),
            )
            incoming = outgoing
        self.stages = nn.ModuleDict(stages)
        self.feature_channels = {f"stage{i}": c for i, c in enumerate(channels, start=1)}
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, images: Tensor, *, return_features: bool = False) -> ModelOutput:
        features = {}
        for name, stage in self.stages.items():
            images = stage(images)
            if return_features:
                features[name] = images
        return ModelOutput(self.classifier(self.pool(images).flatten(1)), features)


class ResNetAdapter(VisionModel):
    """Adapter for torchvision ResNets, including differing stage widths/depths."""

    def __init__(self, name: str, num_classes: int):
        super().__init__()
        builders = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50}
        self.backbone = builders[name](weights=None, num_classes=num_classes)
        expansion = 4 if name == "resnet50" else 1
        self.feature_channels = {f"stage{i}": width * expansion
                                 for i, width in enumerate((64, 128, 256, 512), start=1)}

    def forward(self, images: Tensor, *, return_features: bool = False) -> ModelOutput:
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(images))))
        features = {}
        for i, layer in enumerate((b.layer1, b.layer2, b.layer3, b.layer4), start=1):
            x = layer(x)
            if return_features:
                features[f"stage{i}"] = x
        return ModelOutput(b.fc(torch.flatten(b.avgpool(x), 1)), features)


class CifarCNN(VisionModel):
    """Six-convolution classifier preserving spatial detail in 32x32 images."""

    def __init__(self, width: int, num_classes: int):
        super().__init__()
        self.stages = nn.ModuleDict()
        self.feature_channels = {}
        incoming = 3
        for i, outgoing in enumerate((width, width * 2, width * 4), start=1):
            self.stages[f"stage{i}"] = nn.Sequential(
                nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                nn.BatchNorm2d(outgoing), nn.ReLU(),
                nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                nn.BatchNorm2d(outgoing), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.feature_channels[f"stage{i}"] = outgoing
            incoming = outgoing
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Linear(incoming * 4, num_classes)

    def forward(self, images: Tensor, *, return_features: bool = False) -> ModelOutput:
        features = {}
        for name, stage in self.stages.items():
            images = stage(images)
            if return_features:
                features[name] = images
        return ModelOutput(self.classifier(self.pool(images).flatten(1)), features)


MODEL_NAMES = ("tiny_small", "tiny_medium", "tiny_large", "resnet18", "resnet34", "resnet50",
               "cifar_student", "cifar_teacher")


def create_model(name: str, num_classes: int) -> VisionModel:
    """A single factory keeps architecture selection out of training and losses."""
    tiny_widths = {"tiny_small": (16, 32, 64), "tiny_medium": (32, 64, 128),
                   "tiny_large": (64, 128, 256)}
    if name in tiny_widths:
        return TinyCNN(tiny_widths[name], num_classes)
    if name in {"cifar_student", "cifar_teacher"}:
        return CifarCNN(16 if name == "cifar_student" else 32, num_classes)
    if name in MODEL_NAMES:
        return ResNetAdapter(name, num_classes)
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
