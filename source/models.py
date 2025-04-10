from typing import Type, Callable

import torch
import torch.nn as nn


class SinModel(nn.Module):
    def __init__(self, activation: Type[nn.Module] | Callable[[], nn.Module], device: torch.device, neurons: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._model = nn.Sequential(
            nn.Linear(1, neurons),
            activation(),
            nn.Linear(neurons, neurons),
            activation(),
            nn.Linear(neurons, 1)
        )

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    @property
    def model(self) -> nn.Sequential:
        return self._model

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, activation=nn.ReLU, use_norm: bool = True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_norm else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_norm else nn.Identity()
        self.activation = activation()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels) if use_norm else nn.Identity()
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.activation(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, activation=nn.ReLU, use_norm: bool = True):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if use_norm else nn.Identity()
        self.activation = activation()

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, activation=activation, use_norm=use_norm)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, activation=activation, use_norm=use_norm)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, activation=activation, use_norm=use_norm)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, activation=activation, use_norm=use_norm)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, activation, use_norm: bool):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, activation, use_norm))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

if __name__ == '__main__':
    resnet = ResNet(BasicBlock, [2, 2, 2, 2], use_norm=False)
    sample = torch.randn((1, 3, 32, 32))

    print(resnet(sample))
