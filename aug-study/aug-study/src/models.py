"""ResNet для CIFAR (3x3 stem без maxpool). Нормалізація входу - всередині моделі."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .augment import CIFAR_MEAN, CIFAR_STD


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class CifarResNet(nn.Module):
    def __init__(self, blocks, widths, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        layers, in_c = [], widths[0]
        for i, (n, w) in enumerate(zip(blocks, widths)):
            for j in range(n):
                layers.append(BasicBlock(in_c, w, stride=2 if (j == 0 and i > 0) else 1))
                in_c = w
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(in_c, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


ARCHS = {
    "resnet20": ([3, 3, 3], [16, 32, 64]),            # ~0.27M параметрів, дуже швидка
    "resnet18": ([2, 2, 2, 2], [64, 128, 256, 512]),  # ~11M параметрів
}


def build_model(name, num_classes=10):
    blocks, widths = ARCHS[name]
    return nn.Sequential(Normalize(CIFAR_MEAN, CIFAR_STD), CifarResNet(blocks, widths, num_classes))
