"""Аугментації: побудова пайплайна трансформацій + batch-level Mixup/CutMix.

Порядок для одного зображення: PIL-операції (crop, flip, rotation, jitter, RandAugment)
-> ToTensor ([0,1]) -> Cutout. Нормалізація робиться ВСЕРЕДИНІ моделі (див. models.py),
тому Cutout заповнюється середнім кольором датасету (еквівалент нуля після нормалізації,
як у статті DeVries & Taylor, 2017).
"""
import numpy as np
import torch
from torchvision import transforms as T

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class Cutout:
    """Cutout (DeVries & Taylor, 2017): квадрат size x size із центром у випадковій точці
    (може частково виходити за межі зображення), заповнюється середнім кольором."""

    def __init__(self, size, fill=CIFAR_MEAN):
        self.size = int(size)
        self.fill = torch.tensor(fill).view(3, 1, 1)

    def __call__(self, img):
        _, h, w = img.shape
        cy = torch.randint(h, (1,)).item()
        cx = torch.randint(w, (1,)).item()
        y1, y2 = max(cy - self.size // 2, 0), min(cy + self.size // 2, h)
        x1, x2 = max(cx - self.size // 2, 0), min(cx + self.size // 2, w)
        img = img.clone()
        img[:, y1:y2, x1:x2] = self.fill
        return img

    def __repr__(self):
        return f"Cutout(size={self.size})"


def build_train_transform(aug: dict):
    ops = []
    if aug.get("crop_padding", 0) > 0:
        ops.append(T.RandomCrop(32, padding=aug["crop_padding"]))
    if aug.get("flip", False):
        ops.append(T.RandomHorizontalFlip())
    if aug.get("rotation", 0) > 0:
        ops.append(T.RandomRotation(aug["rotation"]))
    if aug.get("color_jitter", 0) > 0:
        s = aug["color_jitter"]
        ops.append(T.ColorJitter(brightness=s, contrast=s, saturation=s))
    if aug.get("randaugment"):
        n, m = aug["randaugment"]
        ops.append(T.RandAugment(num_ops=n, magnitude=m))
    ops.append(T.ToTensor())
    if aug.get("cutout", 0) > 0:
        ops.append(Cutout(aug["cutout"]))
    return T.Compose(ops)


def build_eval_transform():
    return T.ToTensor()


def mix_batch(x, y, aug: dict):
    """Mixup (Zhang et al., 2018) / CutMix (Yun et al., 2019) на рівні батча.

    Повертає (x_mixed, y_a, y_b, lam); loss = lam*CE(out, y_a) + (1-lam)*CE(out, y_b).
    Якщо задано і mixup, і cutmix - на кожен батч випадково обирається один з них (p=0.5).
    Використовує глобальний numpy/torch RNG (його фіксує set_seed).
    """
    ma = aug.get("mixup_alpha", 0.0)
    ca = aug.get("cutmix_alpha", 0.0)
    if ma <= 0 and ca <= 0:
        return x, y, y, 1.0

    perm = torch.randperm(x.size(0), device=x.device)
    use_cutmix = ca > 0 and (ma <= 0 or np.random.rand() < 0.5)

    if use_cutmix:
        lam = np.random.beta(ca, ca)
        h, w = x.shape[2], x.shape[3]
        r = np.sqrt(1.0 - lam)
        ch, cw = int(h * r), int(w * r)
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = int(np.clip(cy - ch // 2, 0, h)), int(np.clip(cy + ch // 2, 0, h))
        x1, x2 = int(np.clip(cx - cw // 2, 0, w)), int(np.clip(cx + cw // 2, 0, w))
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)  # реальна частка оригіналу
    else:
        lam = float(np.random.beta(ma, ma))
        x = lam * x + (1.0 - lam) * x[perm]

    return x, y, y[perm], float(lam)
