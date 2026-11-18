"""Візуалізація аугментацій (для звіту): верхній рядок - оригінали, далі кожен рядок -
ті самі зображення після одного пресета.
  python visualize_augs.py --augs flip_crop cutout_16 rotation_30 mixup_a1 cutmix_a1
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.augment import build_eval_transform, build_train_transform, mix_batch
from src.data import load_arrays
from src.utils import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augs", nargs="+", default=None, help="за замовчуванням - усі пресети")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out", default="results/aug_examples.png")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    presets = load_yaml("configs/augmentations.yaml")
    names = a.augs or list(presets)
    x, _, _, _ = load_arrays(a.data_root, a.smoke)
    torch.manual_seed(0)
    np.random.seed(0)

    pil = [Image.fromarray(x[i]) for i in range(a.n)]
    rows = [("original", torch.stack([build_eval_transform()(im) for im in pil]))]
    for name in names:
        aug = presets[name] or {}
        tf = build_train_transform(aug)
        imgs = torch.stack([tf(im) for im in pil])
        if aug.get("mixup_alpha") or aug.get("cutmix_alpha"):
            imgs, _, _, _ = mix_batch(imgs, torch.arange(a.n), aug)
        rows.append((name, imgs))

    fig, axes = plt.subplots(len(rows), a.n, figsize=(1.3 * a.n, 1.45 * len(rows)))
    for r, (name, imgs) in enumerate(rows):
        for c in range(a.n):
            axes[r, c].imshow(imgs[c].permute(1, 2, 0).clamp(0, 1).numpy())
            axes[r, c].axis("off")
        axes[r, 0].set_title(name, fontsize=7, loc="left")
    plt.tight_layout()
    plt.savefig(a.out, dpi=150)
    print("Збережено", a.out)


if __name__ == "__main__":
    main()
