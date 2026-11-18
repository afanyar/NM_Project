"""CIFAR-10: завантаження, фіксований стратифікований спліт 45k/5k/10k, підвибірки."""
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import datasets

from .augment import build_train_transform, build_eval_transform

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def load_arrays(root, smoke=False):
    """Повертає (x_train, y_train, x_test, y_test): uint8 [N,32,32,3] та int64 [N]."""
    if smoke:  # випадкові дані - лише для перевірки, що код працює
        rng = np.random.default_rng(0)
        return (rng.integers(0, 256, (1200, 32, 32, 3), dtype=np.uint8),
                rng.integers(0, 10, 1200),
                rng.integers(0, 256, (400, 32, 32, 3), dtype=np.uint8),
                rng.integers(0, 10, 400))
    tr = datasets.CIFAR10(root, train=True, download=True)
    te = datasets.CIFAR10(root, train=False, download=True)
    return tr.data, np.array(tr.targets), te.data, np.array(te.targets)


def stratified_split(y, n_val, seed):
    """Фіксований стратифікований поділ на train/val (однаковий для всіх запусків)."""
    rng = np.random.default_rng(seed)
    n_cls = int(y.max()) + 1
    per_class = n_val // n_cls
    val_idx = []
    for c in range(n_cls):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        val_idx.append(idx[:per_class])
    val_idx = np.sort(np.concatenate(val_idx))
    mask = np.ones(len(y), dtype=bool)
    mask[val_idx] = False
    return np.where(mask)[0], val_idx


def stratified_subset(idx, y, frac, seed):
    """Стратифікована підвибірка частки frac від train-індексів (для sample efficiency)."""
    if frac >= 1.0:
        return idx
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c]
        k = max(1, int(round(len(ci) * frac)))
        out.append(rng.choice(ci, k, replace=False))
    return np.sort(np.concatenate(out))


class ArrayDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = Image.fromarray(self.x[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.y[i])


def make_loaders(cfg):
    """Будує всі DataLoader-и для одного запуску."""
    dcfg, tcfg = cfg["data"], cfg["train"]
    seed, frac, smoke = cfg["seed"], cfg["train_frac"], cfg["smoke"]

    x_tr, y_tr, x_te, y_te = load_arrays(dcfg["root"], smoke)
    val_size = 200 if smoke else dcfg["val_size"]
    train_idx, val_idx = stratified_split(y_tr, val_size, dcfg["split_seed"])
    train_idx = stratified_subset(train_idx, y_tr, frac, seed)

    eval_tf = build_eval_transform()
    train_tf = build_train_transform(cfg["aug"])

    n_clean = min(tcfg["clean_train_eval_size"], len(train_idx))
    clean_idx = np.sort(np.random.default_rng(0).choice(train_idx, n_clean, replace=False))

    nw = 0 if smoke else dcfg["num_workers"]
    bs = tcfg["batch_size"]
    g = torch.Generator().manual_seed(seed)  # детермінований порядок батчів і аугментацій
    kw = dict(num_workers=nw, pin_memory=torch.cuda.is_available(),
              persistent_workers=nw > 0)

    loaders = {
        "train": DataLoader(ArrayDataset(x_tr[train_idx], y_tr[train_idx], train_tf),
                            batch_size=bs, shuffle=True, drop_last=True, generator=g, **kw),
        "train_clean": DataLoader(ArrayDataset(x_tr[clean_idx], y_tr[clean_idx], eval_tf),
                                  batch_size=512, shuffle=False, **kw),
        "val": DataLoader(ArrayDataset(x_tr[val_idx], y_tr[val_idx], eval_tf),
                          batch_size=512, shuffle=False, **kw),
        "test": DataLoader(ArrayDataset(x_te, y_te, eval_tf),
                           batch_size=512, shuffle=False, **kw),
    }
    return loaders, len(train_idx)
