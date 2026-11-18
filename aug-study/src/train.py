"""Навчання одного експерименту.

Приклад:  python -m src.train --aug flip_crop --seed 0 --train_frac 1.0
Перевірка: python -m src.train --aug cutout_8 --smoke --epochs 2
"""
import argparse
import copy
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .augment import mix_batch
from .data import make_loaders
from .evaluate import accuracy, full_test_report, predict
from .models import build_model
from .utils import load_yaml, make_run_id, run_dir, save_json, set_seed


def build_cfg(args):
    cfg = load_yaml(args.base)
    presets = load_yaml(args.augs_file)
    if args.aug not in presets:
        raise KeyError(f"Невідомий пресет '{args.aug}'. Доступні: {list(presets)}")
    cfg["aug_name"] = args.aug
    cfg["aug"] = presets[args.aug] or {}
    cfg["seed"], cfg["train_frac"], cfg["smoke"] = args.seed, args.train_frac, args.smoke
    cfg["out"] = args.out
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.model is not None:
        cfg["model"]["name"] = args.model
    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root
    return cfg


def lr_at(it, warm, total, base_lr):
    if it < warm:
        return base_lr * (it + 1) / warm
    p = (it - warm) / max(1, total - warm)
    return base_lr * 0.5 * (1 + math.cos(math.pi * p))


def run_experiment(cfg, force=False, save_ckpt=False):
    tcfg = cfg["train"]
    run_id = make_run_id(cfg["aug_name"], cfg["train_frac"], cfg["seed"])
    rdir = run_dir(cfg["out"], run_id)
    if (rdir / "metrics.json").exists() and not force:
        print(f"[skip] {run_id} вже виконано")
        return None
    rdir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, rdir / "config.json")

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(tcfg["amp"]) and device.type == "cuda"

    loaders, n_train = make_loaders(cfg)
    model = build_model(cfg["model"]["name"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{run_id}] device={device} n_train={n_train} params={n_params/1e6:.2f}M aug={cfg['aug']}")

    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])
    opt = torch.optim.SGD(model.parameters(), lr=tcfg["lr"], momentum=tcfg["momentum"],
                          nesterov=tcfg["nesterov"], weight_decay=tcfg["weight_decay"])
    scaler = (torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=amp))

    steps = len(loaders["train"])
    total_it, warm_it = tcfg["epochs"] * steps, tcfg["warmup_epochs"] * steps
    it = 0
    history, best = [], {"val_acc": -1, "epoch": -1}
    t_start = time.time()

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        t0, run_loss, n_seen = time.time(), 0.0, 0
        for x, y in loaders["train"]:
            for g in opt.param_groups:
                g["lr"] = lr_at(it, warm_it, total_it, tcfg["lr"])
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, y_a, y_b, lam = mix_batch(x, y, cfg["aug"])
            with torch.autocast(device_type=device.type, enabled=amp):
                out = model(x)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * len(y)
            n_seen += len(y)
            it += 1
        train_time = time.time() - t0

        _, p, t = predict(model, loaders["train_clean"], device, amp)
        train_clean_acc = accuracy(p, t)
        val_loss, p, t = predict(model, loaders["val"], device, amp)
        val_acc = accuracy(p, t)
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "epoch": epoch}

        row = dict(epoch=epoch, lr=opt.param_groups[0]["lr"], train_loss=run_loss / n_seen,
                   train_clean_acc=train_clean_acc, val_loss=val_loss, val_acc=val_acc,
                   epoch_train_time_s=train_time)
        history.append(row)
        pd.DataFrame(history).to_csv(rdir / "epochs.csv", index=False)
        print(f"  ep {epoch:3d}/{tcfg['epochs']} loss={row['train_loss']:.3f} "
              f"train_acc={train_clean_acc:.3f} val_acc={val_acc:.3f} ({train_time:.1f}s)")

    # Тест - ОДИН раз, фінальна модель (без вибору чекпойнта за тестом).
    test_report, cm = full_test_report(model, loaders["test"], device, amp)
    np.savetxt(rdir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    if save_ckpt:
        torch.save(copy.deepcopy(model.state_dict()), rdir / "model.pt")

    last = history[-1]
    metrics = {
        "run_id": run_id, "aug_name": cfg["aug_name"], "seed": cfg["seed"],
        "train_frac": cfg["train_frac"], "n_train": n_train, "model": cfg["model"]["name"],
        "epochs": tcfg["epochs"], "params": n_params,
        "final_train_clean_acc": last["train_clean_acc"], "final_val_acc": last["val_acc"],
        "gen_gap": last["train_clean_acc"] - last["val_acc"],
        "best_val_acc": best["val_acc"], "best_val_epoch": best["epoch"],
        "total_train_time_s": sum(h["epoch_train_time_s"] for h in history),
        "wall_time_s": time.time() - t_start,
        **test_report,
    }
    save_json(metrics, rdir / "metrics.json")
    print(f"[{run_id}] test_acc={metrics['test_acc']:.4f} macro_f1={metrics['test_macro_f1']:.4f} "
          f"robust_mean={metrics['robust_mean']:.4f}")
    return metrics


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--aug", default="baseline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_frac", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--data_root", default=None)
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--augs_file", default="configs/augmentations.yaml")
    p.add_argument("--out", default="results")
    p.add_argument("--smoke", action="store_true", help="випадкові дані, швидка перевірка коду")
    p.add_argument("--force", action="store_true", help="перезаписати вже виконаний запуск")
    p.add_argument("--save_ckpt", action="store_true")
    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    run_experiment(build_cfg(args), force=args.force, save_ckpt=args.save_ckpt)
