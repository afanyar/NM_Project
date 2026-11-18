"""Запуск плану експериментів (configs/plans.yaml). Вже виконані запуски пропускаються,
тому після обриву сесії Kaggle/Colab достатньо запустити ту саму команду ще раз.

  python run_grid.py --plan main --out results
  python run_grid.py --plan sample_efficiency --out results
"""
import argparse
import gc

import torch

from src.train import build_cfg, get_parser, run_experiment
from src.utils import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="smoke")
    ap.add_argument("--plans_file", default="configs/plans.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    plan = load_yaml(a.plans_file)[a.plan]
    # seed - зовнішній цикл: при обриві сесії маємо повні набори для менших seed
    jobs = [(s, f, aug) for s in plan["seeds"] for f in plan["fracs"] for aug in plan["augs"]]
    print(f"План '{a.plan}': {len(jobs)} запусків")

    for i, (seed, frac, aug) in enumerate(jobs, 1):
        print(f"\n===== {i}/{len(jobs)} =====")
        args = get_parser().parse_args(
            ["--aug", aug, "--seed", str(seed), "--train_frac", str(frac), "--out", a.out]
            + (["--epochs", str(a.epochs)] if a.epochs else [])
            + (["--model", a.model] if a.model else [])
            + (["--data_root", a.data_root] if a.data_root else [])
            + (["--smoke"] if a.smoke else []))
        run_experiment(build_cfg(args))
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
