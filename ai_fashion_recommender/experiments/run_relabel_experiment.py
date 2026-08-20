"""5단계 학습 — 재라벨링한 캐시로 같은 레시피(R2_mixup)를 다시 학습한다.

BASE 대조군은 이미 있다: reports/runs/R2_mixup.pt (원본 캐시, 같은 분할, 같은 시드).
그래서 여기서는 D / K / DK 변형만 학습한다.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict

import torch

from eval_lib import PROJECT_DIR, REPORT_DIR, load_cache, split_train_dev, subset
from finetune import RECIPES, train_once

CACHE_DIR = PROJECT_DIR / "data" / "cache"
RELABEL_RUN_DIR = REPORT_DIR / "runs_relabel"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("D", "K", "DK", "D25", "D50"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()

    torch.set_num_threads(arguments.threads)
    recipe = next(r for r in RECIPES if r.name == "R2_mixup")

    train_cache = load_cache(CACHE_DIR / f"fashion_attributes_train_{arguments.variant}.pt")
    fit_indices, dev_indices = split_train_dev(train_cache)
    fit = subset(train_cache, fit_indices)
    dev = subset(train_cache, dev_indices)
    print(
        f"variant={arguments.variant} fit={len(fit['features'])} dev={len(dev['features'])} "
        f"detail_valid_fit={int(fit['valid']['detail'].sum())} "
        f"material_valid_fit={int(fit['valid']['material'].sum())}",
        flush=True,
    )

    runs = []
    for seed in range(arguments.seeds):
        started = time.time()
        run = train_once(recipe, seed, fit, dev, "cpu")
        mean_selection = sum(run["dev_selection"].values()) / len(run["dev_selection"])
        print(
            f"  {arguments.variant} seed={seed} dev_mean={mean_selection:.4f} "
            f"detail_dev={run['dev_selection']['detail']:.4f} "
            f"material_dev={run['dev_selection']['material']:.4f} "
            f"{time.time() - started:.0f}s",
            flush=True,
        )
        runs.append(run)

    RELABEL_RUN_DIR.mkdir(parents=True, exist_ok=True)
    output = RELABEL_RUN_DIR / f"{arguments.variant}.pt"
    torch.save({"config": asdict(recipe), "variant": arguments.variant, "runs": runs}, output)
    print(f"  saved: {output}", flush=True)


if __name__ == "__main__":
    main()
