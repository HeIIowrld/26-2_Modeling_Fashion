"""2단계 — 약한 영역 기준 미세조정.

진단 결론
- train 0.908 vs val 0.748 → 3.4M 파라미터 헤드가 4,789 표본을 외우는 과적합이 주원인.
- 대표 지표(accuracy/micro-F1)가 소수 라벨 붕괴를 가린다. 선택 기준을 macro-F1과 섞어야 한다.
- 기존 학습은 val로 early stopping과 임계값을 고르고 같은 val로 보고했다.

이 스크립트의 규칙
- 학습은 train_fit(4,071)만 사용한다.
- epoch 선택, 레시피 선택, 임계값 선택은 train_dev(718)에서만 한다.
- val(1,195)은 마지막 보고에만 쓴다.
- 헤드 구조(LayerNorm→Linear(768,256)→GELU→Dropout→Linear(256,C))는 그대로 둔다.
  → 태스크별 헤드가 독립이므로 레시피가 달라도 태스크 단위로 조립할 수 있고
    기존 체크포인트 포맷·main.ipynb와 100% 호환된다.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, field, asdict

import torch
import torch.nn.functional as functional

from eval_lib import (
    REPORT_DIR,
    TRAIN_CACHE,
    VAL_CACHE,
    evaluate_logits,
    head_logits,
    load_cache,
    overall_summary,
    split_train_dev,
    subset,
)
from fashion_attribute_model import build_attribute_heads
from fashion_attribute_schema import ATTRIBUTE_TASKS

HIDDEN_DIM = 256  # 체크포인트 호환을 위해 고정
MULTI_THRESHOLD_GRID = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


@dataclass
class Recipe:
    name: str
    dropout: float = 0.3
    weight_decay: float = 1e-4
    learning_rate: float = 5e-4
    epochs: int = 100
    batch_size: int = 256
    # hidden_dim은 체크포인트에 저장·복원되므로 바꿔도 기존 로더와 호환된다.
    hidden_dim: int = HIDDEN_DIM
    label_smoothing: float = 0.0
    noise_sigma: float = 0.0
    mixup_alpha: float = 0.0
    focal_gamma: float = 0.0
    class_weight_power: float = 0.5  # 0.5 = 기존 √역빈도, 1.0 = 역빈도
    cosine_schedule: bool = True
    seeds: tuple[int, ...] = (0, 1, 2)


RECIPES = [
    # 기존 설정 재현(정직한 분할에서만). 비교 기준선.
    Recipe("R0_replica", dropout=0.3, weight_decay=1e-4, learning_rate=5e-4, cosine_schedule=False),
    # 과적합 억제 1: 드롭아웃·weight decay 강화 + 임베딩 가우시안 노이즈
    Recipe("R1_strongreg", dropout=0.5, weight_decay=1e-2, noise_sigma=0.05),
    # 과적합 억제 2: mixup
    Recipe("R2_mixup", dropout=0.4, weight_decay=1e-3, mixup_alpha=0.4),
    # 과적합 억제 3: 최대 규제 + 라벨 스무딩
    Recipe("R3_heavy", dropout=0.6, weight_decay=5e-2, label_smoothing=0.10, noise_sigma=0.08),
    # 소수 라벨 회복: 역빈도 가중치 강화 + focal + mixup
    Recipe(
        "R4_rare",
        dropout=0.4,
        weight_decay=1e-2,
        label_smoothing=0.05,
        mixup_alpha=0.2,
        focal_gamma=1.5,
        class_weight_power=0.85,
    ),
    # 노이즈 + mixup 동시 + 낮은 LR 장기 학습
    Recipe(
        "R5_slow",
        dropout=0.5,
        weight_decay=2e-2,
        learning_rate=3e-4,
        epochs=150,
        label_smoothing=0.05,
        noise_sigma=0.05,
        mixup_alpha=0.2,
    ),
]

# 2라운드 — 1라운드에서 mixup(R2)과 강한 규제(R1)가 dev 상위였다.
# 남은 근본 원인은 헤드 용량이다. 4,789 표본에 256 폭 MLP(3.4M)는 과하다.
RECIPES += [
    Recipe("C1_mixup_h64", dropout=0.4, weight_decay=1e-3, mixup_alpha=0.4, hidden_dim=64),
    Recipe("C2_mixup_h128", dropout=0.4, weight_decay=1e-3, mixup_alpha=0.4, hidden_dim=128),
    Recipe("C3_reg_h64", dropout=0.5, weight_decay=1e-2, noise_sigma=0.05, hidden_dim=64),
    Recipe("C4_reg_h128", dropout=0.5, weight_decay=1e-2, noise_sigma=0.05, hidden_dim=128),
    Recipe(
        "C5_mixupreg_h64",
        dropout=0.4, weight_decay=1e-2, mixup_alpha=0.4, noise_sigma=0.05, hidden_dim=64,
    ),
    Recipe(
        "C6_mixupreg_h128",
        dropout=0.4, weight_decay=1e-2, mixup_alpha=0.4, noise_sigma=0.05, hidden_dim=128,
    ),
]


# ---------------------------------------------------------------- 가중치


def class_weights(cache, power: float, device: str) -> dict[str, torch.Tensor]:
    weights = {}
    for task_name, task in ATTRIBUTE_TASKS.items():
        if task.multi_label:
            continue
        mask = cache["valid"][task_name]
        target = cache["targets"][task_name][mask].long()
        if not len(target):
            continue
        counts = torch.bincount(target, minlength=len(task.labels)).float()
        present = counts > 0
        task_weights = torch.zeros_like(counts)
        task_weights[present] = (counts[present].sum() / counts[present]).pow(power)
        task_weights[present] /= task_weights[present].mean()
        weights[task_name] = task_weights.clamp_max(10.0).to(device)
    return weights


def positive_weights(cache, power: float, device: str) -> dict[str, torch.Tensor]:
    weights = {}
    for task_name, task in ATTRIBUTE_TASKS.items():
        if not task.multi_label:
            continue
        mask = cache["valid"][task_name]
        target = cache["targets"][task_name][mask].float()
        if not len(target):
            continue
        positives = target.sum(dim=0)
        negatives = len(target) - positives
        ratio = (negatives / positives.clamp_min(1.0)).pow(power * 2)
        weights[task_name] = ratio.clamp(1.0, 20.0).to(device)
    return weights


# ---------------------------------------------------------------- 손실


def _single_loss(logits, target, weight, smoothing):
    return functional.cross_entropy(logits, target.long(), weight=weight, label_smoothing=smoothing)


def _multi_loss(logits, target, pos_weight, focal_gamma):
    target = target.float()
    if focal_gamma <= 0:
        return functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    # focal: 쉬운 음성이 loss를 지배하지 않게 (1-p_t)^gamma 로 눌러 소수 라벨을 살린다.
    bce = functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none"
    )
    probability = logits.sigmoid()
    p_t = target * probability + (1 - target) * (1 - probability)
    return (bce * (1 - p_t).clamp_min(1e-6).pow(focal_gamma)).mean()


def batch_loss(heads, features, targets, valid, recipe, weights, pos_weights, *, mix=None):
    """태스크별로 라벨이 있는 행만 해당 헤드에 통과시킨다.

    한 crop에 라벨이 붙는 태스크는 평균 4.4개뿐이라 17개 헤드를 전 행에 돌리면
    연산의 4분의 3이 미주석 행에 버려진다. 결과는 동일하고 속도만 개선된다.
    """
    losses = []
    mixed = []
    mix_targets, mix_valid, lam = mix if mix is not None else (None, None, 1.0)
    for task_name, task in ATTRIBUTE_TASKS.items():
        mask = valid[task_name]
        mix_mask = mix_valid[task_name] if mix_valid is not None else None
        needed = mask if mix_mask is None else (mask | mix_mask)
        if not bool(needed.any()):
            continue
        rows = needed.nonzero(as_tuple=True)[0]
        logits = heads.heads[task_name](features[rows])
        loss_fn = (
            (lambda lg, tg: _multi_loss(lg, tg, pos_weights.get(task_name), recipe.focal_gamma))
            if task.multi_label
            else (lambda lg, tg: _single_loss(lg, tg, weights.get(task_name), recipe.label_smoothing))
        )
        local = mask[rows]
        if bool(local.any()):
            losses.append(loss_fn(logits[local], targets[task_name][rows][local]))
        if mix_mask is not None:
            local_mix = mix_mask[rows]
            if bool(local_mix.any()):
                mixed.append(loss_fn(logits[local_mix], mix_targets[task_name][rows][local_mix]))
    if not losses and not mixed:
        return None
    primary = torch.stack(losses).mean() if losses else torch.zeros((), device=features.device)
    if mixed:
        return lam * primary + (1 - lam) * torch.stack(mixed).mean()
    return primary


# ---------------------------------------------------------------- 태스크 점수


def task_scores(logits: dict[str, torch.Tensor], cache) -> dict[str, dict[str, float]]:
    """epoch·레시피 선택용 점수.

    대표 지표만 보면 소수 라벨 붕괴를 놓치므로 accuracy(또는 micro-F1)와 macro-F1을 반씩 섞는다.
    """
    scores: dict[str, dict[str, float]] = {}
    for task_name, task in ATTRIBUTE_TASKS.items():
        mask = cache["valid"][task_name]
        if not bool(mask.any()):
            continue
        expected = cache["targets"][task_name][mask]
        selected = logits[task_name][mask].float()
        if task.multi_label:
            probabilities = selected.sigmoid()
            expected_bool = expected.bool()
            best = None
            for threshold in MULTI_THRESHOLD_GRID:
                predicted = probabilities >= threshold
                tp = int((predicted & expected_bool).sum())
                fp = int((predicted & ~expected_bool).sum())
                fn = int((~predicted & expected_bool).sum())
                denominator = 2 * tp + fp + fn
                micro = 2 * tp / denominator if denominator else 0.0
                per_label = []
                for index in range(len(task.labels)):
                    label_expected = expected_bool[:, index]
                    if not int(label_expected.sum()):
                        continue
                    label_predicted = predicted[:, index]
                    ltp = int((label_predicted & label_expected).sum())
                    lfp = int((label_predicted & ~label_expected).sum())
                    lfn = int((~label_predicted & label_expected).sum())
                    ld = 2 * ltp + lfp + lfn
                    per_label.append(2 * ltp / ld if ld else 0.0)
                macro = sum(per_label) / len(per_label) if per_label else 0.0
                blended = 0.5 * micro + 0.5 * macro
                if best is None or blended > best["selection"]:
                    best = {
                        "selection": blended, "primary": micro, "macro_f1": macro,
                        "threshold": threshold,
                    }
            scores[task_name] = best
        else:
            predicted = selected.argmax(dim=-1)
            accuracy = float((predicted == expected).float().mean())
            per_label = []
            for index in range(len(task.labels)):
                label_expected = expected == index
                support = int(label_expected.sum())
                if not support:
                    continue
                label_predicted = predicted == index
                tp = int((label_predicted & label_expected).sum())
                predicted_count = int(label_predicted.sum())
                precision = tp / predicted_count if predicted_count else 0.0
                recall = tp / support
                per_label.append(
                    2 * precision * recall / (precision + recall) if precision + recall else 0.0
                )
            macro = sum(per_label) / len(per_label) if per_label else 0.0
            scores[task_name] = {
                "selection": 0.5 * accuracy + 0.5 * macro,
                "primary": accuracy,
                "macro_f1": macro,
                "threshold": None,
            }
    return scores


# ---------------------------------------------------------------- 학습


def train_once(recipe: Recipe, seed: int, fit, dev, device: str = "cpu"):
    """한 레시피/시드로 학습하고 태스크별 dev 최고 시점의 헤드 가중치를 돌려준다."""
    torch.manual_seed(seed)
    input_dim = int(fit["features"].shape[1])
    heads = build_attribute_heads(input_dim, recipe.hidden_dim, recipe.dropout).to(device)
    optimizer = torch.optim.AdamW(
        heads.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe.epochs)
        if recipe.cosine_schedule else None
    )
    weights = class_weights(fit, recipe.class_weight_power, device)
    pos_weights = positive_weights(fit, recipe.class_weight_power, device)

    features_all = fit["features"].to(device)
    targets_all = {name: value.to(device) for name, value in fit["targets"].items()}
    valid_all = {name: value.to(device) for name, value in fit["valid"].items()}
    size = len(features_all)

    best_state: dict[str, dict] = {}
    best_score = {name: -1.0 for name in ATTRIBUTE_TASKS}
    best_epoch = {name: 0 for name in ATTRIBUTE_TASKS}
    best_threshold: dict[str, float | None] = {name: None for name in ATTRIBUTE_TASKS}

    for epoch in range(1, recipe.epochs + 1):
        heads.train()
        permutation = torch.randperm(size)
        for start in range(0, size, recipe.batch_size):
            indices = permutation[start:start + recipe.batch_size]
            features = features_all[indices]
            targets = {name: value[indices] for name, value in targets_all.items()}
            valid = {name: value[indices] for name, value in valid_all.items()}
            mix = None
            if recipe.noise_sigma > 0:
                features = features + torch.randn_like(features) * recipe.noise_sigma
                features = functional.normalize(features, dim=-1)
            if recipe.mixup_alpha > 0:
                lam = float(
                    torch.distributions.Beta(recipe.mixup_alpha, recipe.mixup_alpha).sample()
                )
                lam = max(lam, 1 - lam)  # 원본 쪽을 항상 주 라벨로 둔다
                shuffle = torch.randperm(len(indices))
                partner = indices[shuffle]
                features = functional.normalize(
                    lam * features + (1 - lam) * features_all[partner], dim=-1
                )
                mix = (
                    {name: value[partner] for name, value in targets_all.items()},
                    {name: value[partner] for name, value in valid_all.items()},
                    lam,
                )
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(heads, features, targets, valid, recipe, weights, pos_weights, mix=mix)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        scores = task_scores(head_logits(heads, dev, device), dev)
        for task_name, value in scores.items():
            if value["selection"] > best_score[task_name] + 1e-6:
                best_score[task_name] = value["selection"]
                best_epoch[task_name] = epoch
                best_threshold[task_name] = value["threshold"]
                best_state[task_name] = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in heads.heads[task_name].state_dict().items()
                }
    return {
        "recipe": recipe.name,
        "seed": seed,
        "state": best_state,
        "dev_selection": best_score,
        "dev_epoch": best_epoch,
        "dev_threshold": best_threshold,
    }


RUN_DIR = REPORT_DIR / "runs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipes", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--threads", type=int, default=2)
    arguments = parser.parse_args()

    torch.set_num_threads(arguments.threads)
    device = "cpu"
    train_cache = load_cache(TRAIN_CACHE)
    val_cache = load_cache(VAL_CACHE)
    fit_indices, dev_indices = split_train_dev(train_cache)
    fit = subset(train_cache, fit_indices)
    dev = subset(train_cache, dev_indices)
    print(f"train_fit={len(fit['features'])}  train_dev={len(dev['features'])}  val={len(val_cache['features'])}")

    selected = [r for r in RECIPES if arguments.recipes is None or r.name in arguments.recipes]
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for recipe in selected:
        recipe = copy.copy(recipe)
        if arguments.epochs:
            recipe.epochs = arguments.epochs
        runs = []
        for seed in range(arguments.seeds):
            started = time.time()
            run = train_once(recipe, seed, fit, dev, device)
            elapsed = time.time() - started
            mean_selection = sum(run["dev_selection"].values()) / len(run["dev_selection"])
            print(
                f"  {recipe.name:<14} seed={seed}  dev_selection_mean={mean_selection:.4f}  {elapsed:.0f}s",
                flush=True,
            )
            runs.append(run)
        torch.save({"config": asdict(recipe), "runs": runs}, RUN_DIR / f"{recipe.name}.pt")
        print(f"  saved: {RUN_DIR / f'{recipe.name}.pt'}", flush=True)


if __name__ == "__main__":
    main()
