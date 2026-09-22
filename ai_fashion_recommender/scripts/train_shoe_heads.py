"""Train the existing twelve-output shoe head on frozen embedding caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from shoe_model import SHOE_LABELS, build_shoe_heads, save_shoe_checkpoint, shoe_classification_loss


def load_cache(path):
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(cache["label_names"]) != SHOE_LABELS or cache["preprocessing"] != "squash":
        raise ValueError(f"Incompatible labels/preprocessing: {path}")
    x, y = cache["features"], cache["labels"]
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
        raise ValueError(f"Invalid cache shapes: {path}")
    if not torch.isfinite(x).all() or y.dtype != torch.long or (y < -1).any() or (y >= len(SHOE_LABELS)).any():
        raise ValueError(f"Invalid features/labels: {path}")
    for field in ("paths", "subject_ids", "product_ids", "session_ids", "checksums", "label_sources"):
        if len(cache[field]) != len(y):
            raise ValueError(f"Missing row metadata {field}: {path}")
    return cache


def validate_pair(a, b):
    if a["backbone_model_id"] != b["backbone_model_id"] or a["features"].shape[1] != b["features"].shape[1]:
        raise ValueError("Backbone/dimension mismatch")
    for field in ("paths", "subject_ids", "product_ids", "session_ids", "checksums"):
        if (set(a[field]) & set(b[field])) - {""}:
            raise ValueError(f"Cross-split leakage: {field}")


def metrics(y, probabilities, threshold):
    pred = probabilities.argmax(1)
    confidence = probabilities.max(1)
    accepted = (confidence >= threshold) & (pred < 10)
    valid = y >= 0
    y, pred, accepted = y[valid], pred[valid], accepted[valid]
    if not len(y):
        raise ValueError("Evaluation split has no labelled examples")
    return {
        "accuracy": float((y == pred).mean()),
        "macro_f1": float(f1_score(y, pred, labels=list(range(12)), average="macro", zero_division=0)),
        "per_class": classification_report(y, pred, labels=list(range(12)), target_names=SHOE_LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(12))).tolist(),
        "accepted_count": int(accepted.sum()),
        "accepted_coverage": float(accepted.mean()),
        "accepted_precision": float((y[accepted] == pred[accepted]).mean()) if accepted.any() else None,
        "barefoot_false_accept_rate": float(accepted[y == 11].mean()) if (y == 11).any() else None,
        "other_false_accept_rate": float(accepted[y == 10].mean()) if (y == 10).any() else None,
    }


def select_threshold(y, probabilities, target, minimum):
    # Reject-all is explicitly recorded when validation cannot support acceptance.
    curve = []
    for threshold in np.linspace(0.05, 1, 96):
        m = metrics(y, probabilities, float(threshold))
        curve.append({"threshold": float(threshold), **{k: m[k] for k in ("accepted_count", "accepted_coverage", "accepted_precision")}})
    candidates = [r for r in curve if r["accepted_count"] >= minimum and r["accepted_precision"] >= target]
    best = max(candidates, key=lambda r: (r["accepted_coverage"], r["threshold"])) if candidates else None
    return (best["threshold"] if best else 1.0), curve, bool(best)


def predict(head, x, device, batch):
    head.eval()
    with torch.inference_mode():
        return torch.cat([head(part.to(device)).softmax(-1).cpu() for part in x.split(batch)]).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=.15)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 2026])
    p.add_argument("--target-precision", type=float, default=.85)
    p.add_argument("--minimum-accepted", type=int, default=30)
    p.add_argument("--warmup-only", action="store_true", help="Allow missing classes/catalog data; export research weights, never a service checkpoint")
    p.add_argument("--init-warmup", type=Path, help="Initialize from this pipeline's partial warm-up weights")
    args = p.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA unavailable; refusing silent CPU fallback")
    if min(args.epochs, args.batch_size, args.patience, args.minimum_accepted) <= 0 or not 0 < args.target_precision <= 1:
        p.error("Invalid training settings")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    train, val = [load_cache(args.cache_dir / f"{s}.pt") for s in ("train", "val")]
    validate_pair(train, val)
    if not args.warmup_only:
        for c in (train, val):
            if set(c["labels"].tolist()) - {-1} != set(range(12)):
                p.error("All 12 classes required for final training; use --warmup-only for partial data")
        if not val.get("domains") or set(val["domains"]) != {"outfit"}:
            p.error("Final validation requires actual outfit crops")
        allowed_train_sources = {"human_reviewed", "source_metadata_ground_truth"}
        if (any(source not in allowed_train_sources for source in train["label_sources"])
                or any(source != "human_reviewed" for source in val["label_sources"])):
            p.error("Final service training requires reviewed/ground-truth train labels and human-reviewed validation labels")
    valid = train["labels"] >= 0
    x, y = train["features"][valid].float(), train["labels"][valid]
    if not len(y):
        p.error("No labelled training data")
    support = torch.bincount(y, minlength=12)
    runs, best_state, best_prob, best_score = [], None, None, -1
    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        head = build_shoe_heads(x.shape[1], args.hidden_dim, args.dropout).to(args.device)
        if args.init_warmup:
            init = torch.load(args.init_warmup, map_location="cpu", weights_only=True)
            if (init.get("purpose") != "partial_catalog_warmup_not_for_inference"
                    or tuple(init.get("labels", ())) != SHOE_LABELS
                    or init.get("backbone_model_id") != train["backbone_model_id"]
                    or init.get("input_dim") != x.shape[1]):
                raise ValueError("Incompatible warm-up checkpoint")
            head.load_state_dict(init["state_dict"], strict=True)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)
        scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
        sampler = torch.utils.data.WeightedRandomSampler(1.0 / support[y].float(), len(y), replacement=True)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=args.batch_size, sampler=sampler)
        history, score, stale, state, best_epoch = [], -1, 0, None, 0
        for epoch in range(1, args.epochs + 1):
            head.train(); total = 0.0
            for bx, by in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu", enabled=args.device.startswith("cuda")):
                    loss = shoe_classification_loss(head(bx.to(args.device)), by.to(args.device))
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                total += float(loss.detach()) * len(by)
            prob = predict(head, val["features"].float(), args.device, args.batch_size)
            f1 = metrics(val["labels"].numpy(), prob, 1.0)["macro_f1"]
            scheduler.step(f1)
            history.append({"epoch": epoch, "loss": total / len(y), "validation_macro_f1": f1})
            print(json.dumps({"seed": seed, **history[-1]}), flush=True)
            if f1 > score:
                score, stale, best_epoch = f1, 0, epoch
                state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            else:
                stale += 1
            if stale >= args.patience:
                break
        head.load_state_dict(state)
        prob = predict(head, val["features"].float(), args.device, args.batch_size)
        runs.append({"seed": seed, "best_epoch": best_epoch, "validation_macro_f1": score, "history": history})
        if score > best_score:
            best_score, best_state, best_prob, best_seed = score, state, prob, seed
    threshold, curve, calibrated = select_threshold(val["labels"].numpy(), best_prob, args.target_precision, args.minimum_accepted)
    head.load_state_dict(best_state)
    report = {"warmup_only": args.warmup_only, "service_ready": False,
              "runs": runs, "best_seed": best_seed, "threshold": threshold, "calibration_target_met": calibrated,
              "threshold_curve": curve, "train_support": support.tolist(), "training_examples": len(y),
              "validation": metrics(val["labels"].numpy(), best_prob, threshold),
              "seed_macro_f1_mean": float(np.mean([r["validation_macro_f1"] for r in runs])),
              "seed_macro_f1_std": float(np.std([r["validation_macro_f1"] for r in runs])),
              "backbone_model_id": train["backbone_model_id"],
              "settings": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}}
    report["cache_sha256"] = {s: hashlib.sha256((args.cache_dir / f"{s}.pt").read_bytes()).hexdigest() for s in ("train", "val")}
    report["manifest_sha256"] = train.get("manifest_sha256")
    repo = Path(__file__).resolve().parents[2]
    report["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    report["code_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.warmup_only:
        torch.save({"purpose": "partial_catalog_warmup_not_for_inference", "state_dict": best_state,
                    "labels": SHOE_LABELS, "input_dim": x.shape[1], "backbone_model_id": train["backbone_model_id"]}, args.output)
        test_path = args.cache_dir / "test.pt"
        if test_path.is_file():
            test = load_cache(test_path)
            validate_pair(train, test); validate_pair(val, test)
            report["research_test"] = metrics(
                test["labels"].numpy(),
                predict(head, test["features"].float(), args.device, args.batch_size),
                threshold,
            )
            report["research_test_warning"] = (
                "This held-out split contains pseudo labels and is not a final human-reviewed test set."
            )
    else:
        if not calibrated:
            raise ValueError("Validation acceptance target not met; no service checkpoint exported")
        test = load_cache(args.cache_dir / "test.pt")
        validate_pair(train, test); validate_pair(val, test)
        if (set(test["labels"].tolist()) - {-1} != set(range(12))
                or set(test.get("domains", [])) != {"outfit"}
                or any(source != "human_reviewed" for source in test["label_sources"])):
            raise ValueError("Final test requires all classes and outfit crops")
        report["test"] = metrics(test["labels"].numpy(), predict(head, test["features"].float(), args.device, args.batch_size), threshold)
        save_shoe_checkpoint(args.output, head, backbone_model_id=train["backbone_model_id"], training_examples=len(y), threshold=threshold)
    report["checkpoint_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    (args.report_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
