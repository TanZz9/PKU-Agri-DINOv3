#!/usr/bin/env python3
"""Run linear-probe evaluation for DINOv3 checkpoint(s) on Agri dataset(s)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backbones.factory import build_backbone
from extract_features import extract_and_cache
from paths import ALL_DATASETS, ALL_MODELS, RESULTS_DIR
from train_linear_probe import train_and_evaluate


def parse_list(arg: str, all_values: list) -> list:
    if arg == "all":
        return list(all_values)
    return [x.strip() for x in arg.split(",") if x.strip()]


def append_metrics(record: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="dinov3_vitb16", help="comma-separated or 'all'")
    parser.add_argument("--datasets", default="all", help="comma-separated or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="DINOv3 checkpoint (.pth) for backbone weights",
    )
    parser.add_argument(
        "--model-tag",
        default=None,
        help="Suffix for metrics/cache id, e.g. epoch3 -> dinov3_vitb16_epoch3",
    )
    args = parser.parse_args()

    models = parse_list(args.models, ALL_MODELS)
    datasets = parse_list(args.datasets, ALL_DATASETS)
    metrics_path = Path(args.metrics_path or RESULTS_DIR / "metrics.jsonl")

    for model_id in models:
        run_model_id = model_id
        if args.model_tag:
            run_model_id = f"{model_id}_{args.model_tag}"
        elif args.checkpoint:
            ckpt_name = Path(args.checkpoint).stem
            run_model_id = f"{model_id}_{ckpt_name}"

        print(f"\n=== Loading backbone: {run_model_id} ===")
        if args.checkpoint:
            print(f"    checkpoint: {args.checkpoint}")
        backbone = build_backbone(
            model_id,
            device=args.device,
            checkpoint_override=args.checkpoint,
        )

        for dataset in datasets:
            print(f"--- {run_model_id} x {dataset} ---")
            try:
                feats = extract_and_cache(
                    run_model_id,
                    dataset,
                    backbone,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    use_cache=args.use_cache,
                    refresh_cache=args.refresh_cache,
                )
                metrics = train_and_evaluate(
                    feats["X_train"],
                    feats["y_train"],
                    feats["X_test"],
                    feats["y_test"],
                )
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": run_model_id,
                    "base_model": model_id,
                    "dataset": dataset,
                    "seed": args.seed,
                    "checkpoint": args.checkpoint,
                    "from_cache": feats.get("from_cache", False),
                    **metrics,
                }
                append_metrics(record, metrics_path)
                print(
                    f"  acc={metrics['top1_accuracy']:.4f} "
                    f"macro_f1={metrics['macro_f1']:.4f} "
                    f"train={metrics['num_train']} test={metrics['num_test']}"
                )
            except Exception as e:
                err_record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": run_model_id,
                    "base_model": model_id,
                    "dataset": dataset,
                    "seed": args.seed,
                    "checkpoint": args.checkpoint,
                    "error": str(e),
                }
                append_metrics(err_record, metrics_path)
                print(f"  ERROR: {e}")
                import traceback

                traceback.print_exc()

        del backbone
        if args.device.startswith("cuda") and __import__("torch").cuda.is_available():
            __import__("torch").cuda.empty_cache()

    print(f"\nDone. Metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
