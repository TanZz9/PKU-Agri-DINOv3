#!/usr/bin/env python3
"""Stratified 30% train / 70% test split per class, saved as JSON."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset_registry import load_samples
from paths import ALL_DATASETS, SPLIT_DIR


def split_class(samples: list, test_ratio: float, rng: random.Random) -> tuple[list, list]:
    n = len(samples)
    if n == 0:
        return [], []
    if n == 1:
        return [], samples
    if n == 2:
        shuffled = samples.copy()
        rng.shuffle(shuffled)
        return [shuffled[0]], [shuffled[1]]

    n_test = max(1, int(round(n * test_ratio)))
    n_test = min(n_test, n - 1)
    n_train = n - n_test
    shuffled = samples.copy()
    rng.shuffle(shuffled)
    return shuffled[:n_train], shuffled[n_train:]


def make_split(dataset_name: str, seed: int, test_ratio: float = 0.7) -> dict:
    samples = load_samples(dataset_name)
    by_label: dict[int, list] = defaultdict(list)
    for s in samples:
        by_label[s["label_id"]].append(s)

    rng = random.Random(seed)
    train, test = [], []
    for label_id in sorted(by_label.keys()):
        tr, te = split_class(by_label[label_id], test_ratio, rng)
        for s in tr:
            train.append({"path": s["path"], "label": s["label_id"], "class_name": s["class_name"]})
        for s in te:
            test.append({"path": s["path"], "label": s["label_id"], "class_name": s["class_name"]})

    return {
        "dataset": dataset_name,
        "seed": seed,
        "test_ratio": test_ratio,
        "num_train": len(train),
        "num_test": len(test),
        "num_classes": len(by_label),
        "train": train,
        "test": test,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.7)
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    args = parser.parse_args()

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        split = make_split(name, args.seed, args.test_ratio)
        out_path = SPLIT_DIR / f"{name}_seed{args.seed}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        total = split["num_train"] + split["num_test"]
        train_pct = 100.0 * split["num_train"] / total if total else 0
        print(
            f"[{name}] train={split['num_train']} test={split['num_test']} "
            f"classes={split['num_classes']} train%={train_pct:.1f} -> {out_path}"
        )


if __name__ == "__main__":
    main()
