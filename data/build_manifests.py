#!/usr/bin/env python3
"""Build JSONL manifests for all Agri-dataset subfolders."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset_registry import LOADERS, load_samples
from paths import ALL_DATASETS, MANIFEST_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    args = parser.parse_args()

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        if name not in LOADERS:
            raise SystemExit(f"Unknown dataset: {name}")
        samples = load_samples(name)
        out_path = MANIFEST_DIR / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        n_classes = len({s["label_id"] for s in samples})
        print(f"[{name}] {len(samples)} samples, {n_classes} classes -> {out_path}")


if __name__ == "__main__":
    main()
