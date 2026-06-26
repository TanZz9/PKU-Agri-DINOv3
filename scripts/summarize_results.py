#!/usr/bin/env python3
"""Pivot metrics.jsonl into summary CSV."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from paths import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-path",
        default=str(RESULTS_DIR / "metrics.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "summary.csv"),
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics_path)
    if not metrics_path.exists():
        print(f"No metrics file: {metrics_path}")
        return

    rows = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    if "error" in df.columns:
        ok = df[df["error"].isna()]
    else:
        ok = df

    if "top1_accuracy" not in ok.columns:
        print("No successful metrics rows.")
        return

    out_path = Path(args.output)
    pivot_acc = ok.pivot_table(
        index="dataset",
        columns="model",
        values="top1_accuracy",
        aggfunc="max",
    )
    pivot_acc.to_csv(out_path)
    print(f"Saved accuracy pivot -> {out_path}")

    f1_path = out_path.with_name("summary_macro_f1.csv")
    pivot_f1 = ok.pivot_table(
        index="dataset",
        columns="model",
        values="macro_f1",
        aggfunc="max",
    )
    pivot_f1.to_csv(f1_path)
    print(f"Saved macro_f1 pivot -> {f1_path}")
    print(pivot_acc.round(4).to_string())


if __name__ == "__main__":
    main()
