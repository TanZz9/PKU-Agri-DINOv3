from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C_grid: Optional[List[float]] = None,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    if len(X_train) == 0 or len(X_test) == 0:
        return {
            "top1_accuracy": 0.0,
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "num_train": len(X_train),
            "num_test": len(X_test),
            "num_classes": len(np.unique(np.concatenate([y_train, y_test])) if len(y_train) + len(y_test) else 0),
            "best_C": None,
            "error": "empty_split",
        }

    C_grid = C_grid or [0.01, 0.1, 1.0, 10.0]
    best_C = 1.0
    best_score = -1.0

    unique, counts = np.unique(y_train, return_counts=True)
    min_count = counts.min() if len(counts) else 0
    use_cv = min_count >= cv_folds and len(unique) >= 2

    if use_cv:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        for C in C_grid:
            scores = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                clf = LogisticRegression(
                    C=C,
                    max_iter=1000,
                    solver="lbfgs",
                )
                clf.fit(X_train[train_idx], y_train[train_idx])
                pred = clf.predict(X_train[val_idx])
                scores.append(accuracy_score(y_train[val_idx], pred))
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_C = C
    else:
        best_C = 1.0

    clf = LogisticRegression(
        C=best_C,
        max_iter=1000,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        "top1_accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
        "num_train": int(len(X_train)),
        "num_test": int(len(X_test)),
        "num_classes": int(len(np.unique(np.concatenate([y_train, y_test])))),
        "best_C": best_C,
    }
