from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def precision_recall_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if labels.sum() == 0:
        return 0.0
    order = np.argsort(-probabilities)
    sorted_labels = labels[order].astype(np.float64)
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1.0 - sorted_labels)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / max(tp[-1], 1.0)
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(precision, recall))


def binary_classification_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    labels = labels.astype(np.int64)
    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(labels), 1)
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": precision_recall_auc(labels, probs),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_rate": float(labels.mean()) if len(labels) else 0.0,
    }
