from __future__ import annotations

import numpy as np

from PierNet.training.router.metrics import binary_classification_metrics, sigmoid


def test_sigmoid_handles_extreme_logits_without_overflow_warning():
    logits = np.array([-1000.0, 0.0, 1000.0])
    with np.errstate(over="raise", under="ignore"):
        probs = sigmoid(logits)

    assert probs[0] == 0.0
    assert probs[1] == 0.5
    assert probs[2] == 1.0


def test_binary_metrics_with_extreme_logits_are_finite():
    labels = np.array([0, 1, 1])
    logits = np.array([-1000.0, 1000.0, 0.0])

    metrics = binary_classification_metrics(labels, logits)

    assert metrics["accuracy"] == 1.0
    assert np.isfinite(metrics["pr_auc"])
