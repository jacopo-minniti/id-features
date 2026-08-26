"""GRIDE and held-out linear feature-recovery measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class GrideProfile:
    ranks: NDArray[np.int_]
    scales: FloatArray
    ids: FloatArray
    errors: FloatArray


@dataclass(frozen=True)
class LinearAccessibility:
    mean_auroc: float
    std_auroc: float
    mean_balanced_accuracy: float
    mean_normalized_signed_margin: float
    std_normalized_signed_margin: float
    feature_aurocs: FloatArray


def measure_gride(representations: FloatArray, range_max: int, n_jobs: int = 1) -> GrideProfile:
    """Estimate multi-scale ID with DADApy's generalized-ratios estimator."""

    # Keep the expensive optional backend out of linear-probe-only imports and tests.
    from dadapy.data import Data

    if range_max >= len(representations):
        raise ValueError("range_max must be smaller than the number of representations")
    data = Data(
        coordinates=np.asarray(representations, dtype=np.float64), maxk=range_max, n_jobs=n_jobs
    )
    data.compute_distances(maxk=range_max, n_jobs=n_jobs)
    ids, errors, physical_scales = data.return_id_scaling_gride(range_max=range_max)
    # DADApy returns comparisons (1,2), (2,4), ... so their upper ranks are 2,4,...
    ranks = 2 ** np.arange(1, len(ids) + 1)
    return GrideProfile(
        ranks=ranks,
        scales=np.asarray(physical_scales, dtype=np.float64),
        ids=np.asarray(ids, dtype=np.float64),
        errors=np.asarray(errors, dtype=np.float64),
    )


def measure_linear_accessibility(
    train_h: FloatArray,
    train_labels: BoolArray,
    test_h: FloatArray,
    test_labels: BoolArray,
) -> LinearAccessibility:
    """Recover each known latent feature with an independent held-out probe."""

    if train_labels.shape[1] != test_labels.shape[1]:
        raise ValueError("train and test must have the same feature vocabulary")

    aurocs: list[float] = []
    balanced_accuracies: list[float] = []
    normalized_signed_margins: list[float] = []
    for feature_index in range(train_labels.shape[1]):
        classifier = LogisticRegression(
            solver="liblinear", max_iter=1_000, class_weight="balanced", random_state=0
        )
        classifier.fit(train_h, train_labels[:, feature_index])
        scores = classifier.decision_function(test_h)
        aurocs.append(roc_auc_score(test_labels[:, feature_index], scores))
        balanced_accuracies.append(
            balanced_accuracy_score(test_labels[:, feature_index], classifier.predict(test_h))
        )
        # Euclidean signed distance from the fitted separating hyperplane.  Unlike
        # raw logits this is invariant to rescaling the logistic-regression weights.
        coefficient_norm = np.linalg.norm(classifier.coef_)
        signed_scores = np.where(test_labels[:, feature_index], scores, -scores)
        normalized_signed_margins.append(float(np.mean(signed_scores / coefficient_norm)))

    feature_aurocs = np.asarray(aurocs, dtype=np.float64)
    return LinearAccessibility(
        mean_auroc=float(feature_aurocs.mean()),
        std_auroc=float(feature_aurocs.std(ddof=0)),
        mean_balanced_accuracy=float(np.mean(balanced_accuracies)),
        mean_normalized_signed_margin=float(np.mean(normalized_signed_margins)),
        std_normalized_signed_margin=float(np.std(normalized_signed_margins, ddof=0)),
        feature_aurocs=feature_aurocs,
    )
