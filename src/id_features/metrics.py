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
    mean_same_support_fraction: FloatArray | None = None
    all_same_support_fraction: FloatArray | None = None
    neighbor_indices: NDArray[np.int_] | None = None


@dataclass(frozen=True)
class LinearAccessibility:
    mean_auroc: float
    std_auroc: float
    mean_balanced_accuracy: float
    mean_normalized_signed_margin: float
    std_normalized_signed_margin: float
    feature_aurocs: FloatArray


def _neighbor_support_purity(
    neighbor_indices: NDArray[np.int_], support_ids: NDArray[np.int_], ranks: NDArray[np.int_]
) -> tuple[FloatArray, FloatArray]:
    """Measure mean neighbor purity and fully pure neighborhoods at every rank."""

    if neighbor_indices.shape[0] != len(support_ids):
        raise ValueError("neighbor indices and support IDs must have the same sample count")
    mean_same: list[float] = []
    all_same: list[float] = []
    for rank in ranks:
        neighbors = neighbor_indices[:, 1 : int(rank) + 1]
        same = support_ids[neighbors] == support_ids[:, None]
        mean_same.append(float(same.mean()))
        all_same.append(float(same.all(axis=1).mean()))
    return np.asarray(mean_same), np.asarray(all_same)


def measure_gride(
    representations: FloatArray,
    range_max: int,
    n_jobs: int = 1,
    support_ids: NDArray[np.int_] | None = None,
) -> GrideProfile:
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
    mean_same = None
    all_same = None
    if support_ids is not None:
        support_ids = np.asarray(support_ids, dtype=np.int_)
        if support_ids.shape != (len(representations),):
            raise ValueError("support_ids must have one entry per representation")
        mean_same, all_same = _neighbor_support_purity(data.dist_indices, support_ids, ranks)
    return GrideProfile(
        ranks=ranks,
        scales=np.asarray(physical_scales, dtype=np.float64),
        ids=np.asarray(ids, dtype=np.float64),
        errors=np.asarray(errors, dtype=np.float64),
        mean_same_support_fraction=mean_same,
        all_same_support_fraction=all_same,
        neighbor_indices=np.asarray(data.dist_indices, dtype=np.int_),
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
