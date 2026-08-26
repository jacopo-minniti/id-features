import pytest

from id_features.config import ExperimentConfig


def test_config_rejects_gride_range_larger_than_dataset() -> None:
    with pytest.raises(ValueError, match="smaller than n_id_samples"):
        ExperimentConfig(experiment="load-capacity", n_id_samples=20, gride_range_max=20)


def test_config_rejects_non_superposed_dictionary() -> None:
    with pytest.raises(ValueError, match="exceed"):
        ExperimentConfig(experiment="load-capacity", representation_dim=16, feature_count=16)


def test_geometry_control_requires_isotropic_baseline() -> None:
    with pytest.raises(ValueError, match="requires rho=0"):
        ExperimentConfig(experiment="geometry-control", correlations=(0.25, 0.5))


def test_config_rejects_nonpositive_gride_worker_count() -> None:
    with pytest.raises(ValueError, match="gride_n_jobs"):
        ExperimentConfig(experiment="load-ratio", gride_n_jobs=0)
