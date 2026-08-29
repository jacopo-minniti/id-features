import pytest

from id_features.config import BidAuditConfig, ExperimentConfig, ManifoldGeometryConfig


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


def test_support_pool_requires_balanced_sample_counts() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ExperimentConfig(
            experiment="support-pool",
            support_pool_sizes=(1, 3),
            id_sample_values=(128,),
        )


def test_manifold_config_requires_an_orthogonal_feature_subspace() -> None:
    with pytest.raises(ValueError, match="feature subspace"):
        ManifoldGeometryConfig(representation_dim=9, intrinsic_dims=(8,))


def test_manifold_activity_controls_are_relative_to_each_intrinsic_dimension() -> None:
    config = ManifoldGeometryConfig(
        intrinsic_dims=(4,), activity_multipliers=(0.5, 1.0, 2.0)
    )
    assert config.expected_activities(4) == (2, 4, 8)


def test_bid_audit_requires_balanced_support_pools_and_primary_quantile() -> None:
    with pytest.raises(ValueError, match="divisible"):
        BidAuditConfig(sample_count=2500, support_pool_sizes=(1, 64))
    with pytest.raises(ValueError, match="primary_alpha_max"):
        BidAuditConfig(alpha_max_values=(0.25, 0.5), primary_alpha_max=1.0)
