from id_features.config import ExperimentConfig
from id_features.experiments import conditions_for


def test_capacity_design_crosses_k_and_m() -> None:
    config = ExperimentConfig(
        experiment="load-capacity",
        active_features=(1, 2),
        capacity_values=(64, 128),
    )
    conditions = conditions_for(config)
    assert {(condition.feature_count, condition.k) for condition in conditions} == {
        (64, 1),
        (64, 2),
        (128, 1),
        (128, 2),
    }


def test_load_ratio_design_crosses_k_and_d_at_fixed_capacity() -> None:
    config = ExperimentConfig(
        experiment="load-ratio",
        active_features=(1, 2),
        representation_dims=(16, 32),
        feature_count=64,
    )
    conditions = conditions_for(config)
    assert {(condition.representation_dim, condition.k) for condition in conditions} == {
        (16, 1),
        (16, 2),
        (32, 1),
        (32, 2),
    }


def test_support_pool_design_crosses_k_b_and_n() -> None:
    config = ExperimentConfig(
        experiment="support-pool",
        active_features=(1, 2),
        support_pool_sizes=(1, 4),
        id_sample_values=(128, 256),
        gride_range_max=64,
    )
    conditions = conditions_for(config)
    assert {
        (condition.k, condition.support_pool_size, condition.id_sample_count)
        for condition in conditions
    } == {
        (k, b, n)
        for k in (1, 2)
        for b in (1, 4)
        for n in (128, 256)
    }
