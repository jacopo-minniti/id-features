from id_features.config import ExperimentConfig
from id_features.experiments import _interpret


def test_hypothesis_1_report_requires_and_recognizes_all_controls() -> None:
    config = ExperimentConfig(
        experiment="load-capacity",
        active_features=(1, 2),
        capacity_values=(64, 128),
        repeats=1,
    )
    rows = [
        {
            "experiment": "load-capacity",
            "condition": f"m={m};k={k}",
            "repeat": 0,
            "representation_dim": 32,
            "feature_count": m,
            "k": k,
            "rho": 0.0,
            "local_gride_id_rank_2": float(k),
            "mean_feature_auroc": 0.9,
            "mean_normalized_signed_margin": 0.3,
        }
        for m in (64, 128)
        for k in (1, 2)
    ]
    report = _interpret(config, rows, [])
    assert "## Hypothesis 1" in report
    assert "**Conclusion: PASS.**" in report


def test_hypothesis_3_requires_profile_match_and_replicated_effect() -> None:
    config = ExperimentConfig(
        experiment="geometry-control", correlations=(0.0, 0.5), repeats=2
    )
    summary_rows = [
        {
            "experiment": "geometry-control",
            "condition": f"rho={rho};repeat={repeat}",
            "repeat": repeat,
            "representation_dim": 32,
            "feature_count": 128,
            "k": 8,
            "rho": rho,
            "mean_feature_dot_product": rho,
            "mean_feature_auroc": 0.90 if rho == 0.0 else 0.80,
        }
        for rho in (0.0, 0.5)
        for repeat in (0, 1)
    ]
    profile_rows = [
        {
            "rho": rho,
            "rank": rank,
            "gride_id": 8.0 if rho == 0.0 else 8.05,
        }
        for rho in (0.0, 0.5)
        for rank in (2, 4)
        for _repeat in (0, 1)
    ]
    report = _interpret(config, summary_rows, profile_rows)
    assert "matched counterexample" in report
    assert "**Conclusion: PASS.**" in report
