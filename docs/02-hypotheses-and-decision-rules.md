# Hypotheses and claim boundaries

## Hypothesis 1: concurrent feature load, not capacity

The initial, local-scale hypothesis is `d_ID ≈ k` in a random sparse dictionary. The `load-capacity` sweep crosses `k` with capacity `m` while holding `D` fixed. Smaller dictionaries are prefixes of the same larger dictionary within a replicate. Its report supports the hypothesis only when all three pre-specified checks pass: local-ID/k monotonicity (`rho >= .90`), local ID within 25% of `k`, and at most 10% local-ID variation across capacities at a fixed `k`.

1. Inspect `gride_profiles.csv` or `overview.png`; rank-2 estimates are only a local summary.
2. Ask whether the local profile rises with `k` and is roughly compatible with the fixed-support dimension.
3. Ask separately whether larger ranks rise above the local value, which could reflect crossings between different support configurations.

This run does not establish that intrinsic dimension universally measures active features. It establishes only whether that interpretation is descriptively useful for this exact generator and scale range.

## Hypothesis 2: load-ratio accessibility in an isotropic regime

For each ground-truth feature, a separate logistic-regression probe is trained on an independent generated training set and evaluated on a fresh test set. Mean AUROC across the feature vocabulary is the primary score; balanced accuracy is retained as a thresholded companion metric.

The `load-ratio` sweep crosses `k` and `D` at fixed `m`. It reports held-out AUROC and the mean normalized signed distance to each probe's fitted hyperplane (a scale-invariant geometric margin). Its toy-regime conclusion requires both scores to have Spearman correlation at most `-.80` with measured `d_ID / D`, and equivalent nominal `k / D` conditions to differ in AUROC by no more than .02.

The geometry control is the minimal falsification attempt. Hold `k` fixed, increase shared direction strength, and compare the entire GRIDE profiles with AUROC. It supports an ID-insufficiency counterexample only when a nonzero-`rho` condition is within 10% of the `rho=0` ID estimate at every saved rank *and* its paired AUROC decrease has a 95% lower bound of at least .02. If this does not happen, the result is inconclusive rather than evidence that ID is sufficient.

## Deliberately excluded from this first stage

- No learned encoder or reconstruction objective.
- No support-diversity, irrelevant-latent, curvature, or noise controls.
- No claim about language models or semantic features.
- No selection of a single privileged GRIDE scale after looking at outcomes.

Those extensions belong after these two simple interventions have been inspected.
