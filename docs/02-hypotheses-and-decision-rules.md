# Hypotheses and claim boundaries

## Hypothesis 1: concurrent feature load, not capacity

The initial, local-scale hypothesis is `d_ID ≈ k` in a random sparse dictionary. The `load-capacity` sweep crosses `k` with capacity `m` while holding `D` fixed. Smaller dictionaries are prefixes of the same larger dictionary within a replicate. Its report supports the hypothesis only when all three pre-specified checks pass: local-ID/k monotonicity (`rho >= .90`), local ID within 25% of `k`, and at most 10% local-ID variation across capacities at a fixed `k`.

1. Inspect `gride_profiles.csv` or `overview.png`; rank-2 estimates are only a local summary.
2. Ask whether the local profile rises with `k` and is roughly compatible with the fixed-support dimension.
3. Ask separately whether larger ranks rise above the local value, which could reflect crossings between different support configurations.

This run does not establish that intrinsic dimension universally measures active features. It establishes only whether that interpretation is descriptively useful for this exact generator and scale range.

### Support-pool follow-up

`support-pool` separates concurrent load from support diversity and finite sampling. At fixed `D` and `m`, it crosses exact-k support-pool size `B` and total samples `N`, retaining all GRIDE ranks `r`. A fixed-support PASS requires every `B=1` mean rank-2 estimate to lie within 25% of `k`. A pooled support-pure PASS additionally requires at least one `B>1` cell with at least 90% fully same-support rank-2 neighborhoods, and requires every support-pure rank-2 cell to lie within 25% of `k`.

This decision rule supports only $d_{ID}(h \mid S) \approx k$ in an empirically support-pure finite-scale regime. It does not rescue the stronger claim that marginal ID over arbitrary mixtures equals the active feature count.

## Hypothesis 2: load-ratio accessibility in an isotropic regime

For each ground-truth feature, a separate logistic-regression probe is trained on an independent generated training set and evaluated on a fresh test set. Mean AUROC across the feature vocabulary is the primary score; balanced accuracy is retained as a thresholded companion metric.

The `load-ratio` sweep crosses `k` and `D` at fixed `m`. It reports held-out AUROC and the mean normalized signed distance to each probe's fitted hyperplane (a scale-invariant geometric margin). Its toy-regime conclusion requires both scores to have Spearman correlation at most `-.80` with measured `d_ID / D`, and equivalent nominal `k / D` conditions to differ in AUROC by no more than .02.

The geometry control is the minimal falsification attempt. Hold `k` fixed, increase shared direction strength, and compare the entire GRIDE profiles with AUROC. It supports an ID-insufficiency counterexample only when a nonzero-`rho` condition is within 10% of the `rho=0` ID estimate at every saved rank *and* its paired AUROC decrease has a 95% lower bound of at least .02. If this does not happen, the result is inconclusive rather than evidence that ID is sufficient.

## GRIDE geometry audit

`manifold-geometry` asks when GRIDE recovers a known population dimension before `k` is reintroduced as an unknown scientific target. Its controls have distinct roles:

- `d` and `N` control how small nearest-neighbor neighborhoods become on a high-dimensional manifold.
- `gamma` controls the strength of a sparse nonlinear deformation while the true ID remains fixed.
- expected active count controls how often ReLU support boundaries occur and whether the sparse-only Jacobian has full rank.
- the saved GRIDE rank controls estimator scale without selecting a favorable rank after observing results.

The pre-specified rank-2 calibration tolerance is 10%. The uniform sphere baseline passes only if every `(d,N)` mean lies within this tolerance. The mild combined-map check passes only if every mean condition with `gamma <= 1` does so. An exact distance check must also verify the residual isometry to numerical precision. Strong `gamma=4` conditions are stress tests rather than part of the mild-deformation PASS rule.

The report correlates error with neighborhood radius, condition number, log-volume variation, and active-set change, but treats these as descriptive mechanism diagnostics. The crossed interventions and full profiles are the primary evidence.

## BID estimator audit

`bid-audit` tests two claims separately. The implementation/calibration claim passes only if complete-histogram BID is within 10% of `k` for both independent `k`-bit streams and sign-binarized fixed-support amplitudes. The stronger activation claim passes only if population-centered activation-sign BID is within 10% of `k` for every fixed-support condition at the prespecified local cutoff `alpha_max=0.2`. Paper-style two-bit activation codes are scored separately; adding bits is not assumed to leave BID invariant.

Calling BID a *better estimate of continuous k* additionally requires its mean fixed-support relative error to be lower than matched rank-2 GRIDE. This numerical rule is interpreted only after inspecting empirical/model Hamming histograms and the `alpha_max` sweep. A fit failure, fewer than three populated fitted bins, or qualitative cutoff dependence blocks the claim rather than being dropped.

For pooled supports, the population continuous local dimension remains `k` almost everywhere, but activation BID is a global binary-distance statistic. The exact support-mask view is therefore a diagnostic of support combinatorics, not a second estimate of conditional `k`.

## Deliberately excluded from this first stage

- No learned encoder or reconstruction objective.
- The original hypotheses contain no irrelevant-latent or noise controls; the separate manifold audit adds only controlled curvature and support-continuity diagnostics.
- No claim about language models or semantic features.
- No selection of a single privileged GRIDE scale after looking at outcomes.

Those extensions belong after these two simple interventions have been inspected.
