# Measurement protocol

## GRIDE

The code calls DADApy's `Data.compute_distances(maxk=...)` and `return_id_scaling_gride(range_max=...)`. GRIDE compares successively larger nearest-neighbour ranks. Its saved profile contains the upper rank, a physical distance scale, ID estimate, and DADApy error estimate.

The default ranks are determined by `gride_range_max=64`; DADApy returns estimates at doubling neighbourhood scales. All estimates use the same sample set for a condition. Repeats change both dictionary/dataset random draws as defined in `config.json`.

For `support-pool`, every GRIDE row also records the mean fraction of the first `r` neighbors sharing the query's exact support and the fraction of queries for which all first `r` neighbors share it. The latter is the pre-specified support-purity diagnostic. Support banks use prefixes of one largest bank, and per-support amplitude streams are reset so increasing `N` extends rather than replaces the smaller sample.

For `manifold-geometry`, sample sets are prefixes of one largest sphere draw within each `(repeat,d)` pair. The residual baseline reuses the sphere GRIDE profile only after verifying that sampled pairwise distances agree to `1e-12`; an isometry cannot change a distance-based estimator. The sparse and combined conditions record 1-NN latent chord length, support Jaccard, and exact-support-match fraction.

Local Jacobian diagnostics use a fixed prefix of sphere points. The code computes tangent singular values of the normalized sparse branch exactly. For combined representations it uses orthogonality to obtain singular values $\sqrt{1+\gamma^2\sigma_i^2}$, then records full-rank fraction, mean and 95th-percentile condition number, and the standard deviation of local log-volume. The last quantity measures induced-density heterogeneity: uniform latent sampling need not be uniform with respect to volume in the deformed representation metric.

## Binary intrinsic dimension

The `bid-audit` command uses `dadapy.hamming.Hamming` and `BID`, not DADApy's older continuous binomial-ID methods. Inputs are spins in `{-1,+1}`. Hamming distances are computed for every pair of samples, and BID fits

$$
P(r) \propto 2^{-d(r)} {d(r) \choose r}, \qquad d(r)=d_0+d_1r,
$$

by minimizing the KL divergence to the empirical distance histogram. The reported BID is the small-distance intercept `d0`. Following the official tutorial, optimization uses `delta=5e-3` and `100000` steps. The activation-code primary uses the package's local default `alpha_max=0.2`; the audit also sweeps `alpha_max={0.1,0.2,0.3,0.5,1}` because the result should be qualitatively stable to this fit cutoff. Exact independent-bit controls use the complete histogram at `alpha_max=1`. Fits with fewer than three populated distance bins are recorded as invalid rather than optimized.

The continuous activation is centered by its exact balanced-support population mean before binarization. One-bit encoding applies the sign coordinatewise. Two-bit encoding follows the paper exactly: each centered activation is assigned `00`, `01`, `10`, or `11` using thresholds `-sigma`, `0`, and `+sigma`, where `sigma` is the exact global population RMS across samples and representation coordinates. Natural binary digits are then converted to `-1/+1` spins.

Every fit saves `d0`, `d1`, log-KL, total-variation error, maximum probability error, accepted-move fraction, bit imbalance, constant-bit fraction, unique pattern count, and the fitted empirical/model probabilities. This is required because a numerically plausible `d0` is not evidence when the Hamming ansatz fits poorly or changes strongly with `alpha_max`.

For `B=1`, continuous population ID is exactly `k`, but BID is not mathematically forced to equal it after coordinatewise binarization. The independent `k`-bit and latent-amplitude-sign views are positive controls for `BID=k`; activation signs and two-bit activation codes test whether the binary encoding preserves that estimand. For `B>1`, support-mask BID is labeled as support-combinatorial complexity rather than conditional continuous ID.

## Linear probes

Each feature label is binary and imbalanced because exactly `k` of `m` features are on. Therefore the code uses class-balanced logistic regression and reports AUROC, which is threshold independent, alongside balanced accuracy. It also reports the mean signed decision value divided by the probe weight norm: the average Euclidean signed distance from the fitted separating hyperplane. Training and test samples are generated independently but share the exact same feature matrix for that replicate.

## Reproducibility

The configuration and every per-repeat raw measurement are written before interpretation. A fixed `--seed` reconstructs the random streams. In `load-capacity`, each repeat constructs one largest isotropic matrix and capacity conditions use column prefixes of it. In `load-ratio`, all `k` values at one `D` share a dictionary. In `geometry-control`, a fresh matrix is constructed per `rho`, because changing geometry is the intervention.

In `manifold-geometry`, each `(repeat,d)` pair shares one sphere sample, residual map, ReLU direction matrix, and orthogonal feature dictionary across every `N`, activity, and `gamma` condition. Only the named threshold or strength changes.
