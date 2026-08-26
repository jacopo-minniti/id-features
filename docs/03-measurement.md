# Measurement protocol

## GRIDE

The code calls DADApy's `Data.compute_distances(maxk=...)` and `return_id_scaling_gride(range_max=...)`. GRIDE compares successively larger nearest-neighbour ranks. Its saved profile contains the upper rank, a physical distance scale, ID estimate, and DADApy error estimate.

The default ranks are determined by `gride_range_max=64`; DADApy returns estimates at doubling neighbourhood scales. All estimates use the same sample set for a condition. Repeats change both dictionary/dataset random draws as defined in `config.json`.

## Linear probes

Each feature label is binary and imbalanced because exactly `k` of `m` features are on. Therefore the code uses class-balanced logistic regression and reports AUROC, which is threshold independent, alongside balanced accuracy. It also reports the mean signed decision value divided by the probe weight norm: the average Euclidean signed distance from the fitted separating hyperplane. Training and test samples are generated independently but share the exact same feature matrix for that replicate.

## Reproducibility

The configuration and every per-repeat raw measurement are written before interpretation. A fixed `--seed` reconstructs the random streams. In `load-capacity`, each repeat constructs one largest isotropic matrix and capacity conditions use column prefixes of it. In `load-ratio`, all `k` values at one `D` share a dictionary. In `geometry-control`, a fresh matrix is constructed per `rho`, because changing geometry is the intervention.
