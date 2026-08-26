# Theory and minimal model

## Controlled representation

The representation is generated, not trained:

\[
h = Wz, \qquad W \in \mathbb{R}^{D \times m}, \qquad m > D.
\]

Every column of `W` is a unit-norm feature direction. An example activates exactly `k` columns and gives each active latent a positive amplitude sampled uniformly from `[0.5, 1.5]`. Since `z` is known, the experiment knows the correct label for every feature and every example.

For a fixed support `S`, the continuous part of the representation has dimension `rank(W_S)`, generically `k` for `k <= D`. The entire sample distribution is a union of many such support-conditioned sets, so a finite-scale ID estimator can legitimately see more than `k` at larger neighbourhoods. This is why the implementation saves a GRIDE profile rather than treating ID as a single number.

## Why accessibility may vary with load

For active feature `i`, projecting on its feature direction yields its own positive amplitude plus interference from the other active directions. In an isotropic random dictionary the pairwise dot-product variance is approximately proportional to `1 / D`; with `k` active directions, interference increases roughly with `k / D`. This is a motivation for testing a relationship, not a derivation that ID determines decoding.

## Geometry control

The correlation sweep uses

\[
w_i(\rho) = \frac{\sqrt{1-\rho}u_i + \sqrt{\rho}c}
{\lVert\sqrt{1-\rho}u_i + \sqrt{\rho}c\rVert}.
\]

`c` is shared and `u_i` is feature-specific. `rho` changes feature alignment without changing the support size. The saved mean off-diagonal dot product is the realized geometry quantity to inspect, since normalization means `rho` is only a construction parameter.
