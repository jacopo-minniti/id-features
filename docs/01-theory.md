# Theory and minimal model

## Controlled representation

The representation is generated, not trained:

\[
h = Wz, \qquad W \in \mathbb{R}^{D \times m}, \qquad m > D.
\]

Every column of `W` is a unit-norm feature direction. An example activates exactly `k` columns and gives each active latent a positive amplitude sampled uniformly from `[0.5, 1.5]`. Since `z` is known, the experiment knows the correct label for every feature and every example.

For a fixed support `S`, the continuous part of the representation has dimension `rank(W_S)`, generically `k` for `k <= D`. The entire sample distribution is a union of many such support-conditioned sets, so a finite-scale ID estimator can legitimately see more than `k` at larger neighbourhoods. This is why the implementation saves a GRIDE profile rather than treating ID as a single number.

The `support-pool` control makes this distinction measurable. It samples equally from a fixed bank of `B` exact-k supports and varies total sample count `N`. Conditional recovery predicts $d_{ID} \approx k$ only where the neighbors used by a GRIDE rank remain on the query's support. The pooled estimate is allowed to depart from `k` as `B`, rank, or insufficient samples per support cause neighborhoods to cross supports.

## Boundaryless known-ID sparse-feature model

The estimator audit replaces disconnected bounded support patches with a connected boundaryless latent manifold:

\[
u \sim \operatorname{Uniform}(S^d),\qquad
z(u)=\operatorname{ReLU}(Au-\tau),\qquad
h(u)=Qu+\gamma Wz(u).
\]

Here `u` lies on the unit sphere in $\mathbb{R}^{d+1}$, `Q` has orthonormal columns, and every column of `W` lies in the orthogonal complement of `Q`. Therefore

\[
Q^T h(u)=u,
\]

so the combined map is injective and its population intrinsic dimension is exactly `d` for every finite `gamma`. ReLU makes the map piecewise smooth and changes active supports continuously through zero, but the residual prevents rank collapse. The sparse-only branch $Wz(u)$ lacks this guarantee and is retained only as a warning control.

The threshold is chosen analytically. For a unit feature direction `a` and uniform $u\in S^d$, $(a^T u)^2$ follows $\operatorname{Beta}(1/2,d/2)$. The code inverts this distribution so the marginal activation probability is `s/m`, giving expected active count `s`.

The sparse branch is normalized to unit centered RMS variation before multiplying by `gamma`. Thus `gamma` has a comparable global meaning across `d` and `s`. Because the residual and sparse output subspaces are orthogonal, if the sparse tangent singular values are $\sigma_i(u)$, the combined tangent singular values are exactly

\[
\sqrt{1+\gamma^2\sigma_i(u)^2}.
\]

This gives exact diagnostics for local anisotropy and log-volume variation while leaving true ID fixed.

Even this boundaryless baseline is not exactly Euclidean at finite radius. For Euclidean chord radius `r` on the unit sphere, the small-ball volume is

\[
V(r)=C_d r^d\left[1-\frac{d(d-2)}{8(d+2)}r^2+O(r^4)\right].
\]

For `d > 2`, its finite-radius log-volume slope is below `d`. Since nearest-neighbor radius scales as $N^{-1/d}$, the leading curvature bias decays only as $N^{-2/d}$. The experiment therefore treats boundary removal as necessary but not sufficient: neighborhoods must also be small relative to the manifold's curvature and metric-variation scales.

## Why accessibility may vary with load

For active feature `i`, projecting on its feature direction yields its own positive amplitude plus interference from the other active directions. In an isotropic random dictionary the pairwise dot-product variance is approximately proportional to `1 / D`; with `k` active directions, interference increases roughly with `k / D`. This is a motivation for testing a relationship, not a derivation that ID determines decoding.

## Geometry control

The correlation sweep uses

\[
w_i(\rho) = \frac{\sqrt{1-\rho}u_i + \sqrt{\rho}c}
{\lVert\sqrt{1-\rho}u_i + \sqrt{\rho}c\rVert}.
\]

`c` is shared and `u_i` is feature-specific. `rho` changes feature alignment without changing the support size. The saved mean off-diagonal dot product is the realized geometry quantity to inspect, since normalization means `rho` is only a construction parameter.
