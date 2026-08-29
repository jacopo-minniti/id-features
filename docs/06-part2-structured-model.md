# Part 2: a structured sparse-feature proof of concept

Part 1 proved that population local intrinsic dimension equals `k` when `k` active amplitudes vary independently and the active feature directions have rank `k`. Part 2 asks what survives after adding a small amount of structure motivated by language-model representations.

The goal is still a toy model. “More LLM-like” means that the generator now has several qualitative properties often associated with learned features. It does **not** mean that it is a language model or that success transfers automatically to one.

## The question

The raw hypothesis

> GRIDE ID is approximately the number of active features `k`

is too strong in general. The more defensible hypothesis is

> At a fixed scale, GRIDE ID tracks the number of independently varying and geometrically resolvable active feature directions. Raw `k` is a good proxy only when active features behave approximately like independent coordinates.

The Part 2 suites intervene on the different reasons why raw `k` can cease to be this effective count.

## The structured generator

We retain the same untrained superposition model

\[
h=Wz, \qquad W\in\mathbb{R}^{D\times m}, \qquad m>D.
\]

Every example still activates exactly `k` features. This is important: changes in GRIDE at fixed `k` cannot be attributed to silently changing sparsity.

The `m` features are divided into `G` equal modules. Each example first samples a context module `c`. Its exact-`k` support is sampled without replacement with weight

\[
\log p_i(c)
=
-\alpha\log r_i
+\beta\,\mathbf 1[g(i)=c],
\]

where:

- `g(i)` is the module containing feature `i`;
- `r_i` is the feature's frequency rank inside its module;
- `beta` controls modular co-occurrence;
- `alpha` controls a Zipf-like frequency tail.

Active amplitudes are positive and have matched marginal distributions:

\[
z_i
=
\exp\!\left[
s\left(
\sqrt{\rho}\,u_{g(i)}
+\sqrt{1-\rho}\,\epsilon_i
\right)-\frac{s^2}{2}
\right],
\qquad i\in S.
\]

Here `u_g` is shared by active features in module `g`, while `epsilon_i` is feature-specific.

- At `rho=0`, all active amplitudes vary independently.
- For `0<rho<1`, the exact population source rank remains `k`, but some directions carry much less independent variation.
- At `rho=1`, features in the same module move together exactly. The feature-source rank becomes the number of active modules, which can be much smaller than `k`.

Feature directions also have modular geometry:

\[
w_i
=
\operatorname{normalize}\!\left(
\sqrt{1-\gamma}\,v_i
+\sqrt{\gamma}\,c_{g(i)}
\right).
\]

`gamma` controls within-module coherence. As long as the active matrix remains full rank, coherence does not change the population dimension. It does change conditioning and can therefore change the dimension that GRIDE resolves at finite scale.

Finally, the noise suite adds isotropic Gaussian noise relative to the representation's centered RMS. Any nonzero full-dimensional noise makes the formal infinitesimal population dimension `D`, but whether finite-sample GRIDE reaches that scale is an empirical question.

## Why these properties are LLM-motivated

| Property | Motivation | What it tests |
| --- | --- | --- |
| `m>D` superposition | More candidate features than residual-stream coordinates | Whether sparse features can coexist in a compressed representation |
| Modular supports | Related features tend to co-occur in common contexts | Whether support organization changes finite-scale ID at fixed `k` |
| Zipf-like frequencies | Some learned or semantic features are common while many are rare | Whether sampling density and rare directions distort pooled ID |
| Shared module amplitudes | Co-occurring features may be driven by a common underlying factor | Whether ID follows independent factors rather than nonzero count |
| Coherent feature directions | Learned dictionaries often contain correlated or clustered directions | Whether geometry and conditioning limit GRIDE recovery |
| Dense small noise | Residual representations contain variation not captured by the named sparse features | Whether GRIDE exhibits a scale transition toward ambient dimension |

Each property is only an analogy. The point is to obtain controlled mechanisms, not realism for its own sake.

## The six experiment suites

All suites use `D=32`, `m=256`, `G=8`, `k={2,4,8}`, `N={4,096,16,384}`, five paired repeats, and GRIDE ranks `2,4,8,16,32,64`.

| Suite | Intervention | Fixed controls | Main expectation |
| --- | --- | --- | --- |
| `support` | `beta={0,1,2,4}` | uniform frequencies, independent amplitudes, isotropic `W` | Population conditional ID stays `k`; neighborhood organization and large-rank GRIDE may change |
| `amplitude` | `rho={0,0.5,0.9,1}` | strongly modular support, isotropic `W` | GRIDE should move from raw `k` toward effective amplitude dimension; `rho=1` has known reduced source rank |
| `geometry` | `gamma={0,0.25,0.5,0.75,0.9}` | strongly modular support, independent amplitudes | Population ID remains `k` while finite-scale recovery worsens with conditioning |
| `frequency` | `alpha={0,0.5,1,1.5}` | strongly modular support, independent amplitudes, isotropic `W` | Rare-feature imbalance changes density and neighborhood composition without changing exact `k` |
| `noise` | relative noise `{0,0.01,0.05,0.2}` | modular, Zipfian, correlated-amplitude, coherent model | Small-scale ID can move toward `D`; the rank profile should reveal whether that scale is resolved |
| `combined` | named baseline, single-property controls, and two LLM-like combinations | paired seeds and sample prefixes | Test whether raw `k` or effective degrees better organize the complete structured model |

The combined suite includes `baseline`, `modular_support`, `zipf_frequencies`, `shared_amplitudes`, `coherent_geometry`, `llm_like`, and `llm_like_exact_shared`.

## Known diagnostics saved with every measurement

The experiment does not ask GRIDE to explain itself. It records the known mechanism variables:

- mean number of active modules;
- participation ratio of the log-amplitude covariance;
- exact feature-source rank, including the rank reduction at `rho=1`;
- formal small-scale rank after observation noise;
- full-rank fraction, minimum singular value, and condition number of sampled active matrices `W_S`;
- realized within-module and between-module feature dot products;
- feature-frequency entropy and coefficient of variation;
- nearest-neighbour support overlap, module overlap, and context agreement;
- the full GRIDE rank profile and two nested sample counts.

Common random streams pair interventions within each repeat. Increasing `N` extends rather than replaces the smaller dataset.

## Pre-run interpretation rules

1. A high Spearman correlation between raw `k` and GRIDE is not a sufficient result.
2. Raw-`k` tracking requires both monotonicity and approximate calibration. A 10% rank-2 band is used descriptively because Part 1 calibrated that regime through `k=8`.
3. `k=8` is an edge condition: Part 1 did not establish profile-wide calibration beyond rank 2. `k<=4` is the conservative profile-wide regime.
4. Every structural claim must come from a paired change at fixed `k`, `D`, `m`, `N`, and repeat.
5. Amplitude dependence is supported as a mechanism only if GRIDE moves toward the saved effective-amplitude or source-rank diagnostic as `rho` increases.
6. Geometry is an estimator effect while every active `W_S` remains full rank; an actual rank failure is reported separately.
7. Noise conclusions must use the full rank profile. A change at only one favorable rank is not called a general dimension change.
8. Conflicting signs across repeats or strong dependence on `N` remain `NOT ESTABLISHED`, even if a pooled curve looks smooth.

## Reproduction and launch

Run one suite directly with, for example:

```bash
uv run id-features part2-structured \
  --suite amplitude \
  --output results/part2-amplitude
```

Launch every predeclared suite as a six-task CPU array with:

```bash
sbatch scripts/run_part2_structured.sh
```

Each task writes `config.json`, `summary.csv`, `gride_profiles.csv`, `interpretation.md`, and `overview.png` to a suite-specific results directory.

The launch status and completed findings will be added only after checking scheduler state, durable progress, artifact row counts, and final profiles.

### Launch status — 2026-08-28

Slurm array `169194` was submitted with the following fixed task map:

| task | suite | expected measurements |
| ---: | --- | ---: |
| `169194_0` | support | 120 |
| `169194_1` | amplitude | 120 |
| `169194_2` | geometry | 150 |
| `169194_3` | frequency | 120 |
| `169194_4` | noise | 120 |
| `169194_5` | combined | 210 |

All six tasks entered `RUNNING` and subsequently completed. Artifact verification found:

| suite | completed measurements | GRIDE profile rows | status |
| --- | ---: | ---: | --- |
| support | 120 | 720 | complete |
| amplitude | 120 | 720 | complete |
| geometry | 150 | 900 | complete |
| frequency | 120 | 720 | complete |
| noise | 120 | 720 | complete |
| combined | 210 | 1,260 | complete |

This is 840 complete measurements and 5,040 saved GRIDE estimates in total. Every suite produced all five expected files, no CSV contained a saved `NaN` or `Inf`, and the Slurm logs contained no runtime error, traceback, or resource failure. The completed paired-effect and profile synthesis is in [Part 2 results](07-part2-results.md).
