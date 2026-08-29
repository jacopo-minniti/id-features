# BID audit results

## Completed run

Slurm job `169146` completed all 84 measurements with `D=64`, `m=256`, `N=2560`, `B={1,64}`, `k={2,4,8,16}`, and three repeats. Activation BID uses the predeclared local cutoff `alpha_max=0.2`; independent-bit controls use the complete histogram.

For fixed support (`B=1`), population continuous ID is exactly `k`. All reconstructed `W_S` matrices were full rank; the smallest singular value across the run was `0.413`.

| k | GRIDE rank 2 | independent-bit BID | latent-sign BID | activation-sign BID | activation two-bit BID |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.951 | 1.997 | 1.996 | 2.950 | 10.076 |
| 4 | 3.849 | 3.996 | 3.992 | 13.780 | 30.023 |
| 8 | 7.134 | 7.985 | 7.988 | 23.289 | 57.392 |
| 16 | 12.692 | 15.969 | 15.968 | 32.902 | 72.709 |

Mean fixed-support relative error was `9.4%` for rank-2 GRIDE, `147.2%` for activation-sign BID, and `506.5%` for two-bit activation BID. Therefore BID is correctly calibrated on genuine independent binary coordinates but is not a better estimator of continuous `k` after coordinatewise binarization of `h`.

## What the independent-bit columns mean

The independent-bit view samples exactly `k` balanced independent spins. The latent-sign view applies `sign(a_S - 1)` to the actual independent amplitudes. Their recovery of approximately `0.998 k` verifies the BID implementation, optimizer, and sample count. They are positive controls, not estimates computed from the mixed activation coordinates.

For activations,

$$
\operatorname{sign}(h-\mathbb E h)
=\operatorname{sign}(W_S(a_S-\mathbf 1)).
$$

The `D=64` activation bits describe cells in a hyperplane tessellation of the `k`-dimensional amplitude space. BID measures the binary code's small-distance complexity, which depends on `D`, the orientations in `W_S`, and the quantization protocol—not only on continuous `k`. The two-bit encoding adds thresholds and resolves more cells, so its BID is larger still.

## Reading the support-mixing panel

The middle panel plots `BID/k`, not absolute error. Several curves move toward the ideal line as `k` grows, but this is mainly a saturation/denominator effect. For example, with `B=64`, one-bit BID changes only from `28.68` to `36.66` while `k` grows from `2` to `16`:

| k | one-bit BID | BID / k |
| ---: | ---: | ---: |
| 2 | 28.68 | 14.34 |
| 4 | 33.23 | 8.31 |
| 8 | 35.06 | 4.38 |
| 16 | 36.66 | 2.29 |

Dividing a nearly saturated binary complexity by a growing `k` makes the ratio approach one without demonstrating recovery. Even at `k=16`, one-bit BID remains approximately `2.1k` for `B=1` and `2.3k` for `B=64`; two-bit BID remains much farther away.

The pooled `B=64` BID values are also quantitatively unreliable: their Hamming-model total-variation errors are about `0.25-0.33`, and the estimates change strongly with `alpha_max`. They indicate global support/code complexity rather than conditional continuous dimension.

## Conclusion and next control

The audit rejects activation-coordinate BID as a plug-in estimator of continuous active-feature count in this model. It does not reject BID for its intended purpose of comparing binary correlation and complexity scalings.

The next decisive control is to hold `k` fixed and sweep `D`. Continuous ID remains `k`; a changing activation BID would directly show dependence on the number of hyperplanes/bits.

Raw artifacts are in [`results/bid-audit-169146/`](../results/bid-audit-169146/), with the detailed analysis in [`analysis.md`](../results/bid-audit-169146/analysis.md).
