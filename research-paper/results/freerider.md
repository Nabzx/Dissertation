# Free-riding metrics (final 100 episodes, mean +/- std [95% CI])

| Metric | selfish | cooperative | selfish_indep | cooperative_indep |
|---|---|---|---|---|
| Free-rider fraction | 0.251 +/- 0.015 [±0.019] | 0.307 +/- 0.022 [±0.028] | 0.278 +/- 0.012 [±0.031] | 0.418 +/- 0.048 [±0.118] |
| Contribution Gini | 0.331 +/- 0.012 [±0.014] | 0.384 +/- 0.005 [±0.006] | 0.350 +/- 0.014 [±0.036] | 0.490 +/- 0.039 [±0.097] |

## Cooperative vs others (Welch t-test; higher = more free-riding)

| Metric | comparison | diff | p | Cohen's d |
|---|---|---|---|---|
| Free-rider fraction | cooperative vs selfish | +0.056 | 0.0021 ** | 2.99 |
| Free-rider fraction | cooperative vs selfish_indep | +0.029 | 0.0540 | 1.50 |
| Free-rider fraction | cooperative vs cooperative_indep | -0.111 | 0.0429 * | -3.37 |
| Contribution Gini | cooperative vs selfish | +0.053 | 0.0002 *** | 5.98 |
| Contribution Gini | cooperative vs selfish_indep | +0.034 | 0.0461 * | 3.76 |
| Contribution Gini | cooperative vs cooperative_indep | -0.106 | 0.0417 * | -4.62 |

_* p<0.05, ** p<0.01, *** p<0.001._