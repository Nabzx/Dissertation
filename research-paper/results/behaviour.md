# Behavioural metrics (final 100 episodes, mean +/- s.d. across seeds)

| Metric | selfish | cooperative | mixed | selfish_indep | cooperative_indep |
|---|---|---|---|---|---|
| Territoriality (higher = more separated) | 0.912 +/- 0.023 | 0.945 +/- 0.010 | 0.919 +/- 0.007 | 0.979 +/- 0.006 | 0.996 +/- 0.002 |
| Passivity (higher = more inaction) | 0.510 +/- 0.041 | 0.575 +/- 0.030 | 0.552 +/- 0.022 | 0.620 +/- 0.060 | 0.868 +/- 0.021 |

## Key comparisons (Welch t-test)

| Metric | comparison | diff | p | Cohen's d |
|---|---|---|---|---|
| territoriality | selfish vs cooperative | -0.033 | 0.0307 * | -1.83 |
| territoriality | cooperative vs cooperative_indep | -0.051 | 0.0002 *** | -6.02 |
| territoriality | selfish_indep vs cooperative_indep | -0.017 | 0.0253 * | -4.05 |
| territoriality | selfish vs mixed | -0.008 | 0.5196 | -0.44 |
| passivity | selfish vs cooperative | -0.065 | 0.0232 * | -1.81 |
| passivity | cooperative vs cooperative_indep | -0.293 | 0.0000 *** | -10.73 |
| passivity | selfish_indep vs cooperative_indep | -0.248 | 0.0113 * | -5.55 |
| passivity | selfish vs mixed | -0.042 | 0.0885 | -1.28 |