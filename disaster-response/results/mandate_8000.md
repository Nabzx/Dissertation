# Disaster mandate sweep

- **alpha=0.0**: n=3, seeds=[0, 1, 2]
- **alpha=0.25**: n=3, seeds=[0, 1, 2]
- **alpha=0.5**: n=3, seeds=[0, 1, 2]
- **alpha=0.75**: n=3, seeds=[0, 1, 2]
- **alpha=1.0**: n=1, seeds=[0]

## Final-window metrics (mean +/- s.d. [95% CI])

| Metric | a=0.0 | a=0.25 | a=0.5 | a=0.75 | a=1.0 |
|---|---|---|---|---|---|
| Lives saved | 10.861 +/- 0.726 [±1.803] | 12.844 +/- 1.068 [±2.653] | 12.967 +/- 1.560 [±3.876] | 13.344 +/- 1.558 [±3.871] | 12.000 (n=1) |
| Save rate | 0.272 +/- 0.018 [±0.045] | 0.321 +/- 0.027 [±0.066] | 0.324 +/- 0.039 [±0.097] | 0.334 +/- 0.039 [±0.097] | 0.300 (n=1) |
| Severe save rate | 0.198 +/- 0.019 [±0.046] | 0.225 +/- 0.021 [±0.051] | 0.220 +/- 0.034 [±0.084] | 0.230 +/- 0.013 [±0.032] | 0.188 (n=1) |
| Minor save rate | 0.320 +/- 0.028 [±0.069] | 0.385 +/- 0.032 [±0.080] | 0.395 +/- 0.048 [±0.120] | 0.404 +/- 0.055 [±0.135] | 0.372 (n=1) |
| Joint rescues | 3.189 +/- 0.462 [±1.148] | 3.622 +/- 0.338 [±0.841] | 3.567 +/- 0.737 [±1.830] | 3.678 +/- 0.059 [±0.145] | 2.950 (n=1) |
| Idle rate | 0.017 +/- 0.008 [±0.019] | 0.020 +/- 0.008 [±0.021] | 0.030 +/- 0.008 [±0.019] | 0.021 +/- 0.006 [±0.016] | 0.044 (n=1) |

## H1 — does an individual mandate suppress cooperative rescues?

| Metric | comparison | diff | p | Cohen's d |
|---|---|---|---|---|
| Severe save rate | a=0.0 vs a=1.0 | +0.010 | n/a | n/a |
| Minor save rate | a=0.0 vs a=1.0 | -0.052 | n/a | n/a |
| Joint rescues | a=0.0 vs a=1.0 | +0.239 | n/a | n/a |
| Lives saved | a=0.0 vs a=1.0 | -1.139 | n/a | n/a |

_H1 predicts severe save rate and joint rescues fall as alpha rises._