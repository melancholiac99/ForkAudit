# E4/M0 counterfactual credit predictability audit

Source: e4; 6016 points, 308 game groups; 3 replays/arm.
Nonzero observed gaps: 1586; zero gap: 73.6%.
Out-of-fold MSE: 0.03961; fold-mean baseline: 0.04504; action-only MSE: 0.04312.

## Full-data gap bins (ties preserved)

| Bin | N | Observed gap | Predicted difference | Positive clipped ratio |
|---:|---:|---:|---:|---:|
| 1 | 894 | 0.333 | 0.005 | 0.043 |
| 2 | 462 | 0.667 | 0.014 | 0.035 |
| 3 | 230 | 1.000 | 0.034 | 0.039 |

## Independent-replicate check (select on n−1, evaluate on held-out 1)

| Bin | N | Selection gap | Held-out realized gap | Predicted difference |
|---:|---:|---:|---:|---:|
| 1 | 2818 | 0.500 | 0.053 | 0.006 |
| 2 | 1329 | 1.000 | 0.279 | 0.020 |

## Interpretation guardrails

- A clipped predicted/observed gap ratio is **not** a proven realizability fraction. Its denominator also induces mechanical downward trends as observed gaps grow.
- E4 uses only three binary continuations per arm, so ties are common and ten distinct deciles may be impossible. Never split equal gap values to manufacture deciles.
- Top-minus-bottom on the same rollouts is biased upward (winner's curse). Use the independent replicate check before attributing a trend to hidden-state luck.
- Out-of-fold model failure may reflect limited model capacity, sparse successes or noisy returns; it is not proof that student-visible information is insufficient.
- This audit does not establish EVPI, causal training benefit, or the claimed negative trend. No validation trajectories are used as training data.
