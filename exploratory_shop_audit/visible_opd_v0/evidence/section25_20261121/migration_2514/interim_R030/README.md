# Stage E at the shared update-30 checkpoint

Each sampled rate uses 200 tasks and four trajectories per task. R is continuing toward update 50 on the four-card host. P stopped at migration, P-frozen completed its mechanism pilot, and R-matched reached its registered early-stop condition. These are single-seed results.

| Arm | Step | T_opd_eval successes / 800 | V successes / 800 | Fixed-probe BF16 KL |
|---|---:|---:|---:|---:|
| B | 0 | 317 | 415 | 0.000000 |
| P | 30 | 276 | 383 | 0.237342 |
| P-frozen | 30 | 289 | 376 | 0.274444 |
| R-matched | 30 | 15 | 6 | 0.295700 |
| R | 30 | 190 | 234 | 0.103489 |

[Sampled curves](sample_curves.png) · [Greedy curves](greedy_curves.png) · [All checkpoint values](checkpoint_curves.tsv).

## Paired all-requirements purchase rates

Contrasts use the four sampled episodes per task and ASIN-cluster bootstrap intervals. V contains 200 unseen-ASIN tasks and 0 seen-ASIN tasks. P comparisons between trained arms use their last shared checkpoint.

| Comparison | Role | Checkpoints (left, right) | T_opd_eval difference, pp [95% CI] | V-unseen difference, pp [95% CI] |
|---|---|---:|---:|---:|
| P-R-matched | Primary | 30, 30 | +32.62 [+26.88, +38.50] | +47.12 [+41.12, +53.25] |
| P-B | Primary | 30, 0 | -5.12 [-10.00, -0.38] | -4.00 [-9.50, +1.38] |
| P-R | Secondary | 30, 30 | +10.75 [+6.25, +15.25] | +18.62 [+13.38, +23.88] |
| R-B | Secondary | 30, 0 | -15.88 [-20.50, -11.38] | -22.62 [-28.25, -17.25] |

At update 10 in stage D, P minus P-shuffled was +6.62 [+4.25, +9.00] pp on T_opd_eval and +7.12 [+5.00, +9.38] pp on V. The stage D record includes the R-beta and shuffled controls. [Stage D paired outcomes](../../../section23_20261120/behavior_report.json).
