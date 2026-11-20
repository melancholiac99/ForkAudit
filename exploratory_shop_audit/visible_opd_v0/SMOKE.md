# VC-OPD v0.4 eight-task GPU preflight

The run shop_visible_opd_v04_20261120 completed 120 native episodes across five packet conditions and 540 environment steps. The scorer reconstructed candidate-conditional distributions for 106 unprivileged response prefixes and 1,728 generated tokens. Empty packet IDs matched the student input and the largest repeated empty-score difference was zero. HF greedy argmax agreed with the vLLM output on 117/117 click tokens and 72/73 option tokens. Generation used vLLM BF16 and the paired scoring differences used a single HF BF16 model with FP32 log-softmax at temperature one.

| Condition | Sample, all requirements (16 episodes) | Greedy (8 episodes) |
|---|---:|---:|
| No packet | 6/16 | 5/8 |
| Full | 8/16 | 5/8 |
| Product | 10/16 | 4/8 |
| Evidence on current results | 6/16 | 4/8 |
| Full-template placeholder | 8/16 | 4/8 |

For full and product, 66/586 eligible click/option positions crossed both frozen content thresholds relative to their respective placeholders. Evidence crossed both thresholds at 13/288 positions. Only eight independent tasks contributed. This throughput pilot did not select a teacher.

A parser correction was required after the first report. 103 of 106 unprivileged outputs used native bare commands. CPU replay reconstructed every original input token and terminal reason. A versioned sidecar corrected only the categories and all pre-correction reports were retained. Final counts: 727 search, 347 click, 239 option, 408 format and 7 unknown tokens.

GPU allocation: 0.13983 GPU-hours for behavior, 0.05029 for scoring, total 0.19012. Linear extrapolation for the remaining 92 frozen tasks is 2.1864 GPU-hours. This includes pilot initialization overhead and is not a throughput guarantee. The proposed additional ceiling is 2.5 GPU-hours. Formal packet selection and actor updates have not run.
