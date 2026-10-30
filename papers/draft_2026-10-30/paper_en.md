# When Does an Action Matter? Information, Execution Bottlenecks, and Credit Structure in Long-Horizon Agents

Working paper v0.1 · October 30, 2026 · English version

> Draft convention. Numerical results below come from completed experiments. Sections explicitly marked **[PLANNED EXPERIMENT]** contain designs and result placeholders, not completed findings. Bibliographic entries and final statistical-protocol details remain to be added.

## Abstract

Credit assignment for long-horizon agents is commonly framed as an estimation problem: identify which actions influence success from sparse terminal rewards. We ask a more basic question: **when does an opportunity created by an action become a measurable difference in terminal return?** We measure action-conditioned outcomes at 3,038 ALFWorld decision points using Qwen2.5-7B-Instruct and multiple counterfactual replay banks. The resulting Dead, Indifferent, and Fork regimes reveal a division of labor across information sources: sibling trajectories distinguish Dead from Live states, whereas one-step environment feedback better identifies action forks. Information interventions show that truthful location clues recover substantial value in weak branches, but do not consistently eliminate action differences. A factorial experiment crossing information completion with execution repair reveals a mechanism behind this behavior. A shared execution obstacle can hide the value of acquiring information, while a branch-specific obstacle can remain as the source of credit after information has been equalized. In one diagnostic case, all five actions initially fail. Removing an inventory obstacle makes the information-revealing action succeed with probability 87.5%, while the other actions still fail. Providing the same information to every branch then raises all five success rates to 87.5%–100%. These findings shift attention from estimating existing credit to understanding the conditions under which credit emerges, disappears, or changes its source.

## 1. Introduction

Consider an agent tasked with finding a pen and placing it on a desk. It can visit the side table containing the pen or continue inspecting irrelevant containers. Visiting the side table appears to deserve greater credit. Yet suppose the agent already holds a pencil and cannot reliably release it. Whichever location it visits next, the task may end in failure. The informative action creates an opportunity, but the continuation policy does not convert that opportunity into reward.

This example exposes a joint dependence in long-horizon credit. The current action determines what information and physical state become available; the continuation policy determines whether these differences can be exploited. A shared downstream bottleneck can make terminal returns insensitive to an earlier information advantage. Removing that bottleneck can make a previously uninformative decision point become an actionable fork.

Our investigation began with information loss: when a rich forward decision process is compressed into an executed action and a factual trajectory, which structures relevant to long-term credit are omitted? We compare scalar confidence, decision structure, hidden states, sibling trajectories, one-step feedback, and counterfactual continuations. The evidence leads to a broader distinction. Some action structure is lost in its representation; other useful information has not yet become a return difference under the current policy.

We therefore separate three questions. Can outcomes still vary? Do the current actions produce different long-term outcomes? Is the selected action better or worse than its reference alternatives? These questions concern outcome variation, action heterogeneity, and signed credit, respectively. They need not be answered by the same information source.

Our central thesis is that **long-horizon credit is jointly formed by action-created opportunities and downstream execution**. Information-source comparisons provide observational clues to this structure. Targeted interventions expose how it is formed. We contribute:

1. A multi-action measurement framework separating outcome variation, action differences, and the signed credit of the selected action.
2. Evidence that information sources specialize across these objects, and that chosen-versus-mean credit can omit useful alternative-action preferences.
3. Information and execution interventions showing how shared obstacles hide action differences and how branch-specific obstacles remain after information completion.

**[Figure 1.]** An overview using the measured pen example: five actions create unequal information opportunities; a common obstacle blocks success; repair exposes an information fork; shared information subsequently reduces the difference.

## 2. From Action Values to Credit Structure

### 2.1 Measurement object

For a history prefix $h$ and a fixed candidate set $\mathcal A_h$, we force an action $a$ and continue with policy $\pi$ until termination:

$$q_a=Q^\pi(h,a)=\Pr(R=1\mid h,\operatorname{do}(a),\pi).$$

Our candidate set contains the factual action and four alternatives. We retain the complete five-action vector. Under a uniform reference over these candidates, define

$$\mu=\frac{1}{K}\sum_a q_a,\qquad C=\frac{1}{K}\sum_a(q_a-\mu)^2.$$

Here, $\mu$ measures average success opportunity and $C$ measures differences attributable to choosing among the tested actions. Improving every branch can increase $\mu$ without increasing $C$. Improving a subset of branches can either enlarge or reduce $C$.

For binary returns, the law of total variance yields

$$\operatorname{Var}(R)=\mathbb E_a[q_a(1-q_a)]+\operatorname{Var}_a(q_a).$$

The first term is outcome variability remaining after the current action is fixed. The second is variation in expected return across actions. This standard decomposition distinguishes uncertain outcomes from consequential current choices; it is not a new theoretical contribution.

### 2.2 Three regimes and three prediction tasks

Our frozen finite-replay protocol defines Dead points as those with constant observed outcomes. Among nonconstant, Live points, Fork denotes detected action heterogeneity and Indifferent denotes heterogeneity not detected at the available budget. Dead can include all-failure and all-success outcomes.

We evaluate B1, Dead versus Live; B2, Fork versus Indifferent within Live; and B3, the credit sign within the frozen intersection of heterogeneity and credit-significance criteria. B3 requires more than detecting a difference: it requires identifying which action produces which outcome.

The chosen action's uniform-reference credit is $A_{\mathrm{chosen}}=q_{\mathrm{chosen}}-\mu$. This scalar can be zero despite a large action spread. For example, the first action in $[.5,0,0,1,1]$ equals the mean, although clearly better and worse alternatives exist. Improvement opportunities and significant chosen-versus-mean credit are different objects.

### 2.3 From information opportunities to terminal credit

Let $q_a^{i,e}$ denote the value of the original action under information condition $i$ and downstream execution condition $e$. Execution interventions operate after the original action. If an intervention produces action-specific gains $d_a$, then

$$C_{\mathrm{after}}-C_{\mathrm{before}}=2\operatorname{Cov}_a(q_a,d_a)+\operatorname{Var}_a(d_a).$$

An increase in average success therefore does not determine the direction of change in action heterogeneity. The empirical question is which branches benefit. Does removing a shared obstacle primarily help branches that already possess useful information? Once information is shared, do branch-specific execution constraints remain? Our interventions directly test these questions.

## 3. Experimental Framework

We use ALFWorld and Qwen2.5-7B-Instruct with temperature 0.7, a 50-step episode budget, and the original two-transition interaction history. Counterfactual replay restores the same environment prefix, forces one of the five candidate actions, and continues with the fixed policy. Separate replay banks distinguish label construction, action selection, and subsequent evaluation. The Gold bank provides 20 independent continuations per action.

The main measurement set contains 3,038 counterfactually defined decision points. Information sources include scalar confidence A, hand-designed decision structure B, hidden states C, state value V, sibling trajectories G, one-step world feedback W1, and complete continuations at limited replay depth. W1-pure uses environment responses; W1-model includes model-derived assessments after observing action feedback.

We distinguish the frozen measurement study from subsequent exploratory mechanism experiments. Mechanism studies retain the original regime labels as strata. The two-action, five-action, and execution-repair studies explicitly reuse some records; they are not three independent samples.

**[Final protocol insertion.]** Appendix A will specify the frozen significance thresholds, candidate construction, grouped evaluation procedure, exact bank allocation, and task counts. These definitions will be imported from the existing protocol rather than reselected.

## 4. Credit Is Not a Single Step-Importance Score

### 4.1 Most points do not exhibit a detected action fork

Of 3,038 points, 1,560 are Dead (51.3%), 1,003 are Indifferent (33.0%), and 475 are Fork (15.6%). Much of the variability in long-horizon outcomes therefore does not appear as a detectable difference among the five current actions.

Policy uncertainty is not a direct substitute for this structure. A model can be uncertain among inconsequential options or confidently select an action with important consequences. We next compare information sources across the three identification tasks.

### 4.2 Siblings identify shared outcomes; peeks identify action forks

| Information source | B1 AUROC | B2 AUROC | B3 AUROC |
|---|---:|---:|---:|
| Scalar confidence | .53 | .52 | .52 |
| Decision structure B | .68 | .57 | .60 |
| Hidden states C | .75 | .59 | .59 |
| State value V | .73 | .51 | .62 |
| Sibling trajectories G | **.84** | .48 | .71 |
| W1-pure | .75 | **.67** | .60 |
| W1-model | .69 | .59 | .71 |
| Replay 6/arm | .93 | .76 | .88 |

Values are rounded to two decimals. The final table will import confidence intervals from frozen evaluation outputs. We exclude B+C stacking because its implementation issue has not been verified as resolved in the results used here.

Sibling trajectories are informative for B1 but nearly uninformative for B2. One-step responses offer a more direct signal of forks. A motivating interpretation is that siblings expose shared patterns of success and failure under the current policy, whereas peeks expose local opportunities created by different actions. This interpretation motivates interventions rather than ending the analysis at a feature ranking.

The single `target_revealed` feature reaches approximately .65 B2 AUROC, close to .67 for W1-pure. It indicates that at least one response mentions a target instance; it does not necessarily indicate newly acquired knowledge. The mechanism question is therefore whether making target-location information available changes subsequent success.

**[Figure 2.]** Regime proportions alongside the B1/B2 division of labor; B3 appears in a separate panel rather than being merged into a generic credit-accuracy score.

### 4.3 Scalar credit can omit improvement opportunities

Among 475 Fork points, 305 are not simultaneously detected by the chosen-versus-mean credit-significance test. We call these Masked Forks. At 53 Masked points with independent Gold evaluation, alternatives selected using earlier replay outperform the chosen action by approximately 35.6 percentage points on average, with a 95% interval of [26.2, 44.3].

A selected action near the reference mean can therefore coexist with useful preferences over alternatives. Masked Forks need not all arise from exact positive-negative cancellation. Their shared property is that the chosen-action test does not express the improvement opportunities in the complete action vector. This motivates retaining action–feedback correspondence.

## 5. Why Can One-Step Feedback Predict Long-Term Differences?

### 5.1 Completing missing information in weak branches

We extract a truthful target-location clue from a one-step observation and append the same clue after the original forced action. The original model then continues. The clue is presented once and expires through the normal memory window; the information intervention does not directly alter physical state.

A two-action study covers 32 tasks. Among its 24 original Fork points, average success across the two actions increases by 19.79 percentage points, while noise-corrected action variance decreases from .07457 to .01953. The weaker branches benefit more from the clue, reducing their original disadvantage.

We extend the study to all five original actions at 24 tasks: eight each from the original Dead, Indifferent, and Fork strata. Each action receives 16 continuations in each information condition. The 3,840 records include 1,024 explicitly reused records from the two-action study. In the Fork stratum, average success rises from 39.53% to 76.56%. Five-action variance decreases from .05113 to .03438 on average, but its reduction interval crosses zero: variance decreases at five points and increases at three.

**Information can substantially improve success without consistently reducing action differences.** This suggests that execution conditions determine which branches can realize the value of information.

### 5.2 Information can close a gap or expose a remaining obstacle

In the wine-bottle case s044_t18, the five success probabilities change from `[.8125,.0625,.25,.1875,.125]` to `[.9375,.6875,.875,.8125,.875]`. Information primarily restores opportunities in weaker branches.

In the statue case s010_t29, the vector changes from `[.3125,.25,.1875,0,1]` to `[.9375,1,1,0,1]`. Mean success increases, but one branch continues to fail. That action picks up an irrelevant pencil, preventing the agent from subsequently taking the statue. Equalizing information makes the remaining execution disparity more prominent.

Together, these observations motivate an intervention-based hypothesis: **long-term action differences depend on the joint distribution of information opportunities and execution obstacles across branches.**

**[Figure 3.]** Complete action profiles before and after information completion, contrasting weak-branch recovery with a remaining execution obstacle. Report both mean success and action variance.

## 6. Information × Execution: Revealing and Removing Credit

### 6.1 Factorial intervention

At six previously specified mechanism points, we cross location information with a one-time execution repair. Repair targets an irrelevant object held immediately after the original action. At the first legal placement opportunity, the controller releases it once and returns control to the model. Repair consumes a real step within the original 50-step budget and appears in normal interaction history.

The four conditions are native, information only, repair only, and information plus repair. Each point retains the five original actions with eight continuations per cell. Native and information-only conditions reuse corresponding earlier records. The two repair conditions add 480 continuations, yielding 960 analyzed records.

| Decision point | Native | Information | Repair | Information + repair |
|---|---:|---:|---:|---:|
| s005_t31: place pen | 0% | 0% | 17.5% | 97.5% |
| s006_t24: examine clock under lamp | 0% | 0% | 0% | 0% |
| s021_t15: cool wine bottle | 2.5% | 0% | 70% | 80% |
| s078_t36: heat potato | 0% | 0% | 0% | 5% |
| s010_t29: place statue | 32.5% | 80% | 32.5% | 97.5% |
| s073_t10: clean butterknife | 55% | 62.5% | 60% | 77.5% |

### 6.2 Removing a shared obstacle can make the current action matter more

The pen case s005_t31 provides the clearest interaction. All five actions fail in both native and information-only conditions. Repair alone produces `[0,.875,0,0,0]`: only the action that directly reveals the pen succeeds reliably. Combining repair with shared information produces `[.875,1,1,1,1]`.

Noise-corrected action variance changes from zero to .120 and then approximately zero. The average effect of information is zero without repair and 80 percentage points with repair. The shared obstacle does not remove the environment's information opportunity; it blocks its conversion into terminal success.

### 6.3 Repairing a branch-specific obstacle can close the remaining gap

The statue case shows the complementary direction. In the eight-repeat factorial controls, information alone produces `[1,1,1,0,1]`. Adding repair produces `[1,1,.875,1,1]`. The pencil-taking branch changes from 0/8 to 8/8 successes. Repair triggers only in that branch, directly filling the gap left after information completion.

Improved downstream execution therefore has no single direction of effect on action importance. Removing a shared bottleneck can reveal a previously hidden information fork; removing a branch-specific bottleneck can reduce an existing fork.

### 6.4 Acquiring the target is not completing the task

In the clock case, information plus repair raises actual pickup of the clue-specified target to 85%, while terminal success remains zero. After taking the clock, the model repeatedly examines the desk and object without completing the lamp task. In the potato case, target pickup reaches 90%, but success is only 5%, reflecting subsequent heating, placement, and budget constraints.

These traces distinguish receiving information, acting on it, satisfying a local requirement, and completing the task. Terminal reward observes the final outcome; later failures can hide the value of earlier progress.

**[Figure 4.]** Four-condition profiles for the pen and statue cases, with target-pickup versus task-success panels for the clock and potato cases.

## 7. From Mechanism Cases to Predictive Regularities [PLANNED EXPERIMENT]

### 7.1 Independent mechanism prediction

This experiment will test whether the proposed mechanism makes predictions before new continuation outcomes are observed. Using original prefixes, legal actions, and branch metadata, we will fix a set of independent points with shared execution obstacles and unequal information, branch-specific obstacles, and controls lacking the targeted obstacle. Candidate sets, repair rules, budgets, and readouts will be fixed before collection.

The primary predictions are conditional. When a shared obstacle limits information use, repair should allow branch-level information differences to translate more readily into return differences. When information is already shared, repairing a branch-specific obstacle should preferentially recover that branch. We will compare complete action-value vectors, mean success, action variance, repair activation, and actual target acquisition.

**Result placeholder:** [Insert independent task counts, directional agreement by mechanism group, interaction estimates and intervals, and all case profiles.]

**Interpretation placeholder:** [If predictions repeat independently, strengthen the claim to a reproducible conditional mechanism. If effects remain localized, retain the existence result and identify the execution conditions that delimit it.]

### 7.2 Cross-environment test

For WebShop, the transferable object is the relationship between action-created information and downstream completion ability, rather than the specific ALFWorld release operation. We will identify product-information differences and concrete subsequent selection or purchase obstacles from actual trajectories, then construct the corresponding information-by-execution intervention.

**Result placeholder:** [Insert operational definitions, sample size, action-value vectors, and intervention responses. Compare differences attributable to information structure, recoverability, and execution constraints.]

**[Figure 5, pending.]** Intervention responses across independent cases and environments.

## 8. Implications for Learning and OPSD [LEARNING EXPERIMENT PLANNED]

The observed credit map need not be a fixed map of teaching priorities. The value of an upstream action may become visible only after a downstream prerequisite skill improves. This motivates a focused learning question: can teaching a downstream execution skill expose useful preferences at upstream points that previously showed little terminal differentiation?

We plan a small skill-supervision or OPSD update targeting execution errors such as releasing irrelevant objects, without supplying target-location answers. Before and after training, we will force the same candidate actions at the same independent historical prefixes and continue with the respective policies. We will also evaluate full episodes without privileged clues or forced repair.

Readouts include skill execution success, changes in the original action-value vectors, upstream action distributions, and unseen-task success. Shared parameter updates can alter multiple behaviors, so these changes will be reported together rather than treating training as an automatically isolated skill intervention.

**Result placeholder:** [Insert training examples, update budget, controls, skill improvement, credit-structure changes, and end-to-end performance.]

The objective is to test whether learning can expose new credit. It does not assume the effectiveness of a complex teacher-calibration pipeline. An existing actual Hindsight Teacher probe did not show systematic harm from additional information; correctly aligned one-step evidence produced modest, concentrated gains. Teacher failure under additional knowledge is therefore not a premise of this paper.

## 9. Related Work and Discussion

### 9.1 Relation to credit estimation

We study action-conditioned long-term outcomes and their responses to interventions. Confidence weighting, within-group trajectory comparison, counterfactual replay, and value estimation offer different ways of obtaining credit evidence. Our question is why these sources specialize across identification problems and why changing information or execution conditions changes the credit structure being estimated.

**[Bibliographic completion.]** Add and verify CWGT, CARL, BiPACE, GiGPO, and Counterfactual Rollout Replay. Compare their supervision objects, action correspondence, continuation policies, and environments individually rather than assigning all methods the same assumption.

### 9.2 Where the contribution lies

Policy dependence is already part of the definition of action value, and the variance identities are standard. Our empirical contribution is to expose and intervene on a concrete chain: information opportunities must pass through execution conditions to become terminal credit. The distribution of obstacles across branches determines whether intervention reveals or reduces action differences.

This refines the original information-loss motivation. Some representations omit action structure, as illustrated by improvement opportunities at Masked Forks. Other opportunities have not yet manifested in the current policy's terminal rewards. The former calls for richer measurement and representation; the latter calls for understanding and changing the conditions of realization.

### 9.3 Scope and open questions

Current mechanism evidence comes from purposefully selected ALFWorld cases, with the most complete reveal-then-equalize sequence concentrated in one point. Independent prediction and cross-environment experiments will determine its scope. The current interventions do not explain every Indifferent point, and limited B3 power does not establish that inexpensive information cannot predict credit sign.

Execution repair changes physical state, one step of budget, and visible history. It is a concrete assisted continuation policy. Successful forced repair and a student learning the corresponding behavior are distinct claims; the planned learning experiment addresses the latter.

## 10. Conclusion

Long-horizon credit must be understood as well as estimated. An action can create an opportunity that the current policy cannot convert into reward. Starting from regime measurement and information-source specialization, our information-completion and execution-repair experiments show how downstream obstacles can hide, reveal, or preserve differences between current actions. This provides an experimental perspective on when one-step feedback is informative, when low measured credit can coexist with useful learning opportunities, and why credit structure may need to be remeasured as agents learn.

## Appendix: Evidence Sources and Draft Completion

- Phase 3: [RESULTS.md](../../RESULTS.md), [review](../../REVIEW_2026-10-27_phase3.md). Import final intervals from evaluation files; omit unverified B+C results.
- Discovery and Masked analysis: [research report](../../reports/deep_discovery_2026-10-27.md).
- Information interventions: [two-action report](../../reports/opportunity_completion_results_2026-10-28.md), [five-action report](../../reports/regime_mechanism_2026-10-28.md).
- Factorial experiment: [complete report](../../reports/execution_factorial_results_2026-10-29.md), [all action-cell results](../../exploratory_deep_discovery/execution_factorial_2026_10_29/results/cells.csv).
- Teacher probe: [actual-model results](../../reports/teacher_probe_results_2026-10-29.md).
- Remaining evidence: independent predictive replication (§7.1), cross-environment testing (§7.2), and learning validation (§8). The present abstract and contributions do not claim the unmeasured generalization or training gains.
- Positioning: a mechanism-driven empirical paper. OPSD is a motivated extension and reserved validation, not an implemented method contribution.
