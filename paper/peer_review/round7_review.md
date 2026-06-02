# Round 7 Peer Review -- SSM-WM: Lightweight State Space World Model for Humanoid Robot State Prediction

**Target Journal:** 控制理论与应用 (CTA)
**Date:** 2026-06-02
**Paper Version:** Round 6 Revision (current: ~819 lines)

---

## Summary of Changes Since Round 6

The authors addressed the following Round 6 feedback:

1. **[R1 + Devil]** Added multi-step prediction error table (new Table 4) with H=1,4,8,16.
2. **[R3]** Added standard deviations to sequence length table.
3. **[R3]** Added error bars (std) to MPC table (now Table 7).
4. **[R1]** Added exposure bias discussion with citation [39] (Bengio et al., Scheduled Sampling).
5. **[R1]** Added loss component ablation discussion (lambda in {0, 0.1, 0.5, 1.0}, H in {4, 8, 16}) based on validation set grid search.
6. **[EIC]** Added SSM applications in robotics paragraph (Mamba policy [40], SSM trajectory prediction [41]).
7. **[R1]** Explained LSTM-4L degradation (overfitting on small dataset).
8. Added references [39]-[41].

**Not addressed:** MuJoCo experiments, analytical model MPC baseline, robustness evaluation, B=1 emphasis in abstract.

---

## Reviewer 1: Editor-in-Chief (EIC) -- Overall Quality & Scope Fit

### Dimension Scores

| Dimension | Score | Change | Rationale |
|---|---|---|---|
| Originality | 63 | +1 | The new SSM-robotics related work paragraph [40,41] slightly improves contextualization, showing the authors are aware of adjacent work. Still incremental. |
| Methodological Rigor | 72 | +4 | Exposure bias discussion with formal citation, loss component grid search, and LSTM-4L overfitting explanation all strengthen the methodology section. |
| Evidence Sufficiency | 60 | +5 | Multi-step prediction table (Table 4) is a significant addition -- it directly addresses what MPC actually uses. Std in Tables 4 and 6 improve credibility. |
| Argument Coherence | 77 | +2 | The exposure bias discussion adds nuance. The multi-step error analysis closes a logical gap (single-step vs. MPC-relevant metrics). |
| Writing Quality | 79 | +1 | Minor improvements. New content integrates smoothly. |

### Comments

The Round 6 revisions are directionally correct and address several specific requests. The multi-step prediction table (Table 4) is the most valuable addition, as it provides the metric that actually matters for MPC deployment. The exposure bias discussion shows intellectual honesty.

However, the fundamental limitation persists: all experiments remain on synthetic data. This was flagged as [Critical] in Round 6 and has not been addressed. The new SSM-robotics paragraph [40,41] improves related work coverage but does not substitute for empirical validation on physically meaningful systems.

The loss component ablation is described narratively (grid search results mentioned in text) but not presented as a table. For a journal paper, a small table showing the grid search results would be more convincing than a prose description.

### Remaining Issues

1. **Synthetic data remains the critical barrier.** This was the #1 priority fix in Round 6. Without MuJoCo or real-robot experiments, the paper is not suitable for CTA in its current form. The contribution is a proof-of-concept at best.
2. **Grid search results should be tabulated.** The lambda/H grid search is mentioned in the training objective section but only as prose. A small table (lambda x H -> MSE) would make this ablation reproducible and convincing.
3. **The 55% accuracy gap needs formal error propagation analysis.** How does MSE=1.32 vs 0.85 translate to MPC tracking cost? The multi-step table helps, but a formal sensitivity analysis is still missing.

---

## Reviewer 2: R1 (Methodology Expert) -- Technical Rigor & Novelty

### Dimension Scores

| Dimension | Score | Change | Rationale |
|---|---|---|---|
| Originality | 59 | +1 | Minimal change. The architecture remains a known-combination applied to a new domain. |
| Methodological Rigor | 72 | +7 | Significant improvement. Exposure bias discussion with [39], loss component grid search, and LSTM-4L explanation all address prior concerns. The grid search methodology (lambda in {0,0.1,0.5,1.0}, H in {4,8,16}) is properly described. |
| Evidence Sufficiency | 65 | +5 | Multi-step prediction table is essential and was missing. The H=1,4,8,16 breakdown directly shows error accumulation, which is what MPC relies on. |
| Argument Coherence | 74 | +2 | The exposure bias discussion (teacher forcing vs. autoregressive deployment) is well-articulated and properly cited. The mitigation strategy (joint single+multi-step loss) is reasonable. |
| Writing Quality | 77 | +1 | Consistent quality. |

### Comments

I am satisfied with the methodological improvements. Specifically:

**Exposure bias:** The discussion in Section 4.2 correctly identifies the training-inference mismatch and cites Bengio et al. [39] on scheduled sampling. The mitigation via joint loss is a reasonable first-order approach, though scheduled sampling would be more principled. This is now at an acceptable level for a journal paper.

**Loss component ablation:** The grid search over lambda and H is mentioned: "实验中lambda和H的选择基于验证集上的网格搜索(lambda in {0, 0.1, 0.5, 1.0}, H in {4, 8, 16})." This is good methodology, but the results should be presented in a table. The reader cannot verify the claim that (0.5, 8) is optimal without seeing the full grid.

**LSTM-4L degradation:** The explanation (overfitting on small dataset) is plausible and now explicitly stated in Table 1 discussion. This is reasonable -- 100 episodes is indeed small for a 4-layer LSTM with 0.38M parameters.

**Multi-step prediction:** Table 4 with H=1,4,8,16 is exactly what was requested. The observation that SSM-WM's multi-step gap (38-47%) is smaller than the single-step gap (55%) is interesting and suggests better error accumulation properties. This deserves more analysis -- why does SSM-WM accumulate less error?

### Remaining Issues

1. **Tabulate the grid search.** A small table (4 lambdas x 3 horizons = 12 cells) would make the ablation reproducible and convincing. Currently it is a single sentence.
2. **Analyze why SSM-WM has lower error accumulation.** The multi-step table shows SSM-WM's gap vs. LSTM narrows from 55% (H=1) to 41% (H=16). This is a genuine finding that deserves mechanistic explanation. Is it the SSM's linear recurrence structure? The gating mechanism? This could be a contribution if analyzed.
3. **Still no learning curves.** 20 epochs on 100 episodes -- have the models converged? A training loss curve would resolve this trivially.

---

## Reviewer 3: R2 (Domain Expert -- Embodied AI / Robotics) -- Relevance & Practical Significance

### Dimension Scores

| Dimension | Score | Change | Rationale |
|---|---|---|---|
| Originality | 61 | +1 | The SSM-robotics paragraph [40,41] shows awareness of the field. Still, the core contribution (SSM for state prediction) has not been validated on a real robotics problem. |
| Methodological Rigor | 65 | +2 | Multi-step prediction and exposure bias discussion improve rigor. MPC experiment remains too simple. |
| Evidence Sufficiency | 54 | +4 | Table 4 (multi-step) is valuable. Error bars on MPC table help. But the fundamental gap -- no physical simulation, no real robot -- remains. |
| Argument Coherence | 71 | +1 | The new SSM-robotics paragraph strengthens the motivation. The exposure bias discussion shows awareness of practical deployment issues. |
| Writing Quality | 76 | +1 | No significant change. |

### Comments

The multi-step prediction table (Table 4) is a valuable addition for the robotics audience. In MPC, the controller queries the model for H steps ahead, so multi-step accuracy is the directly relevant metric. The H=8 row (MSE=3.48e-3, corresponding to ~0.6 degrees per joint) provides a concrete, interpretable result.

The new SSM-robotics paragraph correctly cites Mamba policy [40] and SSM trajectory prediction [41], showing that SSM is gaining traction in robotics. This contextualizes the paper's contribution.

However, my core concern from Round 6 stands: the synthetic dynamics (linear + tanh coupling + Gaussian noise) are not representative of real humanoid control challenges. The dynamics lack:
- Contact discontinuities (foot-ground interaction)
- Friction models
- Underactuation (floating base)
- High-dimensional configuration space with kinematic constraints

The multi-step error analysis actually makes this concern more concrete: at H=16, SSM-WM's MSE is 5.82e-3. On real humanoid dynamics with contact switching, this would likely be much worse due to the non-smooth dynamics. The synthetic data smoothness is what makes the SSM's linear recurrence work well -- real dynamics would stress-test this assumption.

### Remaining Issues

1. **MuJoCo experiments remain essential.** Even a simple Humanoid-v4 locomotion task would demonstrate that the approach handles non-smooth dynamics. This was [Critical] in Round 6 and remains so.
2. **No comparison with analytical MPC.** The standard in robotics is model-based MPC with rigid-body dynamics. The paper should show when/why learned models are preferable.
3. **Robustness evaluation is missing.** The paper should test prediction accuracy under varying noise levels and model mismatch.

---

## Reviewer 4: R3 (Broader Impact & Completeness) -- Evaluation Comprehensiveness

### Dimension Scores

| Dimension | Score | Change | Rationale |
|---|---|---|---|
| Originality | 65 | 0 | No change in originality assessment. |
| Methodological Rigor | 72 | +2 | Grid search methodology, exposure bias discussion improve rigor. |
| Evidence Sufficiency | 64 | +6 | Multi-step table, std in sequence table, error bars on MPC table -- all requested items addressed. |
| Argument Coherence | 74 | +1 | Consistent improvement. |
| Writing Quality | 79 | +1 | Tables are now more complete and consistent. |

### Comments

I am largely satisfied with the statistical reporting improvements. Specifically:

**Standard deviations:** Now present in the sequence length table (Table 4 in the paper's numbering, which corresponds to the multi-step prediction table). However, I note that the T=512 row in the sequence length table still lacks standard deviations -- it reports single values (SSM-WM MSE=0.95, LSTM-WM MSE=0.68, SSM-WM time=12.1, LSTM-WM time=228.4) while all other rows have mean +/- std. This is inconsistent.

**MPC error bars:** Table 7 (MPC performance) now includes standard deviations. The SSM-WM-MPC tracking MSE is 0.0043 +/- 0.0004, which shows reasonable consistency across runs.

**Multi-step prediction:** Table 4 is a strong addition. The H=1,4,8,16 breakdown with std is exactly what was requested.

### Remaining Issues

1. **T=512 row still lacks std.** This is a minor but unnecessary inconsistency. Either add std or explain why it is missing (e.g., "single run due to computational cost").
2. **No statistical significance testing.** The MPC table has error bars but no paired t-tests. Is SSM-WM-MPC's tracking MSE (0.0043 +/- 0.0004) significantly different from LSTM-MPC (0.0032 +/- 0.0003)? With the reported std, a rough calculation suggests p < 0.05, but this should be stated explicitly.
3. **Still no cross-dataset generalization.** All experiments use the same synthetic dataset. Testing on a held-out dynamics distribution would demonstrate robustness.

---

## Reviewer 5: Devil's Advocate -- Attack the Claims

### Dimension Scores

| Dimension | Score | Change | Rationale |
|---|---|---|---|
| Originality | 46 | +1 | Minimal change. |
| Methodological Rigor | 60 | +5 | Exposure bias discussion, grid search, and LSTM-4L explanation address prior methodological concerns. |
| Evidence Sufficiency | 52 | +7 | Multi-step table is a real improvement. It shows the authors are willing to report metrics that matter, not just flattering ones. |
| Argument Coherence | 64 | +4 | The exposure bias discussion and multi-step analysis close logical gaps. However, the core claims remain overstated for synthetic-only evidence. |
| Writing Quality | 73 | +1 | Consistent. |

### Comments

I acknowledge genuine improvement. The multi-step prediction table (Table 4) was the most important missing piece, and the authors delivered it. The exposure bias discussion shows intellectual maturity. The LSTM-4L explanation is plausible.

However, I must be blunt: **the paper's core weakness has not been addressed, and it is disqualifying for a control theory journal.**

Let me restate the argument clearly:
- The paper claims to validate SSM-WM for "humanoid robot state prediction" and "model predictive control."
- All experiments use synthetic dynamics: s_{t+1} = A*s_t + B*a_t + 0.1*tanh(s_t * a_t) + noise.
- This dynamics model is linear + small nonlinear perturbation. Any reasonable neural network can learn it.
- The fact that SSM-WM achieves R^2=0.945 on this toy problem tells us nothing about real-world performance.
- The 55% accuracy gap (MSE 1.32 vs 0.85) on trivial dynamics would likely be much larger on real humanoid dynamics with contacts, friction, and underactuation.

The multi-step table actually makes this argument stronger, not weaker. At H=16, SSM-WM's MSE is 5.82e-3 on linear dynamics. On real humanoid dynamics with contact switching, this would likely blow up. The paper has no evidence to counter this.

The new exposure bias discussion is commendable but incomplete. The authors acknowledge the problem and propose joint loss as mitigation, but do not evaluate the actual impact. How much does exposure bias degrade MPC performance? A simple experiment: compare MPC with teacher-forced multi-step predictions vs. autoregressive predictions. This would quantify the problem.

### Remaining Issues

1. **The paper should be explicitly framed as a proof-of-concept.** The abstract and conclusions claim "验证了SSM-WM在具身智能场景下的有效性和实用性" (validates effectiveness and practicality). This is too strong for synthetic-only experiments. Soften to "初步验证" (preliminarily validates) or add MuJoCo experiments.
2. **Quantify exposure bias impact.** The discussion identifies the problem but does not measure it. Compare MPC performance with teacher-forced vs. autoregressive rollouts.
3. **The B=1 speedup (0.9ms vs 2.1ms) is 1.2ms absolute.** This is negligible for most control loops. The paper should be honest that the speedup is most meaningful for batched inference (e.g., parallel MPC sampling), not single-robot deployment.

---

## Summary Score Matrix

| Reviewer | Originality | Rigor | Evidence | Coherence | Writing | Weighted Avg |
|---|---|---|---|---|---|---|
| EIC | 63 | 72 | 60 | 77 | 79 | 70.2 |
| R1 (Method) | 59 | 72 | 65 | 74 | 77 | 69.4 |
| R2 (Domain) | 61 | 65 | 54 | 71 | 76 | 65.4 |
| R3 (Breadth) | 65 | 72 | 64 | 74 | 79 | 70.8 |
| Devil's Advocate | 46 | 60 | 52 | 64 | 73 | 59.0 |
| **Mean** | **58.8** | **68.2** | **59.0** | **72.0** | **76.8** | **67.0** |

**Round 6 Mean:** 64.3 --> **Round 7 Mean:** 67.0 (+2.7)

### Progress Assessment

The authors have made meaningful progress on the "desirable" and "important" fixes from Round 6:
- Multi-step prediction table: DONE (addresses R1 + Devil)
- Std in sequence length table: MOSTLY DONE (T=512 row missing std)
- Error bars on MPC table: DONE (addresses R3)
- Exposure bias discussion: DONE (addresses R1)
- Loss component ablation: PARTIALLY DONE (described but not tabulated)
- LSTM-4L explanation: DONE (addresses R1)
- SSM-robotics related work: DONE (addresses EIC)

The only unaddressed priority is the [Critical] item: **MuJoCo or real-robot experiments.** This remains the single most important barrier to acceptance.

### Overall Assessment

**Score: 67.0/100 -- Weak Accept with Major Reservation**

The paper has improved from 64.3 to 67.0 through targeted revisions. The multi-step prediction table, exposure bias discussion, and loss component grid search are all valuable additions that strengthen the methodology and evaluation. The paper is now technically sound and well-evaluated within its experimental scope.

However, the scope itself is the problem. A control theory journal (CTA) expects validation on physically meaningful systems. The synthetic dynamics used in this paper are too simple to support the claims made. The paper is suitable as a workshop paper or a proof-of-concept contribution, but not as a full CTA journal article without physical simulation experiments.

### Recommendation

**Conditional Accept** if the authors can add MuJoCo humanoid experiments (even 1-2 locomotion tasks with Humanoid-v4 or similar). This single change would push the evidence score from 59 to ~70 and resolve the primary concern shared by 3 of 5 reviewers.

**Reject with encouragement to resubmit** if MuJoCo experiments are not feasible within the revision timeline. The paper has solid methodology and would benefit from resubmission with real simulation data.

### Remaining Fixes Ranked by Priority

1. **[Critical -- Blocking]** Add MuJoCo humanoid experiments. This is the sole remaining critical item.
2. **[Important]** Tabulate the lambda/H grid search results (12-cell table).
3. **[Important]** Add std to T=512 row in sequence length table.
4. **[Important]** Add paired t-tests for MPC results.
5. **[Desirable]** Quantify exposure bias impact (teacher-forced vs. autoregressive MPC).
6. **[Desirable]** Analyze why SSM-WM has lower error accumulation (gap narrows from 55% to 41% as H increases).
7. **[Desirable]** Add training loss curves (convergence verification).
