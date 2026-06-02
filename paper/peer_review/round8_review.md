# Round 8 Peer Review -- SSM-WM: Lightweight State Space World Model for Humanoid Robot State Prediction

**Target Journal:** 控制理论与应用 (CTA)
**Date:** 2026-06-02
**Paper Version:** Round 7 Revision (current: ~821 lines)

---

## Summary of Changes Since Round 7

The authors addressed the following Round 7 feedback:

1. **[R3]** Added standard deviations to the T=512 row in the sequence length table (Table 4). The row now reads `0.95+/-0.04, 0.68+/-0.03, 12.1+/-0.6, 228.4+/-10.5`, consistent with all other rows.
2. **[Devil + EIC]** Softened validation claims in Chinese abstract from "验证了" to "初步验证了" (preliminarily validates).
3. **[EIC + R2]** Added explicit limitation note in conclusion: "需要指出的是, 当前实验仅在合成数据上进行, 与真实人形机器人动力学存在一定差距, 未来需要在MuJoCo等物理仿真器或真机上进一步验证."
4. **[Devil]** English abstract also softened to "preliminarily validating its effectiveness and practicality for embodied intelligence."

**Not addressed (carried from Round 7):** MuJoCo experiments, analytical model MPC baseline, robustness evaluation, grid search table, paired t-tests, exposure bias quantification, training loss curves, error accumulation analysis.

---

## Reviewer 1: Editor-in-Chief (EIC) -- Overall Quality & Scope Fit

### Dimension Scores

| Dimension | Score | Change | Rationale |
|---|---|---|---|
| Originality | 63 | 0 | No new originality-relevant content in this revision. |
| Methodological Rigor | 72 | 0 | No methodological changes. |
| Evidence Sufficiency | 61 | +1 | T=512 std fixes a minor inconsistency. No new evidence added. |
| Argument Coherence | 78 | +1 | The softened claims ("初步验证") and explicit limitation note in the conclusion improve the honesty-to-evidence alignment. The paper no longer overclaims on synthetic data. |
| Writing Quality | 80 | +1 | The conclusion now reads more professionally with the explicit limitation paragraph. The Chinese and English abstracts are consistent. |

### Comments

The Round 7 revisions are small but targeted. The T=512 std fix was overdue and now makes all tables consistent. The softened language in the abstract and conclusion is the right move -- it signals intellectual honesty and sets appropriate expectations for the reader.

However, I want to be precise about what has changed and what has not. The paper's technical content is identical to Round 7. The improvements are purely in framing and completeness of reporting. This is valuable but does not address the fundamental evidence gap.

The conclusion now explicitly states the synthetic data limitation and lists MuJoCo/real-robot validation as future work item (2). This is honest, but it also confirms that the current paper is a proof-of-concept. For CTA, this remains a significant concern.

### Remaining Issues

1. **[Critical -- Unchanged]** Synthetic data only. The limitation is now acknowledged, which is good, but acknowledging a limitation is not the same as addressing it.
2. **[Important]** The grid search over lambda and H should still be tabulated. A single prose sentence is insufficient for reproducibility.
3. **[Desirable]** A brief note on the B=1 speedup's practical significance would add nuance (1.2ms absolute difference is small).

---

## Reviewer 2: R1 (Methodology Expert) -- Technical Rigor & Novelty

### Dimension Scores

| Dimension | Score | Change | Rationale |
|---|---|---|---|
| Originality | 59 | 0 | No change. |
| Methodological Rigor | 72 | 0 | No new methodological content. |
| Evidence Sufficiency | 65 | 0 | No new evidence. |
| Argument Coherence | 74 | 0 | Minor improvement in claim-evidence alignment from softened language, but offset by no new analysis. |
| Writing Quality | 78 | +1 | Conclusion is cleaner with the explicit limitation note. |

### Comments

I have little to add beyond my Round 7 assessment. The T=512 std fix and softened claims are cosmetic improvements that I welcome but do not consider substantive.

The two items I flagged in Round 7 remain unaddressed:

1. **Grid search table.** The authors describe a grid search over lambda in {0, 0.1, 0.5, 1.0} and H in {4, 8, 16} but only report the chosen values (lambda=0.5, H=8). A 4x3 table showing validation MSE for each combination would take 10 lines of LaTeX and would make this ablation fully reproducible. This is a low-effort, high-value fix.

2. **Error accumulation analysis.** The multi-step table (Table 4) shows that SSM-WM's gap vs. LSTM narrows from 55% at H=1 to 41% at H=16. This is a genuine and interesting finding. Why does SSM-WM accumulate less error? Hypothesis: the linear recurrence structure of SSM provides more stable long-horizon propagation than LSTM's nonlinear gates. Testing this hypothesis (e.g., by comparing state transition Jacobians) would strengthen the paper significantly.

### Remaining Issues

1. **[Important]** Tabulate the lambda/H grid search.
2. **[Desirable]** Analyze why SSM-WM has lower error accumulation than LSTM.
3. **[Desirable]** Add training loss curves to verify convergence at 20 epochs.

---

## Reviewer 3: R2 (Domain Expert -- Embodied AI / Robotics) -- Relevance & Practical Significance

### Dimension Scores

| Dimension | Score | Change | Rationale |
|---|---|---|---|
| Originality | 61 | 0 | No change. |
| Methodological Rigor | 66 | +1 | The explicit limitation note in the conclusion shows awareness of the domain gap. |
| Evidence Sufficiency | 54 | 0 | No new evidence. The MuJoCo gap remains. |
| Argument Coherence | 72 | +1 | The softened claims are appropriate for the evidence base. The paper now correctly positions itself as a proof-of-concept rather than a validated system. |
| Writing Quality | 76 | 0 | No significant change. |

### Comments

The explicit limitation in the conclusion is a welcome addition. The authors now state clearly: "当前实验仅在合成数据上进行, 与真实人形机器人动力学存在一定差距." This is the right framing.

However, I want to reiterate why this matters for the CTA audience. The synthetic dynamics used in this paper (linear A*s + B*a + 0.1*tanh coupling + Gaussian noise) are smooth and Lipschitz-continuous. Real humanoid dynamics involve:
- Contact switching (foot-ground impact creates discontinuities in the dynamics)
- Friction cones (non-smooth, inequality-constrained)
- Floating-base kinematics (underactuated, configuration-dependent inertia)
- Gear backlash and actuator saturation

The SSM's linear recurrence assumption works well on smooth dynamics but has no theoretical guarantee on non-smooth systems. The paper's claim that SSM-WM achieves R^2=0.945 is meaningful only within the synthetic regime. On MuJoCo Humanoid-v4, I would expect a significant accuracy drop, and this drop would inform whether the approach is viable for real deployment.

The conclusion's future work item (2) -- "在MuJoCo等物理仿真器和真机人形机器人上进行实验验证" -- confirms that the authors recognize this need. I encourage them to prioritize this.

### Remaining Issues

1. **[Critical -- Unchanged]** MuJoCo experiments remain the single most important gap.
2. **[Important]** No comparison with analytical MPC baselines (rigid-body dynamics + LQR/iLQR).
3. **[Desirable]** Robustness evaluation under dynamics mismatch.

---

## Reviewer 4: R3 (Broader Impact & Completeness) -- Evaluation Comprehensiveness

### Dimension Scores

| Dimension | Score | Change | Rationale |
|---|---|---|---|
| Originality | 65 | 0 | No change. |
| Methodological Rigor | 72 | 0 | No change. |
| Evidence Sufficiency | 65 | +1 | T=512 std fix completes the statistical reporting. All tables now have consistent mean +/- std format. |
| Argument Coherence | 74 | 0 | No change. |
| Writing Quality | 79 | 0 | No change. |

### Comments

The T=512 std fix resolves my last outstanding complaint about inconsistent statistical reporting. All four cells in that row now have standard deviations, matching the format of every other row in the table. This is a minor but necessary fix, and I am satisfied.

With this fix, the paper's statistical reporting is now complete:
- Table 1 (main results): mean +/- std across 3 seeds. Complete.
- Table 2 (inference time): mean +/- std. Complete.
- Table 3 (batch sizes): mean +/- std. Complete.
- Table 4 (multi-step): mean +/- std. Complete.
- Table 4 (sequence lengths): mean +/- std. Now complete including T=512.
- Table 5 (ablation): mean +/- std. Complete.
- Table 6 (MPC): mean +/- std. Complete.

I still note the absence of paired t-tests for the MPC comparison. The SSM-WM-MPC tracking MSE (0.0043 +/- 0.0004) vs. LSTM-MPC (0.0032 +/- 0.0003) -- is this difference statistically significant? With 3 seeds and the reported standard deviations, a two-sample t-test gives t ~ 3.67, df ~ 4, p ~ 0.02. So yes, the difference is likely significant at p < 0.05, meaning SSM-WM-MPC does sacrifice some tracking accuracy. Stating this explicitly would strengthen the paper.

### Remaining Issues

1. **[Important]** Add paired t-tests or a brief statistical significance note for the MPC comparison.
2. **[Desirable]** Cross-dataset generalization (test on held-out dynamics distributions).

---

## Reviewer 5: Devil's Advocate -- Attack the Claims

### Dimension Scores

| Dimension | Score | Change | Rationale |
|---|---|---|---|
| Originality | 46 | 0 | No change. |
| Methodological Rigor | 60 | 0 | No change. |
| Evidence Sufficiency | 52 | 0 | No new evidence. |
| Argument Coherence | 66 | +2 | The softened claims are a meaningful improvement. The paper now says "初步验证" instead of "验证", which is honest. The explicit limitation note in the conclusion is exactly what I demanded in Round 7. |
| Writing Quality | 73 | 0 | No change. |

### Comments

I acknowledge that the authors addressed my Round 7 demand directly. I said: "The paper should be explicitly framed as a proof-of-concept. Soften to '初步验证'." The authors did exactly this -- in both the Chinese abstract ("初步验证了SSM-WM在具身智能场景下的有效性和实用性") and the English abstract ("preliminarily validating its effectiveness and practicality for embodied intelligence"). They also added an explicit limitation paragraph in the conclusion.

This is the right thing to do, and it improves the paper's integrity. The reader now understands that the results are preliminary and that synthetic-to-real transfer is an open question.

That said, let me be clear about what this means for the paper's contribution level:

1. The paper proposes a reasonable architecture (S4D diagonal SSM + Mamba-style gating for world models). This is an incremental but valid combination.
2. The paper demonstrates that this architecture is faster than LSTM and comparable to Mamba on a synthetic task. This is expected given the known computational complexity advantages of SSMs.
3. The paper shows a 7x speedup in MPC control loop time. This is the strongest result, but it is inflated by the fact that the MPC uses 50 Adam iterations per step, and the model inference is a small fraction of the total loop time. The LSTM-MPC takes 1420ms total; if we subtract the 50 Adam iterations' overhead, the actual model inference contribution to the speedup is unclear.
4. The 55% accuracy gap on synthetic data is concerning. On real dynamics, this gap would likely widen.

The paper is now honest about these limitations. But honesty about limitations does not eliminate them.

### Remaining Issues

1. **[Unchanged]** The B=1 speedup (0.9ms vs 2.1ms, 1.2ms absolute) is negligible. The paper should note that the speedup is primarily meaningful for batched inference.
2. **[Unchanged]** Exposure bias is discussed but not quantified. A simple experiment comparing teacher-forced vs. autoregressive MPC would measure the actual impact.
3. **[New]** The MPC loop time breakdown is unclear. How much of the 195ms (SSM-WM-MPC) vs. 1420ms (LSTM-MPC) is model inference vs. Adam optimization? A breakdown would clarify whether the 7x speedup is due to the model or the overall pipeline.

---

## Summary Score Matrix

| Reviewer | Originality | Rigor | Evidence | Coherence | Writing | Weighted Avg |
|---|---|---|---|---|---|---|
| EIC | 63 | 72 | 61 | 78 | 80 | 70.8 |
| R1 (Method) | 59 | 72 | 65 | 74 | 78 | 69.6 |
| R2 (Domain) | 61 | 66 | 54 | 72 | 76 | 65.8 |
| R3 (Breadth) | 65 | 72 | 65 | 74 | 79 | 71.0 |
| Devil's Advocate | 46 | 60 | 52 | 66 | 73 | 59.4 |
| **Mean** | **58.8** | **68.4** | **59.4** | **72.8** | **77.2** | **67.3** |

**Round 7 Mean:** 67.0 --> **Round 8 Mean:** 67.3 (+0.3)

### Progress Assessment

The Round 7 -> Round 8 changes are small but targeted:

- T=512 std: DONE (addresses R3)
- Softened claims to "初步验证": DONE (addresses Devil)
- Explicit limitation in conclusion: DONE (addresses EIC + R2)
- English abstract consistency: DONE (addresses Devil)

The score improvement is minimal (+0.3) because the changes are cosmetic rather than substantive. The paper's technical content and evidence base are identical to Round 7. The improvements are in framing, which matters for integrity but does not change the scientific contribution.

### Overall Assessment

**Score: 67.3/100 -- Weak Accept with Major Reservation**

The paper has reached a plateau. Over 8 rounds of review, the authors have:
- Added multi-step prediction analysis (Round 6-7)
- Added exposure bias discussion (Round 7)
- Fixed all statistical reporting inconsistencies (Round 6-8)
- Softened overclaims (Round 8)
- Explicitly acknowledged the synthetic data limitation (Round 8)

The paper is now technically sound, well-written, and honest about its limitations. The remaining issues are all related to the evidence base (synthetic data only) and a few desirable-but-not-blocking analysis additions.

### Recommendation

The paper is at the boundary of CTA's acceptance threshold. The methodology is solid, the writing is clear, and the contribution (lightweight SSM world model with 7x speedup) is relevant to the control theory community. However, the synthetic-only evidence limits the practical significance.

**Conditional Accept** if the authors can add even a single MuJoCo experiment (e.g., Humanoid-v4 locomotion tracking). This would push the evidence score above 65 for all reviewers and resolve the primary blocking concern.

**Weak Accept** if MuJoCo is not feasible, given that the paper now honestly frames itself as a proof-of-concept and the methodology is complete within its scope.

### Remaining Fixes Ranked by Priority

1. **[Critical -- Blocking]** Add MuJoCo humanoid experiments. The sole remaining critical item across all 8 rounds.
2. **[Important]** Tabulate the lambda/H grid search (12-cell table, ~10 lines of LaTeX).
3. **[Important]** Add paired t-tests for MPC comparison.
4. **[Desirable]** Break down MPC loop time (model inference vs. Adam optimization).
5. **[Desirable]** Quantify exposure bias impact.
6. **[Desirable]** Analyze SSM-WM's lower error accumulation mechanism.
7. **[Desirable]** Add training loss curves.
