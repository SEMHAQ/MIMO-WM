# Round 10 Peer Review -- CTA (控制理论与应用)

**Paper Title:** 面向人形机器人状态预测的轻量级状态空间世界模型
**Manuscript:** /mnt/e/Project/SSM-World-Model/paper/main.tex (~861 lines)

---

## Scoring Summary

| Dimension | EIC | R1-Methodology | R2-Domain | R3-Perspective | Devil's Advocate |
|---|---|---|---|---|---|
| Originality | 69 | 66 | 71 | 73 | 59 |
| Methodological Rigor | 65 | 61 | 67 | 70 | 50 |
| Evidence Sufficiency | 62 | 56 | 65 | 68 | 46 |
| Argument Coherence | 72 | 68 | 73 | 75 | 57 |
| Writing Quality | 76 | 74 | 77 | 79 | 68 |
| **Overall** | **69** | **65** | **71** | **73** | **56** |

**Round-over-Round Delta:**

| Dimension | R9 Mean | R10 Mean | Delta |
|---|---|---|---|
| Originality | 66.6 | 67.6 | +1.0 |
| Methodological Rigor | 60.2 | 62.6 | +2.4 |
| Evidence Sufficiency | 55.8 | 59.4 | +3.6 |
| Argument Coherence | 67.4 | 69.0 | +1.6 |
| Writing Quality | 70.8 | 74.8 | +4.0 |
| **Grand Mean** | **64.2** | **66.7** | **+2.5** |

---

## Reviewer: EIC (Editor-in-Chief)

### Summary
This paper proposes SSM-WM, a lightweight world model based on diagonal state space models (S4D) combined with Mamba-style gated blocks, applied to humanoid robot state prediction and integrated into an MPC framework. Now in its 10th round, the manuscript has addressed several Round 9 blocking issues: duplicate table labels are fixed (Tables 1-8 are now sequential), the reference [22] copy-paste error is corrected, the state dimensionality claim is updated, and the MuJoCo inference time trade-off is honestly discussed. The topic remains timely and the lightweight inference speed is noteworthy for the CTA readership.

### Originality (69/100)
The combination of S4D diagonal SSM with Mamba-style gated blocks for robot world models is a reasonable contribution, though the individual components are well-established. The novelty lies primarily in the application context rather than architectural innovation. The MPC integration adds practical value. The honest acknowledgment that SSM-WM is slower than LSTM on MuJoCo (9.5ms vs 5.0ms) actually strengthens the intellectual contribution -- it shows the authors understand the nuanced performance landscape rather than cherry-picking favorable results.

### Methodological Rigor (65/100)
**Positive developments since Round 9:**
- Table labels are now sequential (Tables 1-8), resolving the duplicate label issue.
- Reference [22] is corrected from "GU A, GU A, GU A" to "GU A, GU A, RE C".
- State dimensionality claim updated from "20-50" to "数十至数百", which is consistent with the MuJoCo 376-dimensional state.
- MuJoCo results now use correct experimental numbers (SSM: MSE=0.834, R^2=0.592, Infer=9.5ms; LSTM: MSE=0.889, R^2=0.566, Infer=5.0ms).

**Remaining issues:**
1. **MuJoCo inference time standard deviation:** 9.5+/-6.5ms has an extremely high coefficient of variation (~68%). This suggests measurement instability, possibly due to FFT computation variability or GPU thermal throttling. The authors should investigate and either explain or reduce this variance. A CV this high undermines confidence in the 9.5ms figure.
2. **Missing MuJoCo MPC experiment:** The paper's central claim -- SSM enables real-time MPC for humanoid control -- is validated only on synthetic data. MuJoCo MPC remains untested.
3. **Inconsistency between abstract and MuJoCo results:** The abstract claims "约2倍提速(B=1, 0.9ms vs 2.1ms)" based on synthetic data, but MuJoCo shows SSM is actually slower (9.5ms vs 5.0ms). The abstract should mention this dataset-dependent behavior or qualify the speedup claim.
4. **Statistical significance on MuJoCo:** The MSE difference (0.834 vs 0.889) and R^2 difference (0.592 vs 0.566) remain untested for significance. With only 3 seeds, these differences may not be statistically significant.

### Evidence Sufficiency (62/100)
The addition of honest MuJoCo inference time discussion is welcome, but the evidence base remains incomplete:
- MPC experiments are only on synthetic data, not MuJoCo.
- MuJoCo comparison includes only LSTM-WM; Mamba-WM and Transformer-WM are absent.
- The R^2 of 0.592 on MuJoCo means 41% of variance is unexplained, raising questions about practical utility.
- No linear regression baseline to establish a lower bound on achievable performance.

### Argument Coherence (72/100)
The paper's narrative arc is logically structured: lightweight architecture -> fast inference -> MPC integration. The discussion section now honestly addresses the MuJoCo inference time trade-off, which strengthens the argument's credibility. However, the central argument remains bifurcated: speed results come from synthetic data while accuracy results come from MuJoCo, and these two datasets show opposite SSM-vs-LSTM orderings. A unified experiment (SSM-MPC on MuJoCo) would close this gap.

### Writing Quality (76/100)
The formatting improvements are noticeable. Tables 1-8 are correctly labeled and sequential. Reference [22] is fixed. The state dimensionality claim is corrected. The limitations section is honest and well-articulated. The abstract remains concise and informative. The English abstract is fluent.

**Remaining minor issues:**
- The abstract's speedup claims should be qualified to acknowledge dataset-dependent behavior.
- Several references still lack complete venue information (e.g., bibitem{40}, bibitem{41}).

### Recommendation: Accept with Minor Revisions
The Round 10 fixes address the most egregious formatting and factual errors. The remaining issues (MuJoCo MPC, MuJoCo baselines, significance tests) are important but not blocking for a journal that values practical contributions. The paper is approaching publishable quality.

---

## Reviewer: R1 (Methodology Expert)

### Summary
I focus on the methodological soundness of the proposed approach and experimental design. Round 10 has fixed the most visible errors (table labels, references, dimensionality claim) but the deeper methodological gaps remain.

### Originality (66/100)
The architecture is a straightforward composition of existing components. S4D parameterization and Mamba-style gating are individually well-known; the contribution is their assembly for robot state prediction. This is acceptable for an application-oriented journal but limits methodological novelty. The honest reporting of MuJoCo inference times actually helps -- it shows intellectual honesty rather than inflated claims.

### Methodological Rigor (61/100)
**Improvements since Round 9:**
- Correct MuJoCo experimental numbers replace previously incorrect values.
- The discussion section now includes an honest paragraph about the MuJoCo inference time trade-off.
- The limitations section explicitly mentions the 9.5ms vs 5.0ms inversion.

**Remaining methodological concerns:**

1. **MuJoCo inference variance (9.5+/-6.5ms):** This standard deviation is 68% of the mean. Possible explanations: (a) FFT computation has variable latency depending on sequence length padding to power-of-2; (b) GPU thermal throttling during measurement; (c) different MuJoCo episode lengths cause variable computation. The authors should identify and control for the source of variance. If the true inference time is, say, 5ms +/- 1ms vs 9.5ms +/- 6.5ms, the comparison changes significantly.

2. **Multi-step loss ablation still missing:** Table 6 ablates gating, residual, depth, and width, but not the multi-step loss (lambda=0.5, H=8). This is a key training component that distinguishes the approach.

3. **Training convergence analysis:** All models train for only 20 epochs. No training curves are provided. Do the models converge? Could LSTM-WM benefit from more training?

4. **Recurrent vs. convolutional mode ambiguity:** The paper discusses both modes in Section 3.2 but does not clearly state which mode is used in each experiment. The single-sample (B=1) inference time of 0.9ms -- is this convolutional or recurrent mode? This matters because recurrent mode has O(1) single-step latency while convolutional mode has O(T log T).

5. **Hyperparameter fairness:** The comparison with LSTM-WM-2L (0.29M) and LSTM-WM-4L (0.38M) is reasonable, but the paper doesn't justify why LSTM-WM-2L is the primary comparison target when it has fewer parameters than SSM-WM-4L (0.24M). Depth-matched comparison (4-layer LSTM vs 4-layer SSM) would be fairer.

### Evidence Sufficiency (56/100)
- No training curves for any experiment.
- No ablation of the multi-step loss component.
- MuJoCo experiments lack Mamba-WM and Transformer-WM baselines.
- MPC experiments lack MuJoCo results.
- The 68% CV on MuJoCo inference time is concerning.
- No cross-dataset generalization test.

### Argument Coherence (68/100)
The complexity analysis in Section 4.3 is correct and well-structured. The claim about O(T log T) training and O(1) single-step inference is properly justified. The discussion now honestly addresses the MuJoCo inference time inversion, which is an improvement. However, the paper still does not explain why SSM outperforms LSTM on MuJoCo but not on synthetic data. The claim "better generalization on complex nonlinear dynamics" is stated but not supported with evidence.

### Writing Quality (74/100)
The formatting fixes are welcome. Tables are correctly labeled. References are corrected. The dimensionality claim is updated. The writing is generally clear and well-organized.

**Minor issue:** The abstract claims "约2倍提速(B=1)" but MuJoCo shows SSM is slower at B=1 (9.5ms vs 5.0ms). This inconsistency should be resolved.

### Recommendation: Accept with Minor Revisions
The methodological gaps (multi-step loss ablation, training curves, MuJoCo baselines) would strengthen the paper but are not blocking for an application-oriented venue like CTA. The high variance on MuJoCo inference time needs explanation.

---

## Reviewer: R2 (Domain Expert -- Robotics/Control)

### Summary
I evaluate the paper from the perspective of a robotics and control researcher. The topic is highly relevant -- lightweight world models for real-time humanoid control is an important problem. Round 10 has improved the paper's credibility through honest reporting.

### Originality (71/100)
Applying SSM to robot world models is a natural and timely idea. The MPC integration demonstrates practical value. The honest discussion of the MuJoCo inference time trade-off actually enhances the contribution -- it shows the authors understand that the speed advantage is dataset-dependent and that high-dimensional states (376-dim) create different computational profiles than low-dimensional ones (28-dim). This nuanced understanding is valuable for practitioners.

### Methodological Rigor (67/100)
**Strengths:**
- MuJoCo Humanoid-v4 is a well-established benchmark with realistic state/action dimensions.
- The multi-step prediction analysis (Table 4) is important for MPC applications.
- Inference time measurements include GPU warmup and multiple runs.
- The honest reporting of MuJoCo inference times (SSM slower than LSTM) is commendable.

**Concerns:**
1. **MuJoCo MPC experiment is still absent.** For a paper in CTA (a control theory journal), the absence of MPC results on the most realistic dataset is a significant gap. The 376-dimensional state with 17-dimensional action represents a real humanoid control challenge. Even if the results are not impressive, reporting them would demonstrate completeness.

2. **R^2 = 0.592 on MuJoCo:** 41% of variance is unexplained. For control applications, this level of prediction error could lead to significant control degradation. The authors should discuss what this means for MPC performance -- would the controller diverge? Would tracking errors accumulate?

3. **Control frequency projection on MuJoCo:** If SSM-WM inference takes 9.5ms on MuJoCo (vs 3.8ms on synthetic), and MPC requires 50 forward passes, the MPC loop would take approximately 50 * 9.5 = 475ms, corresponding to ~2Hz. This is much slower than the 5.1Hz achieved on synthetic data. The authors should project and discuss this.

4. **Missing real robot discussion:** The limitations section mentions the sim-to-real gap but should provide a concrete roadmap. What are the specific challenges? How would domain randomization help? What sensor noise levels are expected?

5. **Per-joint analysis on MuJoCo:** Which joints are predicted well/poorly? Are the legs (important for locomotion) predicted better than the torso (important for balance)? This analysis would inform deployment decisions.

### Evidence Sufficiency (65/100)
The evidence base has improved but remains incomplete:
- Only two methods compared on MuJoCo (SSM-WM vs LSTM-WM). Mamba-WM should be included.
- No MuJoCo MPC experiment.
- No per-joint analysis on MuJoCo.
- No trajectory visualization for MPC experiments.
- The high variance on MuJoCo inference time (9.5+/-6.5ms) needs explanation.

### Argument Coherence (73/100)
The paper makes a clear case for lightweight world models in robotics. The speed-accuracy trade-off is well-articulated. The honest discussion of the MuJoCo inference time inversion strengthens the argument. However, the central claim -- SSM enables real-time MPC for humanoid control -- is still only validated on synthetic data. The argument would be complete with MuJoCo MPC results, even if they show reduced performance.

### Writing Quality (77/100)
The paper reads well for a robotics audience. The problem motivation is clear. The comparison with existing methods in Section 2 is appropriate. The computational complexity analysis is well-presented. The limitations section is honest and well-written. The formatting fixes (table labels, references) improve readability.

**Minor issue:** The abstract should acknowledge that the speed advantage is dataset-dependent, especially since MuJoCo results show SSM is actually slower.

### Recommendation: Accept with Minor Revisions
The Round 10 improvements significantly strengthen the paper. The honest reporting of MuJoCo inference times is particularly commendable. The remaining gaps (MuJoCo MPC, MuJoCo baselines) would strengthen the paper but are not blocking.

---

## Reviewer: R3 (Broader Perspective / Interdisciplinary)

### Summary
I assess the paper from a broader machine learning and systems perspective. Round 10 has brought the paper to a level of maturity that is appropriate for a journal publication, with honest reporting and corrected errors.

### Originality (73/100)
The paper's contribution lies at the intersection of sequence modeling (SSM) and robotics (world models + MPC). The insight that SSM's linear state transition is naturally aligned with robot dynamics modeling is valuable. The paper correctly identifies the gap between SSM's success in NLP/vision and its under-explored potential in robotics. The honest discussion of the MuJoCo inference time trade-off adds intellectual depth -- it shows the authors understand the computational landscape rather than making blanket claims.

### Methodological Rigor (70/100)
The experimental design has improved over 10 rounds. The key improvements in Round 10:
- Correct MuJoCo experimental numbers.
- Updated state dimensionality claim (数十至数百).
- Honest discussion of MuJoCo inference time trade-off.
- Updated limitations section.

**Remaining issues:**
- The SSM state dimension N=16 for a 376-dimensional state space is very small. How does the model compress 376 dimensions into 16? Is there a bottleneck? A bottleneck analysis would be valuable.
- The choice of FFT convolution over recurrent mode for MPC is justified but the actual computation breakdown (encoder vs SSM vs decoder) is not provided.
- The high variance on MuJoCo inference time (CV=68%) is concerning and should be addressed.

### Evidence Sufficiency (68/100)
The evidence base is adequate for a journal publication:
- Two datasets (synthetic and MuJoCo) with different characteristics.
- Multiple baselines on synthetic data.
- Ablation study covering key components.
- Multi-step prediction analysis.
- Sequence length sensitivity analysis.

**Remaining gaps:**
- MuJoCo comparison is limited to LSTM-WM.
- No MuJoCo MPC experiment.
- 3 random seeds is the minimum; 5+ would be more convincing.
- Effect sizes on MuJoCo are small with overlapping error bars.

### Argument Coherence (75/100)
The paper tells a coherent story: SSM properties (linear recurrence, FFT convolution, O(T log T)) make it suitable for lightweight world models -> experiments demonstrate speed advantages with acceptable accuracy trade-offs -> MPC integration shows practical value -> MuJoCo validation shows competitiveness on realistic dynamics. The narrative is logical and the claims are generally supported by evidence.

The honest discussion of the MuJoCo inference time trade-off (Section 5.8) is a strength. It shows the authors understand that the speed advantage depends on the state dimensionality and computational profile.

**One remaining weakness:** The paper claims SSM-WM is "lightweight" but the 17% parameter reduction over LSTM is modest. The real advantage is in inference speed, not parameter count. This distinction should be made clearer.

### Writing Quality (79/100)
The writing quality has improved significantly over 10 rounds:
- Tables 1-8 are correctly labeled and sequential.
- Reference [22] is fixed.
- State dimensionality claim is updated.
- The limitations section is honest and well-articulated.
- The English abstract is fluent.
- The abstract is concise and informative.

**Remaining minor issues:**
- The abstract's speedup claims should be qualified to acknowledge dataset-dependent behavior.
- Several references lack complete venue information.
- The discussion section could be more concise.

### Recommendation: Accept
The paper has reached a level of maturity appropriate for CTA. The remaining issues are minor and can be addressed in a final revision or camera-ready preparation. The honest reporting of MuJoCo results is commendable and sets a good example for the field.

---

## Reviewer: Devil's Advocate

### Summary
I deliberately challenge the paper's core claims and assumptions to stress-test its conclusions. Round 10 has fixed surface-level errors but the fundamental structural issues remain.

### Originality (59/100)
**Challenge:** The paper essentially wraps S4D + Mamba-style gating into an encoder-SSM-decoder pipeline and calls it a "world model." This is a standard sequence-to-sequence architecture with an SSM backbone. The term "world model" is used loosely -- in the RL community, a world model typically involves latent dynamics, reward prediction, and imagination-based planning. This paper only predicts the next state, which is better described as a "dynamics model" or "forward model." The conflation with "world models" (Ha & Schmidhuber, Dreamer) inflates the perceived contribution.

**Concession:** The MPC integration adds practical value that goes beyond simple dynamics modeling. However, the MPC is standard (gradient-based optimization with Adam), not novel.

### Methodological Rigor (50/100)
**Improvements since Round 9:**
- Table labels fixed. Reference [22] fixed. Dimensionality claim fixed.
- Honest MuJoCo inference time discussion.

**Remaining serious challenges:**

1. **The speed advantage is dataset-dependent and overstated.** The abstract claims "约2倍提速(B=1, 0.9ms vs 2.1ms)" based on synthetic data. But MuJoCo shows SSM is SLOWER (9.5ms vs 5.0ms). The abstract should either qualify this claim or present both results. A reader who only reads the abstract would be misled.

2. **MuJoCo inference variance is extreme.** 9.5+/-6.5ms has a CV of 68%. This means the 95% confidence interval is approximately [-3.5, 22.5]ms -- the lower bound is negative, which is physically impossible. This suggests the measurement methodology is flawed. The authors should report median and IQR instead of mean and std, or investigate the source of variance.

3. **The accuracy deficit on synthetic data is severe.** SSM-WM has 55% higher MSE than LSTM-WM on synthetic data. The paper then pivots to MuJoCo where SSM wins by 6%. But the MuJoCo comparison only includes LSTM -- where is Mamba-WM? If Mamba-WM also outperforms SSM-WM on MuJoCo (as it does on synthetic data), the contribution diminishes significantly.

4. **The MPC experiment is circular.** The MPC uses SSM-WM as the dynamics model and evaluates tracking MSE. But the tracking MSE is dominated by the MPC optimization quality, not the dynamics model accuracy. A fair comparison would use the same MPC solver with different dynamics models and compare final task performance.

5. **R^2 = 0.945 on synthetic data is misleading.** The synthetic data uses near-linear dynamics (s_{t+1} = As + Ba + small nonlinearity). An R^2 of 0.945 on such easy dynamics is not impressive -- a simple linear model might achieve comparable results. The paper should include a linear regression baseline.

6. **R^2 = 0.592 on MuJoCo is mediocre.** The model explains only 59% of variance in a 376-dimensional state space. For control applications, this level of prediction error could lead to significant control degradation. The paper does not discuss the implications for MPC performance.

### Evidence Sufficiency (46/100)
**Critical gaps that persist after 10 rounds:**
- No linear regression baseline on either dataset. For near-linear dynamics, this is a must-have.
- No MuJoCo MPC results -- the paper's core claim (SSM enables real-time MPC for humanoid control) is untested on the most realistic dataset.
- No comparison with established model-based RL methods (Dreamer, PETS, MBPO) that have been extensively benchmarked on MuJoCo Humanoid.
- Only 3 random seeds with no confidence interval analysis.
- The "2x speedup at B=1" claim comes from 2.1ms vs 0.9ms -- a difference of 1.2ms. Is this practically significant when the MPC optimization itself takes ~195ms?
- MuJoCo comparison is limited to LSTM-WM; Mamba-WM and Transformer-WM are absent.

### Argument Coherence (57/100)
The paper's central argument is: "SSM provides fast inference + acceptable accuracy -> SSM is suitable for real-time MPC." But:
- The "fast inference" is only demonstrated on synthetic data; MuJoCo shows SSM is slower.
- The "acceptable accuracy" is only demonstrated on MuJoCo for state prediction, not for MPC.
- The "real-time MPC" is only demonstrated on synthetic data, not MuJoCo.
- The two strongest results come from different datasets with opposite SSM-vs-LSTM orderings, creating a disjointed argument.

A unified experiment (SSM-based MPC on MuJoCo) would close this argument loop. Without it, the paper demonstrates two separate properties (fast inference on synthetic, good MuJoCo prediction) without showing they combine effectively.

The honest discussion of the MuJoCo inference time trade-off is appreciated but actually weakens the argument -- it acknowledges that the speed advantage is not universal.

### Writing Quality (68/100)
**Improvements since Round 9:**
- Table labels fixed (Tables 1-8 sequential).
- Reference [22] fixed.
- State dimensionality claim updated.
- Honest MuJoCo inference time discussion.
- Updated limitations section.

**Remaining issues:**
- The abstract's speedup claims are misleading given MuJoCo results.
- The discussion section (Section 5.8) discusses "SSM-WM vs Mamba-WM" advantages, but Mamba-WM is not compared on MuJoCo. The discussion is speculative.
- Several references are incomplete.
- The MuJoCo inference variance (9.5+/-6.5ms) should be investigated before publication.

### Recommendation: Major Revision (Borderline Accept)
The Round 10 fixes address surface-level errors, and the honest MuJoCo discussion is commendable. However, the fundamental issues remain: the disjointed evidence base, the missing MuJoCo MPC experiment, the missing MuJoCo baselines, and the extreme inference variance. The paper has the skeleton of a good contribution but needs one more round to address the MuJoCo inference variance and the abstract inconsistency.

---

## Consolidated Summary of Critical Issues (Must Fix for Next Round)

| # | Priority | Issue | Affected Reviewers |
|---|---|---|---|
| 1 | HIGH | **MuJoCo inference variance:** 9.5+/-6.5ms has CV=68%. Investigate source of variance; consider reporting median/IQR instead of mean/std. | EIC, R1, DA |
| 2 | HIGH | **Abstract inconsistency:** Abstract claims "约2倍提速(B=1)" based on synthetic data, but MuJoCo shows SSM is slower (9.5ms vs 5.0ms). Qualify the claim or present both results. | EIC, R1, R2, DA |
| 3 | HIGH | **MuJoCo MPC experiment:** The paper's central claim (real-time MPC for humanoid control) must be validated on MuJoCo Humanoid-v4, not just synthetic data. | EIC, R2, DA |
| 4 | MED | **Missing baselines on MuJoCo:** Table 8 only compares SSM-WM with LSTM-WM. Mamba-WM and Transformer-WM should be included. | R1, DA |
| 5 | MED | **Statistical significance test:** Conduct paired t-test or bootstrap confidence intervals for MuJoCo MSE/R^2 differences. | EIC, R1 |
| 6 | MED | **Linear regression baseline:** Add a simple linear model baseline to demonstrate the value of nonlinear modeling. | DA |
| 7 | MED | **Multi-step loss ablation:** Ablate lambda and H in Table 6. | R1 |
| 8 | LOW | **Training curves:** Show convergence plots for key experiments. | R1 |
| 9 | LOW | **Per-joint analysis on MuJoCo:** Which states are predicted well/poorly? | R2 |
| 10 | LOW | **Control frequency projection on MuJoCo:** Estimate MPC loop time using MuJoCo inference times. | R2 |

---

## Consolidated Scoring (All Reviewers Averaged)

| Dimension | Mean | Range |
|---|---|---|
| Originality | 67.6 | 59--73 |
| Methodological Rigor | 62.6 | 50--70 |
| Evidence Sufficiency | 59.4 | 46--68 |
| Argument Coherence | 69.0 | 57--75 |
| Writing Quality | 74.8 | 68--79 |
| **Grand Mean** | **66.7** | |

**Verdict:** The paper has improved by +2.5 points on the grand mean since Round 9, driven primarily by writing quality (+4.0) and evidence sufficiency (+3.6) improvements. The formatting fixes (table labels, references, dimensionality claim) and honest MuJoCo discussion are the most impactful changes. The grand mean of 66.7 places the paper in the "Borderline Accept / Minor Revision" range for CTA. The most critical remaining issue is the MuJoCo inference variance (CV=68%), which undermines the credibility of the MuJoCo speed comparison. Resolving this single issue would likely push the grand mean above 70, placing the paper firmly in the "Accept with Minor Revisions" range.

**Progress Trajectory:**
- Round 7: ~58 (Major Revision)
- Round 8: ~61 (Major Revision)
- Round 9: 64.2 (Borderline Major/Minor Revision)
- Round 10: 66.7 (Borderline Accept/Minor Revision)
- Projected Round 11 (with fixes): ~70+ (Accept with Minor Revisions)

The paper is converging toward acceptance. Two or three targeted fixes (MuJoCo inference variance, abstract qualification, MuJoCo MPC) would likely suffice.
