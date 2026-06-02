# Round 9 Peer Review -- CTA (控制理论与应用)

**Paper Title:** 面向人形机器人状态预测的轻量级状态空间世界模型
**Manuscript:** /mnt/e/Project/SSM-World-Model/paper/main.tex (~859 lines)

---

## Scoring Summary

| Dimension | EIC | R1-Methodology | R2-Domain | R3-Perspective | Devil's Advocate |
|---|---|---|---|---|---|
| Originality | 68 | 65 | 70 | 72 | 58 |
| Methodological Rigor | 62 | 58 | 65 | 68 | 48 |
| Evidence Sufficiency | 58 | 52 | 62 | 65 | 42 |
| Argument Coherence | 70 | 66 | 72 | 74 | 55 |
| Writing Quality | 72 | 70 | 74 | 76 | 62 |
| **Overall** | **66** | **62** | **69** | **71** | **53** |

---

## Reviewer: EIC (Editor-in-Chief)

### Summary
This paper proposes SSM-WM, a lightweight world model based on diagonal state space models (S4D) combined with Mamba-style gated blocks, applied to humanoid robot state prediction and integrated into an MPC framework. The paper is now in its 9th round of revision and has added MuJoCo Humanoid-v4 experiments (Table 7, state_dim=376, action_dim=17), which represents a meaningful step toward more realistic evaluation. The topic is timely and the lightweight inference speed is noteworthy.

### Originality (68/100)
The combination of S4D diagonal SSM with Mamba-style gated blocks for robot world models is a reasonable contribution, though the individual components (S4D, Mamba gating) are well-established. The novelty lies primarily in the application context rather than architectural innovation. The MPC integration adds practical value.

### Methodological Rigor (62/100)
**Positive:** Training procedure is well-documented with hyperparameter search details (Appendix Table A1). Ablation study covers key components.

**Critical Issues:**
1. **Duplicate table label (line 534 vs 561):** Both "多步预测MSE对比" and "序列长度敏感性分析" are labeled as Table 4. This must be corrected before publication.
2. **Missing MuJoCo ablation:** The ablation study (Table 5) is only performed on synthetic data. Given that MuJoCo results show a different accuracy ordering (SSM > LSTM), the ablation findings may not transfer.
3. **Reference formatting error (bibitem{22}):** The S4D paper citation reads "GU A, GU A, GU A" -- this is clearly erroneous and must be fixed.
4. **Inconsistent MSE scales:** Synthetic dataset reports MSE in units of 10^-3, while MuJoCo reports raw MSE (0.834). This inconsistency impedes direct comparison. The authors should either normalize both or clearly explain the scale difference.
5. **Statistical significance:** The MuJoCo MSE difference (0.834 vs 0.884) and R^2 difference (0.592 vs 0.568) are reported without significance tests. With only 3 seeds and relatively large standard deviations, these differences may not be statistically significant.

### Evidence Sufficiency (58/100)
The addition of MuJoCo experiments strengthens the paper, but several gaps remain:
- MPC experiments are only conducted on the synthetic dataset, not on MuJoCo. Given that MPC is a core contribution, this omission is significant.
- No comparison with MuJoCo-specific baselines (e.g., model-based RL methods like Dreamer/PETS that have been benchmarked on Humanoid tasks).
- The R^2 of 0.592 on MuJoCo means the model explains only ~59% of variance -- this raises questions about practical utility for control.

### Argument Coherence (70/100)
The paper's narrative arc (lightweight architecture -> fast inference -> MPC integration) is logically structured. However, the discussion section attempts to reconcile the contradictory accuracy results across datasets without fully succeeding. The claim that SSM-WM has "better generalization" on complex dynamics needs stronger supporting evidence.

### Writing Quality (72/100)
The paper is generally well-written in Chinese academic style. The abstract is concise and informative. However, the duplicate table label and reference error are notable blemishes that would be caught in production.

### Recommendation: Major Revision
The MuJoCo addition is a positive step, but the MPC-on-MuJoCo gap, the lack of significance tests, and the reference error are blocking issues.

---

## Reviewer: R1 (Methodology Expert)

### Summary
I focus on the methodological soundness of the proposed approach and experimental design.

### Originality (65/100)
The architecture is a straightforward composition of existing components. S4D parameterization and Mamba-style gating are individually well-known; the contribution is their assembly for robot state prediction. This is acceptable for an application-oriented journal but limits the methodological novelty.

### Methodological Rigor (58/100)
**Critical Concerns:**

1. **Synthetic dataset calibration:** The synthetic dataset uses random dynamics matrices with N(0, 0.01) entries. The state transition rule s_{t+1} = A*s_t + B*a_t + 0.1*tanh(s_t .* a_t) + epsilon is nearly linear with weak nonlinearity. This explains why LSTM outperforms SSM here -- LSTM is well-suited for near-linear dynamics. The MuJoCo dataset, with genuine contact dynamics and strong nonlinearity, favors SSM. The authors should explicitly discuss this dataset-dependent behavior rather than presenting it as a uniform advantage.

2. **Hyperparameter fairness:** SSM-WM uses D=128, N=16, L=4 (0.24M params), while LSTM-WM uses 2 layers with hidden_dim=128 (0.29M). The comparison is reasonable in parameter count, but LSTM-WM-4L (0.38M) performs worse than LSTM-WM-2L, suggesting overfitting. The question is whether the fair comparison should be LSTM-WM-2L (fewer params, better accuracy) or LSTM-WM-4L (same depth). The paper doesn't justify this choice.

3. **Training budget:** All models train for only 20 epochs. Is this sufficient for convergence? No training curves are provided. SSM-based models may require different training dynamics than LSTMs.

4. **Multi-step loss ablation missing:** Table 5 ablates gating, residual, depth, and width, but does not ablate the multi-step loss (lambda=0.5, H=8). This is a key training component that distinguishes the approach and should be ablated.

5. **Recurrent vs. convolutional mode:** The paper claims convolutional mode for inference but recurrent mode is standard for single-step prediction. The 0.9ms single-sample time -- is this with convolutional or recurrent mode? This is ambiguous in the text.

### Evidence Sufficiency (52/100)
- No learning curves for any experiment.
- No ablation of the multi-step loss component.
- MuJoCo experiments lack Mamba-WM and Transformer-WM baselines (only LSTM is compared).
- MPC experiments lack MuJoCo results.
- No cross-dataset generalization test (train on synthetic, test on MuJoCo or vice versa).

### Argument Coherence (66/100)
The complexity analysis in Section 4.3 is correct and well-structured. The claim about O(T log T) training and O(1) single-step inference is properly justified. However, the discussion of why SSM outperforms LSTM on MuJoCo but not on synthetic data is hand-wavy ("better generalization on complex nonlinear dynamics") without theoretical or empirical backing.

### Writing Quality (70/100)
Equations are properly formatted. The architecture diagram (Figure 1) is clear. However, the two tables both labeled "Table 4" is a significant error. Reference [22] has a formatting error with three identical author entries.

### Recommendation: Major Revision
The methodological gaps (multi-step loss ablation, training convergence analysis, missing baselines on MuJoCo) need to be addressed.

---

## Reviewer: R2 (Domain Expert -- Robotics/Control)

### Summary
I evaluate the paper from the perspective of a robotics and control researcher. The topic is highly relevant -- lightweight world models for real-time humanoid control is an important problem.

### Originality (70/100)
Applying SSM to robot world models is a natural and timely idea. The MPC integration demonstrates practical utility. While the individual components are not novel, the system-level contribution of demonstrating SSM-based world models for humanoid control is valuable for the CTA community.

### Methodological Rigor (65/100)
**Strengths:**
- MuJoCo Humanoid-v4 is a well-established benchmark with state_dim=376 and action_dim=17, providing a realistic testbed.
- The multi-step prediction analysis (Table 4) is important for MPC applications.
- Inference time measurements include GPU warmup and multiple runs, which is good practice.

**Concerns:**
1. **MPC only on synthetic data:** The MPC experiment (Table 6) uses only the 28-dimensional synthetic system. For a paper claiming applicability to humanoid robots, the MPC experiment should be conducted on MuJoCo Humanoid. The 376-dimensional state with 17-dimensional action is much more representative of real humanoid control challenges.

2. **MPC tracking task is trivial:** The MPC experiment uses a simple point-tracking objective. Real humanoid tasks involve contact-rich locomotion, balance recovery, and multi-contact manipulation. The authors should discuss how their method would scale to these more demanding scenarios.

3. **R^2 = 0.592 on MuJoCo:** This means 41% of variance is unexplained. For control applications, this level of prediction error could lead to significant control degradation. The authors should report the actual MPC performance on MuJoCo (even if worse than synthetic) to give readers a realistic assessment.

4. **Missing real robot discussion:** The paper acknowledges this limitation but should provide a concrete roadmap for sim-to-real transfer. What are the specific challenges? How would domain randomization help?

5. **Control frequency analysis:** SSM-WM-MPC achieves 5.1Hz on synthetic data. What frequency would be achievable on MuJoCo with 376-dimensional state? This projection is important for readers.

### Evidence Sufficiency (62/100)
The MuJoCo experiment is the most important addition. However:
- Only two methods compared (SSM-WM vs LSTM-WM) on MuJoCo.
- No MuJoCo MPC experiment.
- No analysis of per-joint or per-state prediction quality on MuJoCo (which joints are predicted well/poorly?).
- The trajectory tracking experiment lacks visualization of actual trajectories.

### Argument Coherence (72/100)
The paper makes a clear case for lightweight world models in robotics. The speed-accuracy trade-off is well-articulated. The claim that SSM-WM has "better generalization on complex dynamics" is plausible given the SSM's linear state transition structure, but needs more evidence.

### Writing Quality (74/100)
The paper reads well for a robotics audience. The problem motivation is clear. The comparison with existing methods (Dreamer, RT-2) in Section 2 is appropriate. The computational complexity analysis is well-presented.

### Recommendation: Accept with Minor Revisions
The MuJoCo addition significantly strengthens the paper for a robotics venue. The main gaps are the absence of MuJoCo MPC results and the limited comparison set on MuJoCo.

---

## Reviewer: R3 (Broader Perspective / Interdisciplinary)

### Summary
I assess the paper from a broader machine learning and systems perspective, considering its contribution to the field at large.

### Originality (72/100)
The paper's contribution lies at the intersection of sequence modeling (SSM) and robotics (world models + MPC). This is a growing area of interest. The insight that SSM's linear state transition is naturally aligned with robot dynamics modeling is valuable. The paper correctly identifies the gap between SSM's success in NLP/vision and its under-explored potential in robotics.

### Methodological Rigor (68/100)
The experimental design has improved over 9 rounds. The addition of MuJoCo experiments with realistic state/action dimensions (376/17) addresses a key criticism. The ablation study, while not exhaustive, covers the main architectural choices.

**Remaining issues:**
- The SSM state dimension N=16 seems very small for a 376-dimensional state space. How does the model compress 376 dimensions into a 16-dimensional latent state? Is there a bottleneck analysis?
- The choice of FFT convolution over recurrent mode for MPC is justified but the actual computation breakdown (encoder time vs SSM time vs decoder time) is not provided.

### Evidence Sufficiency (65/100)
The evidence base is adequate for a conference-level contribution but borderline for a journal:
- 3 random seeds is the minimum; 5+ would be more convincing.
- Effect sizes on MuJoCo are small (6% MSE improvement) with overlapping error bars.
- The synthetic dataset (100 episodes x 150 steps) is small by modern standards.

### Argument Coherence (74/100)
The paper tells a coherent story: SSM properties (linear recurrence, FFT convolution, O(T log T)) make it suitable for lightweight world models -> experiments demonstrate speed advantages with acceptable accuracy trade-offs -> MPC integration shows practical value. The narrative is logical and the claims are generally supported by evidence.

One narrative weakness: the paper claims SSM-WM is "lightweight" but the 17% parameter reduction over LSTM is modest. The real advantage is in inference speed, not parameter count. This distinction should be made clearer.

### Writing Quality (76/100)
The writing quality has improved significantly over the revision rounds. The abstract is well-structured. The English abstract is fluent. The limitation section is honest and appropriate.

**Minor issues:**
- Line 114: "状态维度高(通常20--50维)" -- MuJoCo Humanoid has 376 dimensions, which contradicts this claim. Update this statement.
- The reference to "DreamerV3" (bibitem{26}) is incomplete -- the full author list and venue should be included.
- Several references lack complete venue/conference information.

### Recommendation: Accept with Minor Revisions
The paper has reached a reasonable level of maturity. The main remaining issues are minor (statistical analysis, reference formatting, updating claims about state dimensionality).

---

## Reviewer: Devil's Advocate

### Summary
I deliberately challenge the paper's core claims and assumptions to stress-test its conclusions.

### Originality (58/100)
**Challenge:** The paper essentially wraps S4D + Mamba-style gating into an encoder-SSM-decoder pipeline and calls it a "world model." This is a standard sequence-to-sequence architecture with an SSM backbone. The term "world model" is used loosely -- in the RL community, a world model typically involves latent dynamics, reward prediction, and imagination-based planning. This paper only predicts the next state, which is better described as a "dynamics model" or "forward model." The conflation with "world models" (Ha & Schmidhuber, Dreamer) inflates the perceived contribution.

### Methodological Rigor (48/100)
**Serious challenges:**

1. **The speed advantage is overstated.** The 7.3x speedup is measured at batch_size=64. At batch_size=1 (realistic for online control), the speedup is only 2.1x (2.1ms vs 0.9ms). For a single MPC step requiring 50 forward passes, LSTM takes 105ms vs SSM's 45ms -- the advantage shrinks further when amortized.

2. **The accuracy deficit is severe on synthetic data.** SSM-WM has 55% higher MSE than LSTM-WM on synthetic data. The paper then pivots to MuJoCo where SSM wins by 6%. But the MuJoCo comparison only includes LSTM -- where is Mamba-WM? Where is Transformer-WM? The selective reporting of only LSTM comparison on MuJoCo is suspicious. If Mamba-WM also outperforms SSM-WM on MuJoCo (as it does on synthetic data), the contribution diminishes significantly.

3. **The MPC experiment is circular.** The MPC uses SSM-WM as the dynamics model and evaluates tracking MSE. But the tracking MSE is dominated by the MPC optimization quality, not the dynamics model accuracy. A fair comparison would use the same MPC solver with different dynamics models and compare final task performance (e.g., success rate, smoothness, energy consumption).

4. **No failure mode analysis.** When does SSM-WM fail? What types of dynamics are problematic? The paper only reports aggregate metrics without analyzing failure cases.

5. **R^2 = 0.945 on synthetic data is misleading.** The synthetic data uses near-linear dynamics (s_{t+1} = As + Ba + small nonlinearity). An R^2 of 0.945 on such easy dynamics is not impressive -- a simple linear model might achieve comparable results. The paper should include a linear regression baseline.

### Evidence Sufficiency (42/100)
**Critical gaps:**
- No linear regression baseline on either dataset. For near-linear dynamics, this is a must-have.
- No MuJoCo MPC results -- the paper's core claim (SSM enables real-time MPC for humanoid control) is untested on the most realistic dataset.
- No comparison with established model-based RL methods (Dreamer, PETS, MBPO) that have been extensively benchmarked on MuJoCo Humanoid.
- Only 3 random seeds with no confidence interval analysis.
- The "2x speedup at B=1" claim comes from 2.1ms vs 0.9ms -- a difference of 1.2ms. Is this practically significant when the MPC optimization itself takes ~195ms?

### Argument Coherence (55/100)
The paper's central argument is: "SSM provides fast inference + acceptable accuracy -> SSM is suitable for real-time MPC." But:
- The "acceptable accuracy" is only demonstrated on MuJoCo for state prediction, not for MPC.
- The "real-time MPC" is only demonstrated on synthetic data, not MuJoCo.
- The two strongest results come from different datasets, creating a disjointed argument.

A unified experiment (SSM-based MPC on MuJoCo) would close this argument loop. Without it, the paper demonstrates two separate properties (fast inference, good MuJoCo prediction) without showing they combine effectively.

### Writing Quality (62/100)
- Duplicate table labels (Table 4 appears twice) -- unprofessional for a 9th revision.
- Reference [22] has "GU A, GU A, GU A" as authors -- clearly copy-paste error.
- Line 114: claims state dimensions are "typically 20-50" but MuJoCo has 376 -- contradicts the paper's own data.
- The abstract claims "约17%参数减少" -- this is 17% compared to LSTM-2L, not a general claim. Should specify the baseline.
- Section 5.8 (Discussion) discusses "SSM-WM vs Mamba-WM" advantages, but Mamba-WM is not compared on MuJoCo. The discussion is speculative.

### Recommendation: Major Revision (Borderline Reject)
The paper has the skeleton of a good contribution but suffers from selective result presentation, missing critical experiments (MuJoCo MPC, Mamba-WM on MuJoCo, linear baseline), and repeated formatting errors after 9 rounds of revision. The core claim -- that SSM enables real-time MPC for humanoid control -- remains unsubstantiated on the most realistic dataset.

---

## Consolidated Summary of Critical Issues (Must Fix for Next Round)

| # | Priority | Issue | Affected Reviewers |
|---|---|---|---|
| 1 | HIGH | **MuJoCo MPC experiment:** The paper's central claim (real-time MPC for humanoid control) must be validated on MuJoCo Humanoid-v4, not just synthetic data. | EIC, R2, DA |
| 2 | HIGH | **Missing baselines on MuJoCo:** Table 7 only compares SSM-WM with LSTM-WM. Mamba-WM and Transformer-WM must be included. | R1, R2, DA |
| 3 | HIGH | **Duplicate table label:** Both multi-step analysis and sequence sensitivity are labeled "Table 4". Fix immediately. | EIC, R1 |
| 4 | HIGH | **Reference [22] error:** "GU A, GU A, GU A" must be corrected to the actual S4D authors (Gu, Goel, Re). | EIC, R1, DA |
| 5 | MED | **Statistical significance test:** Conduct paired t-test or bootstrap confidence intervals for MuJoCo MSE/R^2 differences. | EIC, R1 |
| 6 | MED | **Linear regression baseline:** Add a simple linear model baseline to demonstrate the value of nonlinear modeling. | DA |
| 7 | MED | **Multi-step loss ablation:** Ablate lambda and H in Table 5. | R1 |
| 8 | MED | **Update dimensionality claim:** Line 114 claims "typically 20-50 dimensions" but MuJoCo has 376. Update to reflect actual range. | R3, DA |
| 9 | LOW | **Training curves:** Show convergence plots for key experiments. | R1 |
| 10 | LOW | **Per-joint analysis on MuJoCo:** Which states are predicted well/poorly? This informs future work. | R2 |

---

## Consolidated Scoring (All Reviewers Averaged)

| Dimension | Mean | Range |
|---|---|---|
| Originality | 66.6 | 58--72 |
| Methodological Rigor | 60.2 | 48--68 |
| Evidence Sufficiency | 55.8 | 42--65 |
| Argument Coherence | 67.4 | 55--74 |
| Writing Quality | 70.8 | 62--76 |
| **Grand Mean** | **64.2** | |

**Verdict:** The paper has improved substantially over 9 rounds. The MuJoCo addition is the single most impactful revision. However, the disjointed evidence base (speed results on synthetic, accuracy results on MuJoCo, MPC only on synthetic) prevents the paper from making a unified, convincing argument. Addressing the top-4 critical issues would likely bring the grand mean above 70, placing the paper in the "Accept with Minor Revisions" range for CTA.
