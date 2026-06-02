# Round 13 Peer Review Report -- Reviewer 1 (Methodology Expert)

**Paper Title:** 面向人形机器人状态预测的轻量级状态空间世界模型
**Journal:** 控制理论与应用 (Control Theory & Applications)
**Manuscript:** /mnt/e/Project/SSM-World-Model/paper/main.tex (~1038 lines)
**Review Date:** 2026-06-02
**Review Round:** Round 13
**Reviewer:** Peer Reviewer 1 (Methodology Expert)
**Previous Round Score (R12):** 68.0 overall

---

## Reviewer Profile

**Expertise:** Deep learning architectures, sequence modeling, statistical validation, experimental design.

**Focus Areas for This Review:** Research design rigor, sampling strategy, data collection, analysis method selection, statistical validity, effect sizes, reproducibility, data transparency.

---

## 1. Summary of Changes Since Round 12

I begin by verifying each fix claimed in the Round 12 editorial decision:

| Issue | Status | Verification |
|-------|--------|--------------|
| Reference [22] corrected | FIXED | Line 945: "GU A, GOEL K, RE C" -- correct. |
| Linear regression in Table 1 | FIXED | Line 475: 线性回归 row with MSE=2.50, MAE=0.035, R^2=0.890. |
| Table 8 now has all 4 baselines | FIXED | Lines 788-791: LSTM-WM, Transformer-WM, Mamba-WM, SSM-WM. |
| Training convergence analysis | FIXED | Lines 667-671: convergence within 20 epochs, train-val gap <0.15x10^-3. |
| Multi-step loss ablation (Table 6b) | FIXED | Lines 639-664: systematic ablation of lambda and H. |
| Statistical significance on MuJoCo | FIXED | p<0.01 for SSM vs LSTM, p=0.27 vs Mamba (lines 800-801). |

All six issues from Round 12 have been addressed. This is a commendable level of responsiveness.

---

## 2. Strengths

### S1: Comprehensive Baseline Coverage Across Both Datasets

The paper now provides complete baseline comparisons on both datasets. Table 1 includes linear regression, LSTM (2L and 4L), Transformer, Mamba, and SSM-WM on the synthetic dataset. Table 8 includes all four neural network baselines on MuJoCo Humanoid. This enables readers to make informed comparisons and positions SSM-WM within the landscape of existing approaches.

### S2: Systematic Ablation Design

The ablation study is now comprehensive in three dimensions:
- **Architecture ablation** (Table 6): gating mechanism, residual connections, layer count (L=2/4/6), SSM state dimension (N=16/32/128), hidden dimension (D=64/128/256).
- **Training loss ablation** (Table 6b): multi-step loss weight lambda (0/0.1/0.5/1.0) and rollout horizon H (4/8/16).
- **MuJoCo ablation** (Table 8b): gating, residual, depth on the high-dimensional dataset.

This three-pronged ablation design is methodologically sound and provides strong evidence for each design choice.

### S3: Statistical Rigor Improvement

The paper now reports:
- Mean and standard deviation across 3 random seeds (42, 123, 456) for all results.
- Paired t-tests with explicit p-values for key comparisons.
- 95% confidence intervals for critical comparisons (e.g., SSM-WM vs LSTM-WM on MuJoCo).
- Explicit significance level (alpha=0.05).

This represents a substantial improvement over earlier rounds where no statistical testing was provided.

### S4: Honest Limitations Section

The limitations section (lines 881-891) is unusually thorough for a methods paper. It explicitly acknowledges:
- The MuJoCo MPC gap (item 4).
- The R^2=0.592 ceiling on MuJoCo (item 3).
- The small synthetic dataset (item 6).
- The limited number of random seeds (item 6).
- The inference time regression on MuJoCo (item 2).

This level of transparency strengthens the paper's credibility.

### S5: Multi-Step Prediction Analysis

Table 4 (multi-step MSE at H=1/4/8/16) is a critical addition for MPC-focused papers. The finding that SSM-WM's error accumulation is slower than its single-step gap (38-47% vs 55% MSE difference from LSTM) is practically meaningful and well-presented.

### S6: Training Convergence Documentation

The training convergence analysis (lines 667-671) with the training curve description (Figure 2) provides necessary evidence that 20 epochs is sufficient. The reported train-validation gap (<0.15x10^-3) and consistent trends demonstrate that overfitting is not a concern, which is important given the small synthetic dataset.

---

## 3. Weaknesses

### W1: Synthetic Dataset Dynamics Are Insufficiently Representative (Methodological Design)

**Problem:** The synthetic dataset uses the dynamics s_{t+1} = A*s + B*a + 0.1*tanh(s.*a) + epsilon (Appendix B, lines 1007-1012). This is fundamentally a near-linear system with a small nonlinear perturbation. The state dimension is only 28, and each episode has 150 steps.

**Why it matters:** The paper's title claims "面向人形机器人状态预测" (for humanoid robot state prediction), but the synthetic dataset does not represent humanoid dynamics. Humanoid robots exhibit contact dynamics, friction, collision, and discontinuous forces -- none of which are captured by the smooth tanh coupling term. The 55% MSE gap between SSM-WM and LSTM on this dataset may be an artifact of the near-linear dynamics favoring LSTM's nonlinear recurrence, rather than a fundamental limitation of SSM.

**Evidence:** On MuJoCo (which has realistic contact dynamics), SSM-WM actually outperforms LSTM by 6%. This reversal supports the hypothesis that the synthetic dataset is not representative.

**Severity:** Moderate. The MuJoCo experiments partially address this, but the synthetic dataset remains the primary basis for the speed claims (Tables 2, 3) and MPC demonstration (Table 7).

**Suggestion:** (1) Explicitly state in Section 5.1 that the synthetic dataset represents a simplified, near-linear dynamics scenario and is not intended to mimic real humanoid dynamics. (2) Consider adding a second synthetic variant with stronger nonlinearities (e.g., contact-like discontinuities) to bridge the gap between synthetic and MuJoCo settings.

### W2: Limited Statistical Power (Statistical Validity)

**Problem:** All experiments use 3 random seeds (42, 123, 456). The paper acknowledges this as a limitation (line 890: "每组实验仅使用3个随机种子, 统计功效有限").

**Why it matters:** With n=3, the paired t-test has very low statistical power. For the SSM-WM vs Mamba-WM comparison on MuJoCo (p=0.27), the non-significant result could be a Type II error (failure to detect a real difference) rather than evidence of equivalence. The 95% CI for this comparison is not reported, making it impossible to assess whether the true difference is negligible or potentially meaningful.

**Quantitative analysis:** With 3 seeds and the reported standard deviations (SSM-WM: 0.029, Mamba-WM: 0.025), a two-sided paired t-test at alpha=0.05 has approximately 15-20% power to detect a 10% difference in means. This means there is an 80-85% chance of missing a real effect.

**Severity:** Moderate. This does not invalidate the results but limits the strength of statistical claims.

**Suggestion:** (1) Report 95% CIs for all pairwise comparisons, not just the LSTM comparison. (2) For the Mamba-WM vs SSM-WM comparison, consider reporting equivalence testing (TOST) rather than relying on non-significant p-values, which cannot establish equivalence. (3) If computational resources allow, increase to 5 seeds for future work.

### W3: Effect Size Reporting Is Incomplete (Statistical Practice)

**Problem:** The paper reports p-values and 95% CIs for some comparisons but does not report standardized effect sizes (e.g., Cohen's d).

**Why it matters:** P-values indicate whether a difference is statistically significant, but not the magnitude of the difference. For practical decision-making (e.g., "should I use SSM-WM or LSTM-WM for my robot?"), effect sizes are more informative than p-values.

**Example:** The SSM-WM vs LSTM-WM difference on MuJoCo (MSE: 0.834 vs 0.889) has p<0.01, but the effect size (Cohen's d approximately 2.1, based on the reported means and SDs) indicates a large effect. This quantitative characterization would strengthen the paper's claims.

**Severity:** Low-moderate. This is a best-practice issue, not a fundamental flaw.

**Suggestion:** Report Cohen's d for key comparisons (at minimum: SSM-WM vs LSTM-WM, SSM-WM vs Mamba-WM) in Tables 1 and 8.

### W4: MPC Fairness Concern (Experimental Design)

**Problem:** The MPC experiment (Table 7) uses a fixed compute budget (50 Adam iterations) rather than a fixed accuracy target. This design inherently favors faster models.

**Why it matters:** If LSTM-MPC were given more iterations (e.g., 350 iterations to match the total time of SSM-WM-MPC), it might achieve significantly better tracking accuracy. The current comparison shows that SSM-WM-MPC is faster but does not demonstrate that it achieves comparable accuracy under equal computational budgets.

**Counter-argument:** The paper's claim is about real-time control, where the time budget is fixed by the control frequency requirement. In this context, the fixed-iteration comparison is reasonable. However, the paper should explicitly state this design choice and its implications.

**Severity:** Low-moderate. The paper's framing (real-time MPC) justifies the design, but the limitation should be stated.

**Suggestion:** Add a sentence in Section 5.7 clarifying that the 50-iteration budget reflects a real-time constraint, and that LSTM-MPC would likely achieve better accuracy with more iterations but at the cost of control frequency.

### W5: MuJoCo R^2 Ceiling (Evidence Interpretation)

**Problem:** SSM-WM achieves R^2=0.592 on MuJoCo, meaning 41% of variance is unexplained. The paper discusses this (lines 870-874) but does not contextualize it against state-of-the-art dynamics models.

**Why it matters:** Without knowing what R^2 values other dynamics models achieve on MuJoCo Humanoid-v4, readers cannot assess whether 0.592 is good, acceptable, or poor. If neural network dynamics models typically achieve R^2 > 0.8 on this benchmark, then 0.592 would indicate a significant limitation.

**Severity:** Low-moderate. The paper acknowledges this as a limitation, but a brief comparison to published baselines would be informative.

**Suggestion:** If available, cite R^2 values from other dynamics models on MuJoCo Humanoid. If not available, state that direct comparison is difficult due to different experimental setups.

### W6: Exposure Bias Not Experimentally Validated (Training Methodology)

**Problem:** The paper discusses exposure bias (line 362) and proposes multi-step loss as a mitigation. Table 6b shows that multi-step loss reduces multi-step MSE by 28.3% (from 4.85 to 3.48 at H=8). However, the paper does not compare against other exposure bias mitigation strategies (e.g., scheduled sampling [39]).

**Why it matters:** The multi-step loss is presented as the primary solution to exposure bias, but without comparison to alternatives, readers cannot assess whether it is the best approach or merely one option.

**Severity:** Low. This is a "nice to have" rather than a requirement. The current ablation is sufficient to validate the multi-step loss's contribution.

**Suggestion:** A brief mention in the discussion that scheduled sampling [39] is an alternative approach, and that multi-step loss was chosen for its simplicity and effectiveness, would suffice.

---

## 4. Methodology Deep-Dive

### 4.1 Data Collection and Sampling Strategy

**Synthetic Dataset:**
- 100 episodes x 150 steps = 15,000 total transitions.
- Train/validation split: 80/20 episodes (12,000 / 3,000 transitions).
- Each episode has independently sampled A, B matrices (N(0, 0.01)).
- Process noise: epsilon ~ N(0, 0.001).

Assessment: The dataset is small by modern standards. The 80/20 split is standard. The per-episode randomization of dynamics parameters provides diversity, but the near-linear dynamics limit the dataset's representativeness for humanoid robotics.

**MuJoCo Humanoid Dataset:**
- 200 episodes x ~200 steps = ~40,000 total transitions.
- Train/validation split: 80/20 (32,000 / 8,000 transitions).
- State dimension: 376 (joint angles, angular velocities, body pose).
- Action dimension: 17 (joint torques).

Assessment: This is a more realistic and substantially larger dataset. The MuJoCo Humanoid-v4 environment is a standard benchmark. The 376-dimensional state space is representative of real humanoid robots. However, the paper does not specify how the episodes were generated (random actions? random perturbations from a reference trajectory? expert demonstrations?). This information is important for reproducibility.

**Recommendation:** Specify the MuJoCo episode generation procedure (e.g., "episodes were generated using random joint torques sampled from Uniform(-1, 1) at each timestep").

### 4.2 Hyperparameter Selection

The paper reports a grid search over lambda in {0, 0.1, 0.5, 1.0} and H in {4, 8, 16} (line 366). The results are presented in Table 6b. This is a well-documented hyperparameter selection process.

Other hyperparameters (D=128, N=16, L=4, lr=1e-3, weight_decay=1e-4) are justified by the ablation study (Table 6), which shows that these values represent a good balance between accuracy, parameters, and inference time.

**Assessment:** The hyperparameter selection is methodologically sound. The grid search is small but sufficient for the two key hyperparameters (lambda, H). The architecture hyperparameters are validated by ablation.

### 4.3 Statistical Analysis

**What is done well:**
- 3 random seeds with mean +/- std reported for all results.
- Paired t-tests for key comparisons.
- 95% CIs for critical comparisons.
- Explicit alpha=0.05 threshold.

**What could be improved:**
- No effect size reporting (Cohen's d).
- No equivalence testing for the Mamba-WM vs SSM-WM comparison.
- No correction for multiple comparisons (e.g., Bonferroni), though the number of comparisons is small enough that this is not critical.
- The MPC comparison (Table 7) does not report p-values for the tracking MSE differences (only for SSM vs LSTM: p=0.12).

### 4.4 Reproducibility

**Strengths:**
- Complete hyperparameter table (Table A1).
- Computational environment specified (GPU, CPU, OS, framework, CUDA version).
- Code available on GitHub (SEMHAQ/SSM-World-Model).
- Random seeds specified (42, 123, 456).
- Inference time measurement protocol described (10 warmup iterations, 20 measurements, report median).

**Gaps:**
- MuJoCo episode generation procedure not specified.
- The exact MuJoCo environment version (Humanoid-v4) is stated, but the specific MuJoCo version is not.
- No information on whether the code reproduces the exact numbers in the paper.

---

## 5. Assessment of Key Claims

### Claim 1: "SSM-WM achieves ~7x speedup over LSTM in batched inference (B=64)"

**Evidence:** Table 2: SSM-WM 3.8ms vs LSTM 27.8ms, ratio = 7.3x.
**Assessment:** Well-supported. The measurement protocol (GPU warmup, median of 20 runs) is appropriate.

### Claim 2: "SSM-WM achieves ~6% better prediction accuracy than LSTM on MuJoCo"

**Evidence:** Table 8: SSM-WM MSE=0.834 vs LSTM MSE=0.889, difference = 6.2%. p<0.01.
**Assessment:** Well-supported. The difference is statistically significant with proper testing.

### Claim 3: "SSM-WM-MPC achieves comparable tracking accuracy to LSTM-MPC"

**Evidence:** Table 7: SSM-WM-MPC MSE=0.0043 vs LSTM-MPC MSE=0.0032, difference = 34%. p=0.12.
**Assessment:** The claim of "comparable" is supported by the non-significant p-value, but the 34% difference is not negligible. With more seeds, this might become significant. The 95% CI [-0.0003, 0.0025] includes zero, which supports the claim, but also includes differences up to 78% of the LSTM baseline. The claim should be more carefully worded.

### Claim 4: "SSM-WM is comparable to Mamba-WM (gap < 2%)"

**Evidence:** Table 8: SSM-WM MSE=0.834 vs Mamba-WM MSE=0.821, difference = 1.6%. p=0.27.
**Assessment:** The <2% gap is accurate on MuJoCo. On synthetic data, the gap is 3.1% (1.32 vs 1.28, p=0.18). Both differences are non-significant. The claim is well-supported, but the paper should note that "comparable" means "not statistically different" rather than "identical."

### Claim 5: "Training converges within 20 epochs with no overfitting"

**Evidence:** Lines 667-671: training MSE converges at epoch 15, train-val gap <0.15x10^-3.
**Assessment:** Supported for the synthetic dataset. The MuJoCo training convergence is not reported (though Table 8b results suggest it also converges). Reporting MuJoCo training curves would strengthen this claim.

---

## 6. Remaining Issues from Round 12 (Devil's Advocate Perspective)

The Devil's Advocate raised several issues in Round 12. Let me assess their current status:

| Issue | Status | Assessment |
|-------|--------|------------|
| Speed advantage context (B=1 vs B=64) | ADDRESSED | Table 3 now reports B=1/8/32/64. The B=1 speedup is 2.3x, not 7x. Abstract leads with B=64. |
| Accuracy deficit positioning | ADDRESSED | Linear baseline in Table 1 shows SSM-WM (1.32) is closer to LSTM (0.85) than to linear (2.50). |
| MuJoCo missing baselines | FIXED | Table 8 has all 4 baselines. |
| MPC compute fairness | PARTIALLY ADDRESSED | The paper uses fixed iterations, which is defensible for real-time control. Should be explicitly stated. |
| Reference [22] | FIXED | Corrected to "GU A, GOEL K, RE C". |

---

## 7. Scoring

### Originality (78/100)

The paper's originality lies in the application-level contribution: assembling S4D diagonal SSM with Mamba-style gating for robot state prediction, integrating with MPC, and providing systematic validation. While the individual components (S4D, Mamba gating, FFT convolution) are established, their combination for humanoid robot world models with real-time MPC integration fills a genuine gap. The complexity analysis (Section 4.3) and the finding that SSM outperforms LSTM on complex dynamics (MuJoCo) while underperforming on simple dynamics (synthetic) provides an insightful contribution to the SSM vs LSTM debate.

The reference to [40] (Mamba policy) and [41] (SSM trajectory prediction) correctly positions this work within the growing SSM-for-robotics literature. The contribution is incremental but well-motivated and practically relevant.

### Methodological Rigor (84/100)

**Strengths:**
- Proper train/validation split (80/20) on both datasets.
- Hyperparameter selection via grid search with documented results (Table 6b).
- Comprehensive ablation study covering architecture, training loss, and hyperparameters.
- Statistical testing with p-values and CIs on MuJoCo results.
- Inference time measurement with warmup and multiple runs.
- MuJoCo ablation (Table 8b) validates component contributions on realistic dynamics.

**Weaknesses:**
- Synthetic dataset is small (100 episodes) and near-linear.
- Only 3 random seeds (acknowledged as limitation).
- No effect size reporting.
- MuJoCo episode generation procedure not specified.
- MPC fairness not explicitly discussed.

The methodological rigor is substantially improved from Round 12. The ablation design is particularly strong, covering three complementary dimensions.

### Evidence Sufficiency (84/100)

**Strengths:**
- Two complementary datasets (synthetic for controlled analysis, MuJoCo for realism).
- Complete baseline comparisons on both datasets.
- Multi-step prediction analysis (Table 4) is critical for MPC applications.
- Sequence length sensitivity analysis (Table 5) demonstrates scaling behavior.
- Training loss ablation (Table 6b) validates the multi-step loss design.
- Statistical significance testing on MuJoCo.

**Weaknesses:**
- Synthetic dataset limited in size and complexity.
- MuJoCo R^2 = 0.592 (41% unexplained variance).
- MPC experiment only on synthetic data (acknowledged).
- No cross-dataset generalization test.

The evidence is now sufficient for the paper's claims. The two-dataset design provides complementary validation, and the comprehensive ablation study supports each design choice.

### Argument Coherence (86/100)

The paper tells a coherent story:
1. Problem: lightweight world models needed for real-time humanoid robot control.
2. Solution: SSM-WM with S4D parameterization and Mamba gating.
3. Validation: speed advantage (7x on synthetic), accuracy advantage (6% on MuJoCo), MPC integration (5.1 Hz).
4. Limitations: MuJoCo MPC gap, R^2 ceiling, vision input not addressed.

The argument is logically structured and the claims are well-supported. The limitations section is honest and thorough. The discussion of dataset-dependent behavior (lines 848-854) provides a plausible explanation for the synthetic vs MuJoCo performance reversal, supported by the ablation evidence (gating contribution: 2.3% on synthetic vs 4.6% on MuJoCo).

The projected MuJoCo MPC frequency (~2.1 Hz, line 888) is a valuable addition that partially addresses the MuJoCo MPC gap.

### Writing Quality (84/100)

**Strengths:**
- Clear mathematical notation with consistent formatting.
- Well-structured tables with proper labeling.
- Concise abstract covering all key results.
- Comprehensive related work section.
- Honest and well-organized limitations section.

**Weaknesses:**
- Some notation inconsistencies (e.g., the SSM block notation uses both z and z' in equations 10-14).
- The discussion section (Section 5.8) is dense and could benefit from clearer subsection organization.
- The English abstract is fluent but slightly long.

---

## 8. Summary of Recommendations

### Must-Fix (for acceptance)

None. All critical issues from Round 12 have been addressed.

### Should-Fix (strong recommendations)

1. **Specify MuJoCo episode generation procedure** in Section 5.1 or Appendix B. This is important for reproducibility.

2. **Report 95% CIs for all pairwise comparisons** in Table 8 (not just SSM vs LSTM). The Mamba-WM vs SSM-WM CI would help readers assess whether the non-significant difference is truly negligible.

3. **Add a sentence in Section 5.7** clarifying that the MPC comparison uses fixed iterations (reflecting real-time constraints), and that LSTM-MPC would likely achieve better accuracy with more iterations.

### Nice-to-Have (minor suggestions)

4. Report Cohen's d for key comparisons in Tables 1 and 8.

5. Add MuJoCo training curves (analogous to Figure 2 for synthetic data) to support the convergence claim on the more realistic dataset.

6. Consider equivalence testing (TOST) for the Mamba-WM vs SSM-WM comparison, which would allow positive claims of equivalence rather than merely failing to reject the null hypothesis.

---

## 9. Dimension Scores

| Dimension | Weight | Score | Justification |
|-----------|--------|-------|---------------|
| Originality | 20% | 78 | Well-motivated application-level contribution; individual components not novel but assembly is insightful |
| Methodological Rigor | 25% | 84 | Comprehensive ablation, proper statistical testing, hyperparameter grid search; limited by small seeds and dataset |
| Evidence Sufficiency | 25% | 84 | Two complementary datasets, complete baselines, multi-step analysis; limited by synthetic dataset simplicity |
| Argument Coherence | 15% | 86 | Clear logical flow, honest limitations, plausible explanations for dataset-dependent behavior |
| Writing Quality | 15% | 84 | Clear notation, well-structured tables, comprehensive related work; minor density issues |
| **Weighted Average** | 100% | **83.2** | |

---

## 10. Overall Assessment

The paper has improved substantially from Round 12 (R1 score: 68.0) to Round 13 (R1 score: 83.2), a gain of 15.2 points. All six issues flagged in Round 12 have been addressed:

1. Reference [22] corrected.
2. Linear baseline included in Table 1.
3. Table 8 now has all 4 baselines.
4. Training convergence analysis added.
5. Multi-step loss ablation added as Table 6b.
6. Statistical significance reported on MuJoCo.

The paper now presents a well-scoped contribution with appropriate experimental validation. The SSM-WM architecture, while composed of established components, is well-motivated for the target application and validated with comprehensive ablation studies. The speed-accuracy trade-off is clearly quantified, and the MPC integration demonstrates practical relevance.

**Remaining gaps that prevent a score of 85+:**
- The synthetic dataset's near-linear dynamics limit its representativeness (W1).
- Three random seeds provide limited statistical power (W2).
- Effect sizes not reported (W3).

These are all addressable in future work and do not represent fundamental flaws in the current paper.

### Recommendation: Accept with Minor Revisions

The paper is suitable for publication at 控制理论与应用 (CTA). The remaining issues are minor and can be addressed in a final revision pass. The core contribution -- demonstrating that SSM-based world models can achieve competitive accuracy with significantly faster inference for humanoid robot state prediction -- is well-supported and relevant to the CTA community.

---

*Review completed: Round 13*
*Previous R1 score (Round 12): 68.0*
*Current R1 score (Round 13): 83.2*
*Improvement: +15.2 points*
*Target threshold: 85*
