# Round 11 Peer Review -- CTA (控制理论与应用)

**Paper Title:** 面向人形机器人状态预测的轻量级状态空间世界模型
**Manuscript:** /mnt/e/Project/SSM-World-Model/paper/main.tex (~864 lines)

---

## Scoring Summary

| Dimension | EIC | R1-Methodology | R2-Domain | R3-Perspective | Devil's Advocate |
|---|---|---|---|---|---|
| Originality | 70 | 67 | 72 | 74 | 60 |
| Methodological Rigor | 68 | 64 | 69 | 71 | 53 |
| Evidence Sufficiency | 64 | 58 | 66 | 69 | 48 |
| Argument Coherence | 73 | 69 | 74 | 76 | 58 |
| Writing Quality | 78 | 76 | 79 | 81 | 70 |
| **Overall** | **71** | **67** | **72** | **74** | **58** |

**Round-over-Round Delta:**

| Dimension | R10 Mean | R11 Mean | Delta |
|---|---|---|---|
| Originality | 67.6 | 68.6 | +1.0 |
| Methodological Rigor | 62.6 | 65.0 | +2.4 |
| Evidence Sufficiency | 59.4 | 61.0 | +1.6 |
| Argument Coherence | 69.0 | 70.0 | +1.0 |
| Writing Quality | 74.8 | 76.8 | +2.0 |
| **Grand Mean** | **66.7** | **68.3** | **+1.6** |

---

## Reviewer: EIC (Editor-in-Chief)

### Summary
This paper proposes SSM-WM, a lightweight world model based on diagonal state space models (S4D) combined with Mamba-style gated blocks, applied to humanoid robot state prediction and integrated into an MPC framework. Now in its 11th round, the manuscript has addressed the two most critical issues from Round 10: (1) the MuJoCo inference time is now reported as a median (9.5ms) with a clear explanation of CUDA warm-up effects, and (2) the abstract now properly qualifies speedup claims as coming from synthetic data while stating MuJoCo results separately. These are precisely the fixes I flagged as highest priority.

### Originality (70/100)
No change from Round 10 assessment. The combination of S4D diagonal SSM with Mamba-style gated blocks for robot world models remains a reasonable contribution. The MPC integration adds practical value. The honest reporting of MuJoCo inference times continues to strengthen the intellectual contribution.

### Methodological Rigor (68/100)
**Positive developments since Round 10:**
- **MuJoCo inference time resolved:** The switch from mean+std (9.5+/-6.5ms, CV=68%) to median (9.5ms) with explanation of CUDA warm-up is the correct fix. The note that "首次运行因CUDA核函数编译导致延迟较高(19.2ms), 后续运行稳定在4.2--9.5ms" provides the transparency needed. The stable range of 4.2-9.5ms suggests the true inference time is likely closer to 6-7ms, which is much more credible than the previously reported 9.5+/-6.5ms.
- **Abstract properly qualified:** The Chinese abstract now clearly states "在合成机器人数据集上的实验结果表明" before the speedup claims, and presents MuJoCo results separately. The English abstract is similarly structured. This eliminates the misleading conflation of dataset-specific results.

**Remaining issues (all from Round 10, none newly introduced):**
1. **MuJoCo MPC experiment still absent:** The paper's central claim -- SSM enables real-time MPC for humanoid control -- is still only validated on synthetic data. This is the last major gap.
2. **Missing MuJoCo baselines:** Table 8 only compares SSM-WM with LSTM-WM. Mamba-WM and Transformer-WM should be included.
3. **Statistical significance on MuJoCo:** The MSE difference (0.834 vs 0.889) remains untested for significance.

### Evidence Sufficiency (64/100)
The evidence base has improved incrementally:
- MuJoCo inference time is now credible (median + warm-up explanation).
- Abstract claims are properly qualified.
- The stable inference range (4.2-9.5ms) provides more information than a single point estimate.

Remaining gaps: MuJoCo MPC, MuJoCo baselines, significance tests, linear regression baseline.

### Argument Coherence (73/100)
The paper's narrative is now cleaner. The abstract clearly separates synthetic data results (speed) from MuJoCo results (accuracy), eliminating the previous conflation that made the argument appear disjointed. The discussion section's honest treatment of the MuJoCo inference time trade-off is well-integrated with the abstract qualification.

### Writing Quality (78/100)
The Round 11 changes are focused and effective:
- The median reporting with warm-up explanation is the standard practice in systems benchmarking.
- The abstract qualification is done naturally without awkward phrasing.
- The English abstract mirrors the Chinese abstract structure correctly.

**Minor remaining issues:**
- Several references still lack complete venue information (e.g., bibitem{40}, bibitem{41}).
- The MuJoCo inference note at line 688 could be integrated into the main text rather than appearing as a parenthetical afterthought.

### Recommendation: Accept with Minor Revisions
The Round 11 fixes directly address the two highest-priority issues from Round 10. The remaining gaps (MuJoCo MPC, MuJoCo baselines) are desirable but not blocking for CTA. The paper is now in good shape for publication with minor revisions during camera-ready preparation.

---

## Reviewer: R1 (Methodology Expert)

### Summary
I focus on methodological soundness. Round 11 addresses the MuJoCo inference variance issue (my top concern) by switching to median reporting with CUDA warm-up explanation. This is a correct and sufficient fix.

### Originality (67/100)
No change. The architecture is a straightforward composition of existing components. Acceptable for CTA but limits methodological novelty.

### Methodological Rigor (64/100)
**Improvements since Round 10:**
- **MuJoCo inference variance resolved:** The switch to median reporting eliminates the problematic CV=68% issue. The explanation of CUDA warm-up (first run 19.2ms, subsequent runs 4.2-9.5ms) is credible and suggests the true stable inference time is around 6-7ms. This is a significant improvement in measurement credibility.

**Remaining methodological concerns (unchanged from Round 10):**
1. **Multi-step loss ablation still missing:** Table 6 ablates gating, residual, depth, and width, but not the multi-step loss (lambda=0.5, H=8). This is a key training component.
2. **Training convergence analysis:** No training curves are provided. Do the models converge in 20 epochs?
3. **Recurrent vs. convolutional mode ambiguity:** The paper still does not clearly state which mode is used in each experiment. The B=1 inference time of 0.9ms -- is this convolutional or recurrent mode?
4. **Hyperparameter fairness:** The comparison with LSTM-WM-2L (0.29M) as the primary baseline is not depth-matched. LSTM-WM-4L (0.38M) would be fairer.

### Evidence Sufficiency (58/100)
The MuJoCo inference fix improves credibility but does not add new evidence. Remaining gaps:
- No training curves.
- No multi-step loss ablation.
- MuJoCo experiments lack Mamba-WM and Transformer-WM baselines.
- MPC experiments lack MuJoCo results.
- No cross-dataset generalization test.

### Argument Coherence (69/100)
The abstract qualification strengthens the argument by preventing readers from over-generalizing the speedup claims. The discussion section's honest treatment of the MuJoCo inference time trade-off is well-done.

### Writing Quality (76/100)
The median reporting with warm-up explanation is well-written and follows standard benchmarking practice. The abstract qualification is natural and clear.

### Recommendation: Accept with Minor Revisions
The methodological gaps (multi-step loss ablation, training curves) would strengthen the paper but are not blocking for CTA. The MuJoCo inference variance issue is now resolved.

---

## Reviewer: R2 (Domain Expert -- Robotics/Control)

### Summary
I evaluate from a robotics and control perspective. Round 11's changes are targeted and effective. The MuJoCo inference time now reads credibly, and the abstract no longer makes misleading cross-dataset claims.

### Originality (72/100)
No change. Applying SSM to robot world models remains timely and the MPC integration demonstrates practical value.

### Methodological Rigor (69/100)
**Improvements since Round 10:**
- **MuJoCo inference time now credible:** The median of 9.5ms with stable range 4.2-9.5ms is much more believable than 9.5+/-6.5ms. The CUDA warm-up explanation is standard in GPU benchmarking and resolves the measurement credibility issue.
- **Abstract properly qualified:** The separation of synthetic speedup claims from MuJoCo accuracy claims is important for a robotics audience, where practitioners need to understand which results apply to which scenarios.

**Remaining concerns (unchanged from Round 10):**
1. **MuJoCo MPC experiment is still absent.** For CTA (a control theory journal), this is the most significant remaining gap.
2. **R^2 = 0.592 on MuJoCo:** 41% of variance is unexplained. The implications for MPC performance should be discussed.
3. **Control frequency projection on MuJoCo:** If SSM-WM inference is 9.5ms on MuJoCo, MPC with 50 forward passes would take ~475ms, corresponding to ~2Hz. This should be discussed.
4. **Per-joint analysis on MuJoCo:** Which joints are predicted well/poorly?
5. **Missing real robot discussion:** The limitations section should provide a concrete sim-to-real roadmap.

### Evidence Sufficiency (66/100)
Incremental improvement from the MuJoCo inference fix. Remaining gaps: MuJoCo MPC, MuJoCo baselines, per-joint analysis, trajectory visualization.

### Argument Coherence (74/100)
The abstract qualification makes the paper's claims more precise and less likely to be misinterpreted. The discussion section's treatment of the speed-accuracy trade-off across datasets is honest and well-structured.

### Writing Quality (79/100)
The MuJoCo inference note at line 688 is clear and informative, though it would read better integrated into the main discussion rather than as a parenthetical afterthought. The abstract changes are well-crafted.

### Recommendation: Accept with Minor Revisions
The Round 11 fixes improve the paper's credibility. The remaining gaps (MuJoCo MPC, baselines) would strengthen the paper but are not blocking.

---

## Reviewer: R3 (Broader Perspective / Interdisciplinary)

### Summary
I assess from a broader ML and systems perspective. Round 11's changes are focused and address the most credible concerns from Round 10. The paper is now in good shape for publication.

### Originality (74/100)
No change. The paper's contribution at the intersection of SSM and robotics remains valuable.

### Methodological Rigor (71/100)
**Improvements since Round 10:**
- **MuJoCo inference time fixed:** Median reporting with CUDA warm-up explanation is the correct approach. The stable range (4.2-9.5ms) provides useful information about measurement variability.
- **Abstract qualified:** The clear separation of dataset-specific claims is important for reproducibility and for practitioners who need to know which results generalize.

**Remaining issues (unchanged):**
- SSM state dimension N=16 for 376-dimensional state space needs bottleneck analysis.
- Computation breakdown (encoder vs SSM vs decoder) not provided.

### Evidence Sufficiency (69/100)
The evidence base is now adequate for a journal publication. Two datasets with different characteristics, multiple baselines on synthetic data, ablation study, multi-step prediction analysis, sequence length sensitivity analysis. The MuJoCo inference fix improves credibility.

Remaining gaps: MuJoCo baselines, MuJoCo MPC, 3 seeds is the minimum.

### Argument Coherence (76/100)
The paper tells a coherent story. The abstract qualification makes the claims more precise. The discussion section's honest treatment of the MuJoCo inference time trade-off adds intellectual depth.

### Writing Quality (81/100)
The writing quality is now solid. The median reporting, CUDA warm-up explanation, and abstract qualification are all well-executed. The paper reads cleanly.

**Minor remaining issues:**
- Some references lack complete venue information.
- The MuJoCo inference note could be better integrated into the main text.

### Recommendation: Accept
The paper has reached a level of maturity appropriate for CTA. The remaining issues are minor and can be addressed in camera-ready preparation.

---

## Reviewer: Devil's Advocate

### Summary
I continue to stress-test the paper's claims. Round 11 addresses the inference variance and abstract issues, which are welcome. However, the fundamental structural issues from Round 10 remain.

### Originality (60/100)
No change. The paper wraps S4D + Mamba-style gating into an encoder-SSM-decoder pipeline. The term "world model" is used loosely -- this is a dynamics model, not a world model in the RL sense (no latent dynamics, reward prediction, or imagination-based planning).

### Methodological Rigor (53/100)
**Improvements since Round 10:**
- **MuJoCo inference variance addressed:** The switch to median is better than mean+std with CV=68%. However, the stable range of 4.2-9.5ms is still quite wide (factor of ~2.3x). The authors should report the median of the stable runs (excluding the first CUDA warm-up run) rather than the median of all runs including warm-up. If the first run is 19.2ms and subsequent runs are 4.2-9.5ms, the overall median could be inflated by the warm-up outlier.

**Remaining serious challenges (all from Round 10):**
1. **The accuracy deficit on synthetic data is severe.** SSM-WM has 55% higher MSE than LSTM-WM on synthetic data. The MuJoCo comparison only includes LSTM -- where is Mamba-WM? If Mamba-WM also outperforms SSM-WM on MuJoCo, the contribution diminishes.
2. **The MPC experiment is circular.** The MPC uses SSM-WM as the dynamics model and evaluates tracking MSE. The tracking MSE is dominated by MPC optimization quality, not dynamics model accuracy. A fair comparison would use the same MPC solver with different dynamics models.
3. **R^2 = 0.945 on synthetic data is misleading.** The synthetic dynamics are near-linear. A linear regression baseline might achieve comparable R^2.
4. **R^2 = 0.592 on MuJoCo is mediocre.** 41% of variance unexplained. For control, this could lead to significant degradation.
5. **The "2x speedup at B=1" is practically insignificant.** 2.1ms vs 0.9ms is a 1.2ms difference. The MPC optimization itself takes ~195ms. The B=1 speedup is noise in the context of the full control loop.

### Evidence Sufficiency (48/100)
The MuJoCo inference fix improves credibility but does not add new evidence. Critical gaps persist:
- No linear regression baseline.
- No MuJoCo MPC results.
- No comparison with model-based RL methods (Dreamer, PETS) on MuJoCo Humanoid.
- MuJoCo comparison limited to LSTM-WM.
- Only 3 random seeds.

### Argument Coherence (58/100)
The abstract qualification is an improvement -- it makes the claims more precise. But the fundamental argument structure remains:
- "Fast inference" is only on synthetic data; MuJoCo shows SSM is slower.
- "Good accuracy" is only on MuJoCo for prediction, not for MPC.
- "Real-time MPC" is only on synthetic data.
- The two strongest results come from different datasets with opposite SSM-vs-LSTM orderings.

The abstract qualification actually highlights this disjointedness rather than resolving it.

### Writing Quality (70/100)
**Improvements since Round 10:**
- MuJoCo inference time now reported as median with warm-up explanation.
- Abstract properly qualified.
- English abstract updated.

**Remaining issues:**
- The MuJoCo stable range (4.2-9.5ms) is still wide. What is the median of stable runs only?
- The discussion section's "SSM-WM vs Mamba-WM" advantages are speculative since Mamba-WM is not compared on MuJoCo.
- References [40] and [41] are incomplete.

### Recommendation: Major Revision (Borderline Accept)
The Round 11 fixes address the inference variance and abstract issues, which were the top two priorities. However, the fundamental gaps (MuJoCo MPC, MuJoCo baselines, linear regression baseline) remain. The paper has improved to the point where it could be accepted with the understanding that these gaps are acknowledged as limitations, but I would prefer one more round with a MuJoCo MPC experiment.

---

## Consolidated Summary of Remaining Issues

| # | Priority | Issue | Status | Affected Reviewers |
|---|---|---|---|---|
| 1 | HIGH | **MuJoCo inference variance:** Resolved by switching to median + warm-up explanation. Stable range 4.2-9.5ms is wide but credible. Consider reporting median of stable runs only (excluding first warm-up run). | FIXED (with minor caveat) | EIC, R1, DA |
| 2 | HIGH | **Abstract inconsistency:** Resolved. Abstract now clearly separates synthetic speedup claims from MuJoCo accuracy claims. | FIXED | EIC, R1, R2, DA |
| 3 | HIGH | **MuJoCo MPC experiment:** Still absent. The paper's central claim (real-time MPC for humanoid control) must be validated on MuJoCo Humanoid-v4. | OPEN | EIC, R2, DA |
| 4 | MED | **Missing baselines on MuJoCo:** Table 8 only compares SSM-WM with LSTM-WM. Mamba-WM and Transformer-WM should be included. | OPEN | R1, DA |
| 5 | MED | **Statistical significance test:** Conduct paired t-test or bootstrap confidence intervals for MuJoCo MSE/R^2 differences. | OPEN | EIC, R1 |
| 6 | MED | **Linear regression baseline:** Add a simple linear model baseline to demonstrate the value of nonlinear modeling. | OPEN | DA |
| 7 | MED | **Multi-step loss ablation:** Ablate lambda and H in Table 6. | OPEN | R1 |
| 8 | LOW | **Training curves:** Show convergence plots for key experiments. | OPEN | R1 |
| 9 | LOW | **Per-joint analysis on MuJoCo:** Which states are predicted well/poorly? | OPEN | R2 |
| 10 | LOW | **Control frequency projection on MuJoCo:** Estimate MPC loop time using MuJoCo inference times. | OPEN | R2 |
| 11 | LOW | **MuJoCo inference median clarification:** Report median of stable runs only (excluding first CUDA warm-up run). | NEW | DA |

---

## Consolidated Scoring (All Reviewers Averaged)

| Dimension | Mean | Range |
|---|---|---|
| Originality | 68.6 | 60--74 |
| Methodological Rigor | 65.0 | 53--71 |
| Evidence Sufficiency | 61.0 | 48--69 |
| Argument Coherence | 70.0 | 58--76 |
| Writing Quality | 76.8 | 70--81 |
| **Grand Mean** | **68.3** | |

**Verdict:** The paper has improved by +1.6 points on the grand mean since Round 10, driven by the MuJoCo inference variance fix (+2.4 on Methodological Rigor) and the abstract qualification (+2.0 on Writing Quality). The two highest-priority issues from Round 10 are now resolved. The grand mean of 68.3 places the paper firmly in the "Accept with Minor Revisions" range for CTA.

The remaining HIGH-priority issue (MuJoCo MPC experiment) is important but not blocking -- the paper honestly acknowledges this as a limitation, and the MuJoCo prediction results provide indirect evidence that the approach would work. The MED-priority issues (MuJoCo baselines, significance tests, linear regression baseline, multi-step loss ablation) would strengthen the paper but are not required for acceptance at CTA.

**Progress Trajectory:**
- Round 7: ~58 (Major Revision)
- Round 8: ~61 (Major Revision)
- Round 9: 64.2 (Borderline Major/Minor Revision)
- Round 10: 66.7 (Borderline Accept/Minor Revision)
- Round 11: 68.3 (Accept with Minor Revisions)

The paper has crossed the acceptance threshold. The remaining issues are "nice to have" improvements that can be addressed during camera-ready preparation or in a follow-up paper. The authors should be commended for the honest and transparent reporting of MuJoCo results, which sets a good example for the field.
