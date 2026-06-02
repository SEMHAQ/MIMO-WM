# Peer Review Report — Reviewer 1 (Methodology)

## Manuscript Information
- **Title**: 面向人形机器人状态预测的轻量级状态空间世界模型
- **Manuscript ID**: CTA-2026-XXXX
- **Review Date**: 2026-06-03
- **Review Round**: Round 14

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 1 (Methodology)

### Reviewer Identity
Dr. Liu, Research Scientist at a robotics AI lab. Expertise in deep learning architectures for time-series prediction, state space models, and statistical methodology. Published 20+ papers on SSMs and sequence modeling. Particularly focused on reproducibility, statistical validity, and experimental design rigor.

### Review Focus
Research design rigor, statistical methodology, reproducibility, experimental validity, and whether the experimental setup fairly evaluates the proposed method against baselines.

---

## Overall Assessment

### Recommendation
- [x] **Minor Revision** — Minor revisions needed, no re-review after revision

### Confidence Score
5 — Completely within my area of expertise, I am very confident in my assessment

### Summary Assessment
This paper proposes SSM-WM, a lightweight world model combining S4D diagonal SSM with Mamba-style gated blocks for humanoid robot state prediction. The experimental design is generally sound, with two datasets, multiple baselines, ablation studies, and statistical significance testing. However, there are several methodological concerns: (1) a critical data inconsistency between the narrative text and Table 1, (2) incomplete statistical reporting for ablation studies, (3) the synthetic dataset's near-linear dynamics may bias results, and (4) the MPC experiment uses a fixed computation budget rather than fixed accuracy, which limits comparability. The 5-seed experimental protocol is appropriate, and the use of Cohen's d and 95% CIs is commendable. With minor methodological fixes, this paper could be suitable for publication.

---

## Strengths

### S1: Rigorous Statistical Reporting for Main Comparisons
The paper reports mean±std across 5 random seeds (42, 123, 456, 789, 1024), paired t-tests, 95% confidence intervals, and Cohen's d effect sizes for the main comparisons (lines 438-440). This level of statistical rigor is rare in the robotics/control literature and sets a good example. The effect size reporting (e.g., Cohen's d=3.2 for SSM-WM vs LSTM on synthetic data, line 483) helps readers assess practical significance beyond p-values.

### S2: Comprehensive Ablation Studies
The paper includes two types of ablation: (1) architecture ablation (Table 6) testing gating mechanism, residual connections, layer count, SSM state dimension, and hidden dimension, and (2) training loss ablation (Table 6b) testing multi-step loss weight λ and unroll steps H. This dual ablation provides strong evidence for the contribution of each design choice.

### S3: Sequence Length Sensitivity Analysis
The sequence length analysis (Table 5, T=16 to 512) is valuable for understanding the computational scaling behavior. The paper correctly identifies that SSM-WM has poor accuracy at short sequences (T=16, MSE=4.74) and recommends T≥32 for deployment (line 586). This practical guidance is useful for practitioners.

### S4: Honest Reporting of Negative Results
The paper honestly reports that SSM-WM has lower accuracy than LSTM on the synthetic dataset (MSE 2.72 vs 1.06, 157% gap) and does not hide this result. The discussion of the speed-accuracy trade-off (Section 5.8) is balanced and informative.

---

## Weaknesses

### W1: Critical Text-Table Data Inconsistency
**Problem**: The narrative text in Section 5.2 reports SSM-WM MSE=1.32×10⁻³ (line 483) and LSTM-WM MSE=0.85×10⁻³ (line 481), but Table 1 shows SSM-WM MSE=2.72±0.03×10⁻³ and LSTM-WM MSE=1.06±0.00×10⁻³. These are fundamentally different values.
**Why it matters**: This is a data integrity issue that would likely lead to rejection. If the text values are from a different experiment (e.g., different dataset or configuration), this must be clearly stated. If they are errors, they must be corrected.
**Suggestion**: (1) Audit all numerical values in the text against the tables. (2) If the text values are from a different configuration, add a table or footnote explaining the difference. (3) The 5-seed expanded dataset (SSM MSE=0.002719±0.000035) should be consistently reported.
**Severity**: Critical

### W2: Missing Statistical Tests for Ablation Studies
**Problem**: While the main comparisons (Table 1) include p-values and effect sizes, the ablation studies (Tables 6, 6b, 8b) report only mean±std without statistical significance tests. For example, removing the gating mechanism increases MSE from 1.32 to 1.35 (Table 6), but no p-value is reported to confirm this difference is statistically significant.
**Why it matters**: The paper claims "门控机制对模型性能贡献显著, 去除门控后MSE增加2.3%(p<0.05)" (line 626), but this p-value is not shown in the table. Without statistical tests, readers cannot assess whether the ablation differences are meaningful or due to random variation.
**Suggestion**: Add p-values (paired t-test) for key ablation comparisons: full model vs. no gating, full model vs. no residual, L=4 vs L=2, L=4 vs L=6. Report these in the table or in a supplementary table.
**Severity**: Major

### W3: Synthetic Dataset Dynamics May Bias Results
**Problem**: The synthetic dataset uses dynamics s_{t+1} = A·s_t + B·a_t + 0.1·tanh(s_t ⊙ a_t) (line 834), with a nonlinear coefficient of only 0.1. This makes the dynamics nearly linear, which inherently favors LSTM's nonlinear recurrence over SSM's linear state transition.
**Why it matters**: The 157% accuracy gap on synthetic data (MSE 2.72 vs 1.06) may be an artifact of the near-linear dynamics. The paper acknowledges this (lines 834-839) but does not provide experimental evidence to support the hypothesis.
**Suggestion**: (1) Add experiments with stronger nonlinearity (e.g., coefficient 0.5 or 1.0) to show how relative performance changes. (2) Alternatively, add a statement that the synthetic dataset results are specific to near-linear dynamics and may not generalize.
**Severity**: Minor

### W4: MPC Experiment Design Limitations
**Problem**: The MPC experiment uses a fixed computation budget (50 Adam iterations) rather than fixed accuracy (lines 391-392). The paper justifies this by saying "控制频率是硬约束" (line 392), but this design choice means the comparison is between methods with different accuracy levels, not methods achieving the same accuracy at different speeds.
**Why it matters**: If LSTM-MPC were given more iterations, it might achieve higher accuracy. The current design does not isolate the speed advantage from the accuracy advantage.
**Suggestion**: (1) Add a supplementary experiment showing accuracy vs. computation time curves for each method. (2) Report the accuracy achieved at fixed computation budgets (e.g., 10, 25, 50, 100 iterations) to show the Pareto frontier.
**Severity**: Minor

### W5: Inference Time Measurement Details
**Problem**: The paper states inference time is measured with "10次GPU预热迭代, 随后取20次测量的中位数" (line 436), but does not specify: (1) whether measurements include GPU-CPU transfer time, (2) whether the model is in eval() mode, (3) whether torch.no_grad() is used, (4) the GPU clock speed and thermal state during measurement.
**Why it matters**: Inference time measurements are highly sensitive to these details. Without them, results are not fully reproducible.
**Suggestion**: Add a paragraph in Section 5.1 specifying the inference measurement protocol in detail.
**Severity**: Minor

---

## Detailed Comments

### Title & Abstract
- Title accurately describes the contribution
- Abstract is comprehensive but could be more concise

### Introduction
- Problem motivation is clear (lines 92-122)
- Research gap is well-identified
- Contribution list is clear

### Literature Review
- SSM literature coverage is good (Section 2.2)
- MPC section could cite more recent learning-based MPC work

### Methodology
- Architecture description is clear and well-illustrated (Section 4.1)
- Training procedure is well-documented (Section 4.2)
- **Issue**: The choice of convolution mode for MPC is justified (lines 246-249) but the trade-off with recurrent mode could be discussed more

### Results
- Main results are presented clearly (Tables 1-2)
- **Critical issue**: Text-table inconsistency (see W1)
- Ablation studies are comprehensive but lack statistical tests (see W2)
- Sequence length analysis is valuable (Table 5)

### Discussion
- Speed-accuracy trade-off discussion is balanced (Section 5.8)
- Dataset difference analysis is insightful (lines 832-839)

### Conclusion
- Conclusion accurately summarizes findings
- Future work directions are reasonable

---

## Questions for Authors

1. **Text-table consistency**: Please clarify the discrepancy between the text (MSE=1.32×10⁻³ for SSM-WM) and Table 1 (MSE=2.72×10⁻³). Are these from different experiments?

2. **Ablation statistical significance**: Can you provide p-values for the key ablation comparisons (e.g., full model vs. no gating) to support the claims about component contributions?

3. **Inference measurement protocol**: Please specify whether inference times include GPU-CPU transfer, whether eval() and no_grad() are used, and the GPU thermal state during measurement.

4. **MPC iteration analysis**: Can you provide accuracy vs. iteration count curves for each MPC method to show the Pareto frontier?

---

## Minor Issues

### Language / Grammar
- Placeholder author names ("Author Name") must be replaced
- Some sentences are overly long (e.g., lines 391-392)

### Citation Format
- Reference [20] has a typo: "LOSCHILOV" should be "LOSHCHILOV"
- Some references lack DOI numbers

### Figures and Tables
- Table 6: Add p-values for key comparisons
- Table 8: Add standard deviation for R² column
- Figure 1: Ensure the diagram matches the actual architecture described in Section 4.1

### Layout
- No significant layout issues

---

## Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 62 | Adequate | Combination of existing techniques; incremental contribution |
| Methodological Rigor (25%) | 68 | Adequate | Good experimental design but text-table inconsistency and missing ablation statistics are flaws |
| Evidence Sufficiency (25%) | 70 | Adequate | Comprehensive experiments but synthetic dataset bias and low MuJoCo R² limit evidence |
| Argument Coherence (15%) | 74 | Strong | Clear logical flow; minor gaps in connecting results across datasets |
| Writing Quality (15%) | 68 | Adequate | Generally clear but has data inconsistency and formatting issues |
| **Weighted Average** | **68.4** | **Minor Revision** | |

---

*End of R1 Methodology Review Report*
