# Peer Review Report — EIC

## Manuscript Information
- **Title**: 面向人形机器人状态预测的轻量级状态空间世界模型
- **Manuscript ID**: CTA-2026-XXXX
- **Review Date**: 2026-06-03
- **Review Round**: Round 14

---

## Reviewer Information

### Reviewer Role
Editor-in-Chief (EIC), 控制理论与应用 (Control Theory & Applications)

### Reviewer Identity
Prof. Zhang, Associate Editor for Control Theory & Applications. Expertise in model predictive control, robot dynamics, and real-time control systems. 15 years of editorial experience in control theory journals. Review preferences: practical relevance, methodological soundness, and clarity of presentation.

### Review Focus
Journal fit, originality, overall quality, significance for the control engineering community, and whether the paper advances the state of the art in a meaningful way.

---

## Overall Assessment

### Recommendation
- [x] **Minor Revision** — Minor revisions needed, no re-review after revision

### Confidence Score
4 — Mostly within my area of expertise, high confidence

### Summary Assessment
This paper proposes a lightweight world model based on diagonal state space models (SSM-WM) for humanoid robot state prediction, combining S4D-style diagonal SSM parameterization with Mamba-style gated blocks. The model is integrated into an MPC framework for online motion planning. The paper presents systematic experiments on synthetic and MuJoCo Humanoid datasets, comparing against LSTM, Transformer, and Mamba baselines.

The paper addresses a relevant problem for the control community—real-time world models for humanoid robots—and the MPC integration adds practical value. However, the originality is incremental rather than transformative, as the core architecture is a combination of existing techniques (S4D + Mamba gating). The experimental results show a significant accuracy gap on the synthetic dataset (SSM-WM MSE=2.72 vs LSTM MSE=1.06, a 157% disadvantage), though SSM-WM performs competitively on MuJoCo. There is a critical data inconsistency between the text and Table 1 that must be resolved before publication.

### Decision Rationale
The paper makes a solid contribution to the application of SSMs in robotics control, with thorough experimental validation including ablation studies, statistical significance testing, and MPC experiments. The speed advantage (7.3x over LSTM) is compelling for real-time applications. However, the data inconsistency between the narrative text (claiming MSE=1.32×10⁻³ for SSM-WM) and Table 1 (showing MSE=2.72×10⁻³) is a serious error that undermines credibility. Additionally, the R² score of 0.592 on MuJoCo indicates substantial room for improvement. Minor revision is recommended to fix the data inconsistency and strengthen the discussion of limitations.

---

## Strengths

### S1: Practical Relevance for Real-Time Control
The paper addresses a genuine need in the robotics community: lightweight world models that can run at control-loop frequencies. The MPC integration (Section 4.4) and the demonstration of 5.1Hz control frequency on synthetic data and 2.1Hz on MuJoCo are practically meaningful results. The paper correctly identifies that control frequency is a hard constraint in real-time systems (line 392).

### S2: Systematic Experimental Design
The experimental design is thorough, including: (1) two datasets with different complexity levels, (2) multiple baselines (LSTM, Transformer, Mamba), (3) ablation studies on architecture components, (4) training loss ablation, (5) sequence length sensitivity analysis, (6) statistical significance testing with effect sizes and confidence intervals, and (7) MPC control experiments. This level of experimental rigor is commendable for a control theory journal.

### S3: Clear Architecture Description
The SSM-WM architecture is clearly described in Section 4.1, with well-formatted equations and a diagram (Figure 1). The combination of S4D diagonal SSM with Mamba-style gating is explained step by step, making it accessible to control engineers who may not be deep learning specialists.

### S4: Honest Reporting of Limitations
The paper honestly reports that SSM-WM has lower accuracy than LSTM on the synthetic dataset (Section 5.2) and discusses the trade-off between speed and accuracy (Section 5.8). The limitations section (line 871-879) is candid about the gaps between simulation and real robot deployment.

---

## Weaknesses

### W1: Critical Data Inconsistency Between Text and Table
**Problem**: The narrative text in Section 5.2 states "SSM-WM的预测精度(MSE=1.32×10⁻³)" (line 483) and "LSTM-WM(2层)在预测精度上表现最优(MSE=0.85×10⁻³)" (line 481), but Table 1 shows SSM-WM MSE=2.72±0.03×10⁻³ and LSTM-WM MSE=1.06±0.00×10⁻³. These values are inconsistent.
**Why it matters**: This is a fundamental data integrity issue. Readers cannot trust the results if the text and tables contradict each other. This would likely be caught by any reviewer and could lead to rejection.
**Suggestion**: Carefully audit all numerical results in the paper. Ensure every number cited in the text matches the corresponding table. The 5-seed expanded dataset (SSM MSE=0.002719±0.000035, LSTM MSE=0.001061±0.000002) should be consistently reported throughout.
**Severity**: Critical

### W2: Low R² Score on MuJoCo Dataset
**Problem**: SSM-WM achieves R²=0.592 on the MuJoCo Humanoid dataset (Table 8), meaning 41% of the variance is unexplained. The paper discusses this briefly (line 860-864) but does not adequately address why a model claiming to be effective has such low explanatory power.
**Why it matters**: For a world model intended for MPC control, unexplained variance directly impacts control quality. The paper claims SSM-WM is "effective" for MPC, but the low R² raises questions about reliability in safety-critical scenarios.
**Suggestion**: (1) Add a per-joint analysis showing which joints have high vs. low R² to identify where the model struggles. (2) Discuss whether the low R² is due to inherent stochasticity in MuJoCo contact dynamics or model capacity limitations. (3) Consider reporting R² on a per-timestep basis to show whether accuracy degrades over longer horizons.
**Severity**: Major

### W3: Synthetic Dataset Bias
**Problem**: The synthetic dataset uses dynamics s_{t+1} = A·s_t + B·a_t + 0.1·tanh(s_t ⊙ a_t) (line 834), which is near-linear (nonlinear coefficient only 0.1). This inherently favors LSTM's nonlinear recurrence over SSM's linear state transition structure.
**Why it matters**: The 157% accuracy gap on synthetic data (MSE 2.72 vs 1.06) may be an artifact of the near-linear dynamics rather than a fundamental limitation of SSM-WM. The paper acknowledges this (line 834-839) but the discussion could be stronger.
**Suggestion**: (1) Add experiments with varying nonlinearity coefficients (e.g., 0.1, 0.5, 1.0) to show how the relative performance changes. (2) Explicitly state that the synthetic dataset results should not be generalized to real robot scenarios without MuJoCo validation.
**Severity**: Minor

### W4: Incomplete Statistical Reporting in Some Tables
**Problem**: While the main results (Table 1) report mean±std across 5 seeds, some tables (e.g., Table 6 ablation, Table 6b training loss ablation) report standard deviations that are very small relative to the mean differences, making it unclear whether the differences are statistically significant.
**Why it matters**: The paper claims rigorous statistical analysis (line 439-440) but does not report p-values or effect sizes for ablation studies, only for the main comparisons.
**Suggestion**: Add p-values for key ablation comparisons (e.g., full model vs. no gating) to support the claims about component contributions.
**Severity**: Minor

---

## Detailed Comments

### Title & Abstract
- Title is appropriate and clearly describes the contribution
- Abstract is comprehensive but could be more concise—currently 250+ words in English
- The abstract leads with methodology rather than the key finding; consider restructuring to lead with the main result (e.g., "achieves 7x speedup with competitive accuracy")

### Introduction
- Good motivation of the problem (lines 92-122)
- The research gap is clearly identified (lines 114-122)
- Contribution list (lines 125-129) is clear but could be more specific about what "first" means—clarify that the novelty is in the specific combination, not in each individual component

### Literature Review
- Coverage of SSM literature is adequate (Section 2.2)
- The MPC section (2.4) is relevant but could cite more recent work on learning-based MPC
- Missing discussion of other lightweight architectures for robot control (e.g., TinyML approaches)

### Methodology
- Architecture description is clear (Section 4.1)
- Training procedure is well-documented (Section 4.2)
- The choice of convolution mode over recurrent mode for MPC is justified (lines 246-249)

### Results
- Main results are presented clearly (Tables 1-2)
- The sequence length analysis (Table 5) is valuable
- **Critical issue**: Text-table inconsistency must be fixed (see W1)

### Discussion
- Good discussion of speed-accuracy trade-off (Section 5.8)
- The dataset difference analysis (lines 832-839) is insightful
- The "complementary experimental design" argument (lines 847-851) is well-reasoned

### Conclusion
- Conclusion accurately summarizes the findings
- Future work directions are reasonable

---

## Questions for Authors

1. **Data consistency**: The text reports SSM-WM MSE=1.32×10⁻³ but Table 1 shows 2.72×10⁻³. Which value is correct? Please audit all numerical results and provide a corrected version.

2. **MuJoCo R² interpretation**: With R²=0.592, 41% of variance is unexplained. Can you provide a per-joint breakdown showing which joints contribute most to the prediction error? This would help readers assess whether the model is suitable for their specific application.

3. **Deployment considerations**: The paper mentions that SSM-WM can switch to recurrent mode for O(1) single-step latency (line 249). Have you measured the actual latency in recurrent mode? This would strengthen the deployment argument.

4. **Comparison with Mamba-WM**: The paper shows SSM-WM and Mamba-WM have similar accuracy, but SSM-WM is "simpler to implement." Can you quantify the implementation complexity difference (e.g., lines of code, dependency count)?

---

## Minor Issues

### Language / Grammar
- Line 18: "作者名" (Author Name) is a placeholder—should be replaced with actual author names
- Line 67: "Author Name" placeholder in English abstract
- Some Chinese-English mixing could be more consistent

### Citation Format
- References [C1]-[C4] are Chinese journal articles with bilingual formatting—ensure this matches the journal's style guide
- Some references lack DOI numbers

### Figures and Tables
- Figure 1 (architecture diagram) should be referenced in the text before it appears
- Table 8 footnote mentions "首次运行因CUDA编译导致延迟较高(约19ms)"—this should be discussed in the main text

### Layout
- The paper uses two-column format, which is appropriate for the target journal

---

## Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 65 | Adequate | Combination of existing techniques (S4D + Mamba) applied to a new domain; incremental rather than transformative |
| Methodological Rigor (25%) | 72 | Adequate | Systematic experiments with ablations and statistical testing, but text-table inconsistency is a serious flaw |
| Evidence Sufficiency (25%) | 68 | Adequate | Good experimental coverage but low MuJoCo R² and near-linear synthetic dataset limit evidence strength |
| Argument Coherence (15%) | 75 | Strong | Clear logical flow from problem to solution to validation; minor gaps in connecting synthetic and MuJoCo results |
| Writing Quality (15%) | 70 | Adequate | Generally clear but has data inconsistency and placeholder author names |
| **Weighted Average** | **69.7** | **Minor Revision** | |

---

*End of EIC Review Report*
