# Peer Review Report — Reviewer 2 (Domain)

## Manuscript Information
- **Title**: 面向人形机器人状态预测的轻量级状态空间世界模型
- **Manuscript ID**: CTA-2026-XXXX
- **Review Date**: 2026-06-03
- **Review Round**: Round 14

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 2 (Domain Expert)

### Reviewer Identity
Prof. Chen, Full Professor at a top robotics institute. Expertise in robot dynamics, model predictive control, and humanoid robot locomotion. Published 50+ papers on robot control and world models. Deep knowledge of MuJoCo simulation and sim-to-real transfer. Particularly focused on theoretical framework, literature completeness, and domain contribution.

### Review Focus
Literature coverage and positioning, theoretical framework appropriateness, domain contribution significance, and whether the paper advances the field of robot world models in a meaningful way.

---

## Overall Assessment

### Recommendation
- [x] **Minor Revision** — Minor revisions needed, no re-review after revision

### Confidence Score
5 — Completely within my area of expertise, I am very confident in my assessment

### Summary Assessment
This paper proposes SSM-WM for humanoid robot state prediction, combining S4D diagonal SSM with Mamba-style gated blocks and integrating it into an MPC framework. The domain contribution is clear: applying SSMs to robot world models is an underexplored area, and the paper provides systematic validation on both synthetic and MuJoCo Humanoid datasets. The literature review covers the key SSM papers but could be strengthened with more recent robot learning work. The theoretical framework is sound, though the connection between SSM theory and robot dynamics could be deeper. The MuJoCo results (SSM-WM outperforming LSTM by 6%) are promising, but the low R² (0.592) raises questions about practical applicability. The MPC integration is a strong domain contribution, demonstrating real-time control capability.

---

## Strengths

### S1: Novel Application Domain for SSMs
The paper makes a clear contribution by applying SSMs to robot world models, an area where SSMs have been underexplored. While SSMs have been successful in language modeling and time-series forecasting, their application to robot state prediction with MPC integration is novel. The paper correctly identifies this gap (lines 114-119) and fills it with a systematic study.

### S2: MuJoCo Humanoid Validation
The use of MuJoCo Humanoid-v4 (376-dimensional state, 17-dimensional action) is a strong validation choice. This environment includes realistic contact dynamics, friction, and gravity effects, making it much more representative of real robot scenarios than synthetic datasets. The paper shows SSM-WM outperforms LSTM by 6.2% (p<0.01, Cohen's d=1.8) on this dataset, which is a meaningful domain result.

### S3: MPC Integration with Real-Time Demonstration
The MPC integration (Section 4.4) is a significant domain contribution. The paper demonstrates: (1) SSM-WM-MPC achieves 5.1Hz on synthetic data and 2.1Hz on MuJoCo, (2) tracking accuracy is comparable to LSTM-MPC (p=0.12 and p=0.18 respectively), and (3) the speed advantage (6-7x) is maintained in the control loop. This closes the gap between world model research and practical robot control.

### S4: Comprehensive Baseline Comparison
The paper compares against LSTM (2-layer and 4-layer), Transformer, and Mamba baselines. Including both 2-layer and 4-layer LSTM is a good design choice, as it shows that deeper LSTM does not always improve performance (Table 1, line 485). The Mamba comparison is particularly relevant as it validates the diagonal SSM approach against the state-of-the-art selective SSM.

---

## Weaknesses

### W1: Data Inconsistency Between Text and Table
**Problem**: The text reports SSM-WM MSE=1.32×10⁻³ (line 483) but Table 1 shows 2.72×10⁻³. Similarly, LSTM-WM MSE is reported as 0.85×10⁻³ in the text (line 481) but 1.06×10⁻³ in the table.
**Why it matters**: This is a fundamental data integrity issue. Domain experts will immediately notice this discrepancy and question the validity of the results.
**Suggestion**: Audit all numerical values and ensure consistency throughout the paper.
**Severity**: Critical

### W2: Low R² on MuJoCo Limits Practical Claims
**Problem**: SSM-WM achieves R²=0.592 on MuJoCo (Table 8), meaning 41% of variance is unexplained. The paper claims this is "基本可用" for trajectory tracking (line 862), but for a world model intended for MPC control, this level of accuracy may be insufficient for many practical applications.
**Why it matters**: In safety-critical robot control, unexplained variance can lead to unpredictable behavior. The paper should more honestly assess the practical limitations of this accuracy level.
**Suggestion**: (1) Provide a per-joint R² breakdown to identify which joints are poorly predicted. (2) Discuss whether the low R² is due to model limitations or inherent stochasticity in MuJoCo contact dynamics. (3) Compare with the R² achieved by other world models on similar tasks.
**Severity**: Major

### W3: Literature Gap on Recent Robot Learning Work
**Problem**: The literature review (Section 2) covers SSM and MPC literature well but misses several recent relevant works: (1) diffusion models for robot dynamics, (2) graph neural networks for robot state prediction, (3) neural ODE for continuous dynamics, (4) recent sim-to-real transfer methods for humanoid robots.
**Why it matters**: Positioning the paper relative to these alternative approaches would strengthen the motivation for SSM-based world models.
**Suggestion**: Add a paragraph in Section 2.1 discussing these alternative approaches and explaining why SSM is preferred.
**Severity**: Minor

### W4: Missing Comparison with Analytical Models
**Problem**: The paper compares SSM-WM only with learned models (LSTM, Transformer, Mamba) but does not compare with analytical models (e.g., rigid body dynamics from MuJoCo). This limits the ability to assess whether the learned model is actually better than the ground-truth physics.
**Why it matters**: If MuJoCo's built-in physics model is more accurate, the motivation for learned world models is weakened.
**Suggestion**: Add a brief discussion explaining why analytical models are not included as baselines (e.g., they require exact knowledge of system parameters, they don't generalize to new environments, etc.).
**Severity**: Minor

---

## Detailed Comments

### Title & Title
- Title accurately describes the contribution
- "轻量级" (lightweight) is well-justified by the 0.14M parameter count

### Abstract
- Abstract is comprehensive but could lead with the key finding (7x speedup)
- The MuJoCo MPC result (2.1Hz) should be highlighted more prominently

### Introduction
- Problem motivation is strong (lines 92-122)
- The "three unique challenges" of humanoid state prediction (lines 120-122) are well-articulated
- Contribution list is clear

### Literature Review
- SSM coverage is good (Section 2.2)
- MPC section is relevant (Section 2.4)
- **Gap**: Missing recent robot learning literature (see W3)

### Methodology
- Architecture description is clear (Section 4.1)
- The S4D + Mamba combination is well-motivated
- Training procedure is well-documented

### Results
- Main results are clear (Tables 1-2)
- **Critical issue**: Text-table inconsistency (see W1)
- MuJoCo results are promising but R² is low (see W2)
- MPC results are strong (Tables 7, 9)

### Discussion
- Speed-accuracy trade-off is well-discussed
- Dataset difference analysis is insightful
- Limitations section is honest

### Conclusion
- Conclusion accurately summarizes findings
- Future work directions are relevant

---

## Questions for Authors

1. **R² interpretation**: Can you provide a per-joint R² breakdown for MuJoCo? This would help readers understand which joints are well-predicted and which are not.

2. **Comparison with analytical models**: Have you considered comparing with MuJoCo's built-in physics model? If not, can you explain why learned models are preferred?

3. **Sim-to-real gap**: The paper mentions the gap between simulation and real robots (line 873). What specific challenges do you expect in real-robot deployment, and how does SSM-WM's architecture help address them?

4. **Scaling to higher DoF**: The MuJoCo Humanoid has 17 action dimensions. Have you considered testing on higher-DoF robots (e.g., 30+ DoF) to assess scalability?

---

## Minor Issues

### Literature
- Add recent works on diffusion models for dynamics (e.g., Diffusion Policy, 2023)
- Add recent works on neural ODE for continuous dynamics
- Consider citing DreamerV3 for world model comparison

### Figures and Tables
- Table 8: R² standard deviation is reported (±0.014) but not discussed
- Figure 8 (radar chart): Ensure all axes are clearly labeled

### Terminology
- "世界模型" (world model) is used broadly; clarify whether this refers to dynamics models or includes perception

---

## Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 68 | Adequate | Novel application domain for SSMs; architecture is combination of existing techniques |
| Methodological Rigor (25%) | 72 | Adequate | Systematic experiments but text-table inconsistency is a flaw |
| Evidence Sufficiency (25%) | 70 | Adequate | Good MuJoCo validation but low R² limits practical claims |
| Argument Coherence (15%) | 76 | Strong | Clear logical flow from problem to solution to validation |
| Writing Quality (15%) | 72 | Adequate | Generally clear but has data inconsistency |
| Literature Integration | 68 | Adequate | Good SSM coverage but missing recent robot learning works |
| **Weighted Average** | **71.2** | **Minor Revision** | |

---

*End of R2 Domain Review Report*
