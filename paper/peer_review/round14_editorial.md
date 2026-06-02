# Editorial Decision

## Manuscript Information
- **Title**: 面向人形机器人状态预测的轻量级状态空间世界模型
- **Manuscript ID**: CTA-2026-XXXX
- **Submission Date**: 2026-06-03
- **Decision Date**: 2026-06-03
- **Review Round**: Round 14

---

## Decision *

### Minor Revision

---

## Reviewer Summary

| Reviewer | Role | Recommendation | Confidence | Weighted Score |
|----------|------|---------------|------------|----------------|
| EIC | 控制理论与应用 Editor | Minor Revision | 4 | 69.7 |
| Reviewer 1 | Methodology Expert | Minor Revision | 5 | 68.4 |
| Reviewer 2 | Domain Expert (Robotics) | Minor Revision | 5 | 71.2 |
| Reviewer 3 | Cross-Disciplinary (Efficient Computing) | Minor Revision | 4 | 72.4 |
| Devil's Advocate | Adversarial Critic | Major Revision | — | 62.8 |

**Average Score**: 68.9 (Minor Revision range: 65-79)

---

## Consensus Analysis *

### Points of Agreement (Consensus)

**[CONSENSUS-5]** (All reviewers agree):
1. **Data inconsistency is Critical**: All 5 reviewers identified the text-table inconsistency (text says MSE=1.32×10⁻³, Table 1 shows 2.72×10⁻³) as a critical issue that must be fixed. This is the highest-priority revision.
2. **MPC integration is a strength**: All reviewers agree that the MPC integration (Section 4.4) and real-time control demonstration (5.1Hz/2.1Hz) are genuine contributions that go beyond typical ML papers.
3. **Statistical rigor is commendable**: The use of 5 seeds, paired t-tests, confidence intervals, and Cohen's d is appreciated by all reviewers.

**[CONSENSUS-4]** (4/5 reviewers agree):
1. **MuJoCo R² of 0.592 is a concern**: EIC, R1, R2, and DA agree that 41% unexplained variance limits practical applicability. R3 acknowledges this but focuses more on computational efficiency.
2. **Synthetic dataset may bias results**: EIC, R1, R2, and DA agree that the near-linear dynamics (coefficient 0.1) may favor LSTM. R3 notes this but focuses on other issues.
3. **"First" claim needs qualification**: R2, R3, and DA agree that the "first" claim is overstated. EIC and R1 note this is a minor issue.

**[CONSENSUS-3]** (3/5 reviewers agree):
1. **Missing embedded hardware evaluation**: R3 and DA strongly advocate for embedded benchmarks. R1 notes it as a minor issue. EIC and R2 do not emphasize it.

### Points of Disagreement

**Disagreement 1: Severity of the accuracy gap on synthetic data**
- **EIC/R1 view**: The 157% accuracy gap (MSE 2.72 vs 1.06) is a significant limitation that should be discussed more prominently.
- **DA view**: The gap is "conveniently downplayed" and the synthetic dataset discussion involves "circular reasoning."
- **R2 view**: The synthetic dataset results are less important than MuJoCo results; the paper should focus on MuJoCo.
- **Disagreement type**: Severity disagreement
- **Editor's Resolution**: The synthetic dataset results should be presented honestly but contextualized. The paper should explicitly state that the synthetic dataset has near-linear dynamics and that the results may not generalize to real robot scenarios. The MuJoCo results should be given equal or greater weight.
- **Resolution Rationale**: The synthetic dataset serves a specific purpose (testing computational efficiency and MPC integration), not as a proxy for real robot dynamics. The paper should be clear about this.

**Disagreement 2: Importance of embedded hardware evaluation**
- **R3/DA view**: Embedded hardware benchmarks are essential for the paper's claims to be relevant.
- **EIC/R1/R2 view**: GPU benchmarks are sufficient for a first paper; embedded evaluation can be future work.
- **Disagreement type**: Perspective difference
- **Editor's Resolution**: The paper should acknowledge that all benchmarks are GPU-specific and discuss expected performance on embedded hardware. Actual embedded benchmarks can be future work.
- **Resolution Rationale**: Requiring embedded benchmarks would significantly delay publication. The paper's contribution is in the architecture and MPC integration, not in embedded deployment.

**Disagreement 3: Originality assessment**
- **DA view**: Originality is "Weak" (58) — the "first" claim is overstated.
- **EIC/R1/R2/R3 view**: Originality is "Adequate" (62-70) — novel application domain, even if architecture is incremental.
- **Disagreement type**: Severity disagreement
- **Editor's Resolution**: Originality is "Adequate" (65-68 range). The novelty is in the application domain (SSM for robot world models) and the MPC integration, not in the architecture itself. The "first" claim should be reworded to focus on the application rather than the architecture combination.
- **Resolution Rationale**: The DA's assessment is too harsh. Applying SSMs to robot world models with MPC integration is a meaningful contribution, even if the architecture is a combination of existing techniques.

---

## Decision Rationale *

The paper proposes SSM-WM for humanoid robot state prediction, combining S4D diagonal SSM with Mamba-style gated blocks and integrating it into an MPC framework. All reviewers agree that the paper makes a solid contribution to the field, with practical relevance for real-time robot control. The MPC integration is a genuine strength that distinguishes this paper from typical ML papers.

However, there is one critical issue that must be fixed: the data inconsistency between the narrative text and Table 1. The text reports SSM-WM MSE=1.32×10⁻³ but Table 1 shows 2.72×10⁻³. This is a fundamental data integrity issue that undermines the paper's credibility. All 5 reviewers identified this as Critical.

Additionally, the paper has several major issues: (1) the low R² on MuJoCo (0.592) should be more honestly discussed, (2) the synthetic dataset discussion should avoid circular reasoning, and (3) the "first" claim should be reworded. These issues are fixable with minor revisions.

The Devil's Advocate raised the concern about hardware-specific speed claims. While this is a valid concern, requiring embedded hardware benchmarks would significantly delay publication. The paper should acknowledge this limitation and discuss expected performance on embedded hardware.

Overall, the paper is suitable for publication in 控制理论与应用 after addressing the critical data inconsistency and the major issues. The paper's contributions (SSM for robot world models, MPC integration, systematic experiments) are valuable for the control engineering community.

---

## Required Revisions * (Must Fix)

| # | Revision Item | Source Reviewer | Severity | Section | Estimated Effort |
|---|--------------|----------------|----------|---------|-----------------|
| R1 | Fix text-table data inconsistency | All 5 | Critical | Section 5.2 | 1 day |
| R2 | Add statistical tests for ablation studies | R1 | Major | Section 5.6 | 2 days |
| R3 | Improve MuJoCo R² discussion | EIC, R1, R2, DA | Major | Section 5.8 | 1 day |
| R4 | Revise synthetic dataset discussion to avoid circular reasoning | DA | Major | Section 5.8 | 1 day |
| R5 | Reword "first" claim | DA, R2, R3 | Major | Abstract, Introduction | 1 day |

### Required Item Details

**R1: Fix Text-Table Data Inconsistency**
- **Problem**: Text reports SSM-WM MSE=1.32×10⁻³ (line 483) but Table 1 shows 2.72×10⁻³. Similarly, LSTM-WM MSE is 0.85×10⁻³ in text (line 481) but 1.06×10⁻³ in table.
- **Source**: All 5 reviewers identified this as Critical.
- **Requirement**: Audit all numerical values in the text against the tables. Ensure every number cited in the text matches the corresponding table. The 5-seed expanded dataset (SSM MSE=0.002719±0.000035, LSTM MSE=0.001061±0.000002) should be consistently reported throughout.
- **Acceptance criteria**: All numerical values in the text match the corresponding tables. No contradictions between narrative and data.

**R2: Add Statistical Tests for Ablation Studies**
- **Problem**: Ablation studies (Tables 6, 6b, 8b) report only mean±std without statistical significance tests.
- **Source**: R1 (Methodology)
- **Requirement**: Add p-values (paired t-test) for key ablation comparisons: full model vs. no gating, full model vs. no residual, L=4 vs L=2, L=4 vs L=6.
- **Acceptance criteria**: p-values are reported for at least 4 key ablation comparisons.

**R3: Improve MuJoCo R² Discussion**
- **Problem**: R²=0.592 on MuJoCo means 41% variance unexplained, but the paper does not adequately discuss implications.
- **Source**: EIC, R1, R2, DA
- **Requirement**: (1) Add per-joint R² analysis if possible. (2) Discuss whether low R² is due to model limitations or inherent stochasticity. (3) Compare with other world models' R² on similar tasks.
- **Acceptance criteria**: The discussion addresses why R² is low and what it means for practical applicability.

**R4: Revise Synthetic Dataset Discussion**
- **Problem**: The paper designed the synthetic dataset with near-linear dynamics, then uses this to explain away poor performance. This is circular reasoning.
- **Source**: DA
- **Requirement**: (1) Explicitly state that the synthetic dataset has near-linear dynamics and results may not generalize. (2) Focus on MuJoCo results as the primary validation. (3) Add experiments with varying nonlinearity if possible.
- **Acceptance criteria**: The synthetic dataset discussion avoids circular reasoning and clearly states limitations.

**R5: Reword "First" Claim**
- **Problem**: The claim "首次将S4D风格的对角SSM参数化与Mamba风格的门控块结构相结合" implies novelty in the architecture combination, which is overstated.
- **Source**: DA, R2, R3
- **Requirement**: Reword to focus on the application novelty (first systematic study of SSM for robot world models with MPC integration) rather than architecture novelty.
- **Acceptance criteria**: The "first" claim is reworded to accurately reflect the contribution.

---

## Suggested Revisions (Should Fix)

| # | Revision Item | Source Reviewer | Priority | Section | Expected Improvement |
|---|--------------|----------------|----------|---------|---------------------|
| S1 | Add embedded hardware discussion | R3, DA | P2 | Section 5.1 | Strengthens deployment relevance |
| S2 | Add literature on recent robot learning | R2 | P2 | Section 2 | Better positioning |
| S3 | Replace placeholder author names | EIC | P3 | Title page | Professional appearance |
| S4 | Fix reference typo (LOSCHILOV → LOSHCHILOV) | R1 | P3 | References | Correctness |
| S5 | Add energy consumption discussion | R3 | P3 | Section 5.8 | Broader impact |

---

## Revision Roadmap *

### Priority 1 — Critical Fixes (Estimated total effort: 3 days)
- [ ] R1: Fix text-table data inconsistency — audit all numerical values
- [ ] R2: Add statistical tests for ablation studies (p-values for 4+ comparisons)
- [ ] R3: Improve MuJoCo R² discussion (per-joint analysis, implications)
- [ ] R4: Revise synthetic dataset discussion to avoid circular reasoning
- [ ] R5: Reword "first" claim in abstract and introduction

### Priority 2 — Content Supplementation (Estimated total effort: 2 days)
- [ ] S1: Add embedded hardware discussion in Section 5.1
- [ ] S2: Add recent robot learning literature in Section 2

### Priority 3 — Text and Formatting (Estimated total effort: 1 day)
- [ ] S3: Replace placeholder author names
- [ ] S4: Fix reference typo (LOSCHILOV → LOSHCHILOV)
- [ ] S5: Add energy consumption discussion
- [ ] Language polishing

### Total Estimated Effort
- **Minor Revision**: 4-6 days

---

## Revision Deadline

- **Recommended deadline**: 2026-06-17 (2 weeks)
- **Basis**: Minor Revision, 2-4 weeks standard
- **Extension policy**: If extension is needed, notify 1 week before the deadline

---

## Response Letter Instructions

Please use the format in `templates/revision_response_template.md` to respond to every reviewer comment item by item.

**Must include**:
1. Response and revision description for each Required Revision (R1-R5)
2. Response for each Suggested Revision (S1-S5, adopted or reason for not adopting)
3. Change markup (mark all changes in the revised manuscript with color or track changes)
4. Cross-reference table of new page numbers/paragraphs

---

## Closing

We invite you to submit a revised version of your manuscript, addressing the points raised by the reviewers. The reviewers have identified a critical data inconsistency that must be fixed, along with several major issues that should be addressed. The paper's contributions (SSM for robot world models, MPC integration, systematic experiments) are valuable, and we look forward to receiving your revision within 2 weeks.

---

## Appendix: Reviewer Score Summary

| Reviewer | Originality | Methodology | Evidence | Coherence | Writing | Weighted Avg |
|----------|------------|-------------|----------|-----------|---------|--------------|
| EIC | 65 | 72 | 68 | 75 | 70 | 69.7 |
| R1 Methodology | 62 | 68 | 70 | 74 | 68 | 68.4 |
| R2 Domain | 68 | 72 | 70 | 76 | 72 | 71.2 |
| R3 Perspective | 70 | 72 | 72 | 76 | 72 | 72.4 |
| Devil's Advocate | 58 | 62 | 60 | 68 | 68 | 62.8 |
| **Average** | **64.6** | **69.2** | **68.0** | **73.8** | **70.0** | **68.9** |

---

*End of Editorial Decision*
