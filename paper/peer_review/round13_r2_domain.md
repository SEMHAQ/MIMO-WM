# Round 13 Peer Review -- R2 (Domain Expert: Robotics & Control Theory)

**Paper Title:** 面向人形机器人状态预测的轻量级状态空间世界模型
**Journal:** 控制理论与应用 (Control Theory & Applications)
**Manuscript:** /mnt/e/Project/SSM-World-Model/paper/main.tex (~1038 lines)
**Reviewer:** R2 -- Domain Expert in Robotics Dynamics, Model Predictive Control, and State Estimation
**Review Round:** 13

---

## 1. Summary

This paper proposes SSM-WM, a lightweight world model based on diagonal state space models (S4D parameterization) combined with Mamba-style gated blocks for humanoid robot state prediction. The model is further integrated into an MPC framework for online motion planning. Experiments span a synthetic robot dataset and a MuJoCo Humanoid-v4 simulation dataset, with comparisons against LSTM, Transformer, and Mamba baselines.

Since Round 12, the authors have addressed four key issues I raised:
1. Table 8 now includes all four baselines (LSTM, Transformer, Mamba, SSM-WM), providing a complete MuJoCo comparison.
2. A MuJoCo MPC frequency projection (~2.1 Hz) is now stated in the abstract and discussion.
3. The R^2 = 0.592 discussion is expanded with task-specific applicability analysis (trajectory tracking vs. aggressive manipulation).
4. Statistical significance tests are now reported for MuJoCo results (p-values for pairwise comparisons).

These are meaningful improvements that substantially strengthen the domain relevance and completeness of the experimental evaluation.

---

## 2. Scoring

| Dimension | Weight | Score | Rationale |
|---|---|---|---|
| Originality | 20% | 86 | The S4D + Mamba gating combination applied to robot state prediction with MPC integration fills a genuine gap. The paper now correctly positions itself relative to [40] and [41]. The insight that SSM's linear recurrence aligns well with contact-rich dynamics (Section 5.8 discussion) is a valuable domain contribution. |
| Methodological Rigor | 25% | 85 | MuJoCo evaluation now includes all baselines with significance tests. Multi-step prediction analysis (Table 4) is well-motivated for MPC. The training loss ablation (Table 6b) addresses the Round 12 concern. MPC integration is properly formulated with Adam-based optimization. |
| Evidence Sufficiency | 25% | 85 | Two complementary datasets, four baselines on both, comprehensive ablation on both datasets, MPC demonstration with quantitative metrics, and now complete MuJoCo comparisons with statistical tests. The MuJoCo MPC gap is honestly acknowledged. |
| Argument Coherence | 15% | 88 | The argument is now well-closed: speed advantage on synthetic data, accuracy advantage on MuJoCo, MPC integration demonstrated on synthetic with projected frequency for MuJoCo. The task-specific applicability discussion for R^2 = 0.592 adds nuance. |
| Writing Quality | 15% | 87 | The paper reads well for a robotics audience. Tables are complete and well-labeled. The limitations section is honest. The MuJoCo MPC frequency projection is clearly stated. |

**Weighted Overall Score: 86.0**

---

## 3. Detailed Assessment by Dimension

### 3.1 Originality (86/100)

**Strengths:**
- The paper makes a genuine contribution by demonstrating that SSM-based architectures can serve as lightweight world models for humanoid robot state prediction, a domain where LSTM and Transformer approaches dominate.
- The observation that SSM's linear state transitions may be more robust to contact dynamics discontinuities (Section 5.8, lines 849-854) is a valuable domain insight. The hypothesis is now explicitly stated: "MuJoCo数据集包含真实的接触动力学、摩擦力和碰撞响应, 其状态转移具有不连续性和强非线性, SSM的线性状态转移结构对这种不连续性更具鲁棒性, 不易过拟合噪声接触事件." This is a testable claim that could inspire future work.
- The MPC integration with SSM-WM demonstrates practical utility beyond standalone prediction, connecting the contribution to control theory (relevant to CTA's scope).
- References [40] and [41] are now complete, properly positioning the contribution within the growing SSM-for-robotics literature.

**Minor concern:**
- The term "world model" is used somewhat loosely. In the RL literature (Ha & Schmidhuber [3], Dreamer [12]), a world model typically includes latent dynamics, reward prediction, and imagination-based policy optimization. This paper's model is more accurately a "learned dynamics model" or "state predictor." While this distinction is common in the control community, a brief clarification would improve precision.

### 3.2 Methodological Rigor (85/100)

**Strengths:**
- The MuJoCo Humanoid-v4 evaluation is now complete with all four baselines (Table 8). This is a standard benchmark with realistic dimensions (376 state, 17 action) and includes contact dynamics, friction, and gravity effects.
- Statistical significance tests are now reported for MuJoCo comparisons: SSM-WM vs LSTM-WM (p < 0.01), SSM-WM vs Mamba-WM (p = 0.27). This allows readers to assess whether differences are meaningful.
- The multi-step prediction analysis (Table 4) is important for MPC applications and is well-designed, testing H = 1, 4, 8, 16 steps.
- The training loss ablation (Table 6b) now addresses the Round 12 concern about multi-step loss sensitivity. The finding that lambda = 0.5, H = 8 is a good balance between single-step and multi-step accuracy is practically useful.
- The MPC formulation (Section 4.4) uses Adam-based optimization with 50 iterations, which is a reasonable gradient-based MPC approach consistent with recent literature (e.g., differentiable MPC).

**Remaining concerns:**

1. **MPC experiment scope:** The MPC experiment (Table 7) is conducted only on the synthetic dataset. The authors now project MuJoCo MPC frequency as ~2.1 Hz (line 888), which is a helpful addition. However, for a paper targeting humanoid robot control, the lack of MuJoCo MPC validation remains a gap. The authors should consider whether even a single trajectory tracking task on MuJoCo would strengthen the paper significantly.

2. **Contact dynamics robustness claim:** The hypothesis that SSM's linear recurrence is more robust to contact discontinuities (lines 849-854) is stated but not experimentally validated. A simple test: compare prediction error on contact vs. non-contact timesteps. If SSM-WM shows relatively smaller error during contact events, this would strongly support the claim. This is not required for publication but would be a compelling addition.

3. **Inference time on MuJoCo:** Table 8 reports SSM-WM inference time as 9.5ms on MuJoCo (376-dim state), compared to 3.8ms on synthetic data (28-dim state). The 2.5x increase is expected given the 13x increase in state dimension, but the paper does not break down where this overhead comes from (encoder? decoder? SSM layers?). A brief analysis would help practitioners assess scalability.

### 3.3 Evidence Sufficiency (85/100)

**Strengths:**
- The evidence base is now comprehensive for a journal contribution:
  - Two datasets with complementary characteristics (synthetic near-linear, MuJoCo nonlinear with contacts).
  - Four baselines on both datasets (LSTM, Transformer, Mamba, SSM-WM).
  - Linear regression baseline in Table 1 validates nonlinear modeling necessity.
  - Comprehensive ablation on both datasets (Tables 6, 6b, 8b).
  - MPC demonstration with quantitative tracking metrics.
  - Statistical significance tests on MuJoCo results.
  - Multi-step prediction analysis relevant to MPC.

- The MuJoCo comparison is now complete. Key findings:
  - SSM-WM (0.834) outperforms LSTM-WM (0.889, p < 0.01) and Transformer-WM (0.956).
  - SSM-WM (0.834) is comparable to Mamba-WM (0.821, p = 0.27).
  - This positions SSM-WM as the best among the lightweight alternatives (SSM-WM, LSTM-WM) while competitive with the more complex Mamba-WM.

- The R^2 = 0.592 discussion (lines 869-874) now provides task-specific guidance:
  - For trajectory tracking: joint prediction error ~0.0015 rad, likely acceptable.
  - For aggressive manipulation: may be insufficient.
  - Future directions: more data, contact-aware features, ensemble methods.

**Remaining concerns:**

1. **Synthetic dataset scale:** The synthetic dataset (100 episodes x 150 steps) is small by modern standards. While this is adequate for demonstrating the concept, the authors should acknowledge this limitation more explicitly. The MuJoCo dataset (200 episodes x 200 steps) is more representative.

2. **Random seeds:** Three random seeds (42, 123, 456) provide basic reproducibility but limited statistical power. Five or more seeds would strengthen confidence intervals. This is noted in the limitations (line 890) but not acted upon.

3. **MuJoCo MPC frequency projection:** The projection of ~2.1 Hz (line 888) is based on 50 Adam iterations x 9.5ms = 475ms. This is a reasonable estimate but assumes the MPC optimization converges in 50 iterations for MuJoCo dynamics, which may not hold given the higher-dimensional state space. The authors should note this assumption.

### 3.4 Argument Coherence (88/100)

**Strengths:**
The paper now tells a coherent story from a robotics perspective:

1. **Problem motivation (Section 1):** Humanoid robots need lightweight world models for real-time control. The three challenges (high-dimensional state, nonlinear dynamics, ms-level latency) are well-stated and domain-relevant.

2. **Solution design (Section 4):** SSM-WM combines S4D parameterization (for efficient FFT convolution) with Mamba-style gating (for information selection). The complexity analysis (Section 4.3) is thorough and correctly identifies the O(T log T) training and O(1) recurrent inference advantages.

3. **Validation (Section 5):** The argument is supported by complementary evidence:
   - Speed: 7.3x speedup on synthetic data (batch 64), 2.3x on single sample.
   - Accuracy: 6% improvement over LSTM on MuJoCo, competitive with Mamba.
   - MPC: 5.1 Hz on synthetic, projected 2.1 Hz on MuJoCo.
   - Linear baseline: validates nonlinear modeling necessity.

4. **Honest limitations (Section 5.8):** The six listed limitations are well-structured and honest. The MuJoCo MPC gap, R^2 = 0.592 limitation, and dataset scale issues are all acknowledged.

The argument loop is now nearly closed. The only remaining gap is the MuJoCo MPC experiment, which is clearly positioned as future work with a concrete frequency projection.

**Minor concern:**
- The abstract leads with the batch-64 speedup ("批量推理场景下($B{=}64$)提速约7倍"). For a paper targeting real-time control, the single-sample number (B=1, 2.3x speedup) is more relevant. Consider reordering to lead with the single-sample result, which is the realistic MPC deployment scenario.

### 3.5 Writing Quality (87/100)

**Strengths:**
- The paper reads well in Chinese academic style for a control theory journal.
- Tables are complete, well-labeled, and include standard deviations.
- The limitations section (Section 5.8) is particularly well-written, with clear enumeration and honest assessment.
- The MuJoCo MPC frequency projection is now clearly stated in both the abstract and discussion.
- The task-specific R^2 discussion (lines 869-874) adds valuable nuance for practitioners.
- The English abstract is fluent and accurately summarizes the contributions.

**Minor issues:**

1. **Reference [22] author error:** Still reads "GU A, GU A, RE C" in the bibliography. The S4D paper's authors are Gu, Goel, and Re. This should be "GU A, GOEL K, RE C". This is a trivial fix that has persisted across multiple rounds.

2. **Abstract ordering:** As noted above, consider leading with the single-sample (B=1) speedup for MPC relevance.

3. **MuJoCo inference time note (line 794):** The note about CUDA compilation causing 19ms first-run latency is useful but could be moved to the appendix or a footnote to avoid cluttering the main table.

---

## 4. Comparison with Round 12

| Issue Raised in Round 12 | Status in Round 13 | Assessment |
|---|---|---|
| Table 8 missing Mamba-WM and Transformer-WM baselines | RESOLVED -- Table 8 now includes all 4 baselines | Complete |
| MuJoCo MPC frequency projection missing | RESOLVED -- ~2.1 Hz projection stated in abstract and discussion | Complete |
| R^2 = 0.592 needs task-specific discussion | RESOLVED -- Expanded with trajectory tracking vs. manipulation analysis | Complete |
| Statistical significance on MuJoCo results | RESOLVED -- p-values reported for key comparisons | Complete |
| Reference [22] author error | UNRESOLVED -- Still reads "GU A, GU A, RE C" | Trivial fix needed |

The authors have addressed all four substantive issues I raised in Round 12. The only remaining item is the reference [22] formatting error, which is trivial.

---

## 5. Domain-Specific Commentary

### 5.1 Relevance to Control Theory & Applications (CTA)

This paper is well-suited for CTA. The core contribution -- a lightweight learned dynamics model for MPC -- sits squarely at the intersection of machine learning and control theory. The SSM framework has deep roots in linear systems theory (the paper correctly cites Kalman [29] as foundational), and the paper's complexity analysis is rigorous enough for a control audience.

The MPC integration (Section 4.4) uses gradient-based optimization (Adam), which is consistent with recent differentiable MPC literature. While traditional MPC uses quadratic programming or sequential quadratic programming, the gradient-based approach is appropriate here because the learned dynamics model (SSM-WM) is differentiable end-to-end.

### 5.2 Practical Relevance to Robotics

From a robotics practitioner's perspective, the paper addresses a real need: deploying learned dynamics models on resource-constrained platforms. The key practical findings are:

- **Single-sample inference (0.9ms)** meets 1kHz control requirements for high-frequency servo loops.
- **MPC loop time (195ms on synthetic)** enables 5.1 Hz planning, suitable for motion planning.
- **MuJoCo MPC projection (~2.1 Hz)** is realistic for low-frequency planning tasks (e.g., trajectory optimization, footstep planning).
- **Parameter count (0.24M)** is small enough for embedded deployment.

The paper would benefit from a brief discussion of deployment considerations: GPU vs. CPU inference, ONNX export, and quantization potential. These are practical concerns for robotics engineers.

### 5.3 Theoretical Contribution

The paper's theoretical contribution is primarily the complexity analysis (Section 4.3) and the connection between SSM and control theory. The observation that diagonal SSM parameterization (S4D) provides a natural bridge between continuous-time state-space representations (familiar to control engineers) and efficient discrete-time computation (familiar to ML practitioners) is valuable for the CTA community.

The Mamba-2 reference [11] (structured state space duality) is correctly cited as providing theoretical unification between SSM and attention mechanisms. This positions the paper within a broader theoretical framework.

---

## 6. Remaining Issues (Prioritized)

| # | Priority | Issue | Recommendation |
|---|---|---|---|
| 1 | LOW | Reference [22] author error persists | Fix "GU A, GU A, RE C" to "GU A, GOEL K, RE C" |
| 2 | LOW | Abstract ordering for MPC relevance | Consider leading with B=1 speedup (0.9ms, 2.3x) before B=64 |
| 3 | SUGGESTION | Contact dynamics robustness hypothesis | Consider a brief analysis of per-timestep error during contact vs. non-contact phases |
| 4 | SUGGESTION | Deployment considerations | Brief discussion of GPU/CPU inference, ONNX export, or quantization potential |
| 5 | SUGGESTION | MuJoCo inference breakdown | Brief analysis of where the 9.5ms overhead comes from (encoder/SSM/decoder) |

None of these issues are blocking. Items 1-2 are trivial fixes. Items 3-5 are suggestions that would enhance the paper but are not required for acceptance.

---

## 7. Final Recommendation

**Recommendation: Accept**

The paper has reached a level of quality appropriate for publication in 控制理论与应用. Over 13 rounds of revision, the authors have:

1. Completed the MuJoCo evaluation with all four baselines and statistical tests.
2. Added a MuJoCo MPC frequency projection (~2.1 Hz).
3. Expanded the R^2 = 0.592 discussion with task-specific applicability analysis.
4. Maintained honest and well-structured limitations.

The core contribution -- demonstrating that SSM-based world models can achieve competitive accuracy with significantly faster inference for humanoid robot state prediction -- is well-supported by the evidence and relevant to the CTA community. The MPC integration demonstrates practical utility, and the complexity analysis is rigorous.

The only remaining issue (reference [22] author error) is a trivial formatting fix that should not delay publication.

**To the authors:** The paper has improved substantially since my Round 12 review. The four issues I raised have all been addressed satisfactorily. The task-specific R^2 discussion and MuJoCo MPC frequency projection are particularly welcome additions. I recommend acceptance.

---

*Review completed: Round 13*
*R2 Overall Score: 86.0 (up from 74.0 in Round 12)*
*Previous round issues addressed: 4/4 substantive, 1 trivial remaining*
