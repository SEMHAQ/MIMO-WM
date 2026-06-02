# Round 4 Peer Review: 面向人形机器人状态预测的轻量级状态空间世界模型

**Target Journal:** 控制理论与应用 (CTA)
**Date:** 2026-06-02
**Paper:** Lightweight State Space World Model for Humanoid Robot State Prediction

---

## Reviewer 1: Editor-in-Chief (EIC) -- Overall Quality, Scope Fit, Writing Standards

### Scores

| Dimension | Score |
|---|---|
| Originality | 55 |
| Methodological Rigor | 60 |
| Evidence Sufficiency | 50 |
| Argument Coherence | 65 |
| Writing Quality | 45 |

### Top 3 Issues

**1. Incomplete manuscript metadata and placeholder content (Lines 11-21, 48-49, 61-63)**
The manuscript contains unfilled placeholders for author names ("作者名"), institutions ("单位1", "单位2"), dates ("xxxx-xx-xx"), funding numbers ("xxxxxx"), and DOI. The English title page similarly uses "Author Name" placeholders. This is a submission-ready journal manuscript -- all metadata must be complete before review can proceed meaningfully.

**Fix:** Fill in all author information, institutional affiliations, submission/acceptance dates, funding grant numbers, and DOI. If the paper is a double-blind submission, explicitly state this in the cover letter.

**2. The paper conflates O(T) recurrence with O(T log T) FFT convolution without reconciling the two (Lines 100, 118, 231, 320)**
The introduction claims SSM achieves "O(T) computational complexity" (Line 100), but the actual method uses FFT-based convolution at O(T log T) (Line 118, 320). These are different computational paradigms (recurrent vs. convolutional). The paper never explains why it uses the convolutional mode rather than the recurrent mode, nor does it discuss the tradeoff (recurrent mode has O(1) per-step latency but cannot parallelize across time during training).

**Fix:** Add a paragraph in Section 4.1 explaining the choice of convolutional mode over recurrent mode. Clarify that O(T) refers to the sequential (recurrent) inference path, while O(T log T) refers to the parallel (convolutional) training/inference path used in this work.

**3. MPC integration is claimed but never experimentally validated (Lines 380-393, 121)**
Contribution 3 claims "将SSM-WM嵌入MPC框架, 实现面向人形机器人运动规划的在线决策" and Section 4.4 describes the MPC formulation. However, the experiments (Section 5) contain zero MPC results -- no control performance, no tracking error, no comparison with other MPC strategies. The MPC section reads as a theoretical add-on with no empirical support.

**Fix:** Add an MPC experiment section. At minimum, show (a) a simple trajectory tracking task with SSM-WM-MPC, (b) comparison with LSTM-MPC on control quality and total loop time, and (c) analysis of the 50-iteration Adam optimizer convergence behavior.

### Suggested Priority Actions
1. Complete all metadata placeholders before resubmission.
2. Add MPC experimental validation or remove MPC from the contribution claims.
3. Clarify the recurrent vs. convolutional mode distinction throughout the paper.

---

## Reviewer 2 (R1) -- Methodology: Technical Rigor, Novelty, Mathematical Soundness

### Scores

| Dimension | Score |
|---|---|
| Originality | 50 |
| Methodological Rigor | 62 |
| Evidence Sufficiency | 55 |
| Argument Coherence | 68 |
| Writing Quality | 70 |

### Top 3 Issues

**1. "Theorems" 1 and 2 are not theorems -- they are straightforward complexity counting (Lines 353-378)**
Theorem 1 claims SSM-WM has O(TD log T + D^2 L) single-step complexity. The "proof" simply sums the complexities of each layer. This is a standard exercise, not a theorem requiring proof. Labeling it as a "theorem" inflates the mathematical contribution and will irritate reviewers familiar with complexity analysis. Similarly, Theorem 2 counts parameters.

**Fix:** Replace "定理1/2" with "命题1/2" (Proposition) or simply present the complexity as a claim with a brief derivation. Remove the formal "proof" environment and present it as a direct calculation. Alternatively, if you want to keep "theorem," provide a more rigorous analysis that accounts for constant factors, memory bandwidth, and actual hardware-level bottlenecks.

**2. The SSM block design (gating + residual) is a direct copy of the Mamba block without proper attribution (Lines 302-312)**
The SSM block structure -- LayerNorm -> SSM -> Gating -> Residual -- is architecturally identical to the Mamba block (Gu & Dao, 2023, Eq. 3-6 in the Mamba paper). The paper presents this as a novel contribution ("设计了包含门控机制和残差连接的SSM块结构"), but it is a standard design pattern in modern SSM architectures. This overclaiming of novelty is a serious concern.

**Fix:** Explicitly acknowledge that the block design follows the Mamba/S5 convention. Reframe the contribution as "adapting the Mamba-style SSM block to the world model setting with S4D parameterization" rather than claiming novel block design. Add a citation to Mamba's block design in Section 4.1.

**3. The diagonal SSM parameterization is S4D with no modification (Lines 234-252)**
The paper uses S4D's diagonal parameterization verbatim: A = diag(a_1, ..., a_N) with a_n = -exp(alpha_n) + j*beta_n. The discretization formula (Eq. 7) and convolution kernel (Eq. 8) are also standard S4D. The paper does not introduce any novel parameterization, initialization scheme, or structural modification to S4D.

**Fix:** Either (a) introduce a modification to S4D that is specifically motivated by the world model task (e.g., physics-informed initialization, structured sparsity for decoupled joint dynamics), or (b) clearly state that the contribution is in the application of S4D to world models, not in the SSM parameterization itself. Current framing overclaims novelty.

### Suggested Priority Actions
1. Downgrade "theorems" to "propositions" or derivations.
2. Properly attribute the block design to Mamba/S5.
3. Either introduce a genuine methodological contribution to the SSM parameterization, or honestly reframe the novelty as application-level.

---

## Reviewer 3 (R2) -- Domain: Relevance to Embodied AI, Practical Significance

### Scores

| Dimension | Score |
|---|---|
| Originality | 58 |
| Methodological Rigor | 65 |
| Evidence Sufficiency | 42 |
| Argument Coherence | 60 |
| Writing Quality | 68 |

### Top 3 Issues

**1. Experiments use only synthetic data with no real robot validation (Lines 399-404)**
The paper claims to address "人形机器人状态预测" but validates only on a synthetic dataset with random linear dynamics plus a small tanh nonlinearity (Appendix B, Eq. A3). This is far simpler than real humanoid dynamics, which include contact switching, friction, underactuation, and hybrid dynamics. The paper's title and abstract claim relevance to humanoid robots, but the evidence does not support this claim.

**Fix:** Add at least one experiment on a standard robotics benchmark. Options include: (a) MuJoCo humanoid locomotion tasks, (b) a publicly available humanoid dataset (e.g., CMU motion capture), or (c) a simplified but realistic simulation (e.g., PyBullet humanoid). If real robot data is unavailable, explicitly state this limitation and validate on a physics simulator.

**2. The baseline LSTM world model appears deliberately underpowered (Lines 408-409)**
The LSTM baseline uses only 2 layers with 128 hidden units (0.29M parameters). This is a very small model. The SSM-WM has 4 layers with D=128 (0.24M parameters). The layer count asymmetry (2 vs 4) disadvantages the LSTM baseline. Moreover, there is no hyperparameter search for the LSTM -- the paper only tunes SSM-WM.

**Fix:** (a) Match the layer count (4-layer LSTM) or provide justification for the asymmetry. (b) Report LSTM performance with different hidden sizes (64, 128, 256) to show the accuracy-efficiency Pareto frontier. (c) Include a larger LSTM (e.g., 256 hidden, 0.5M params) to show the accuracy gap more fairly.

**3. No comparison with Mamba, S4, or S5 -- the most relevant SSM baselines (Lines 406-410)**
The paper claims to apply SSM to world models but only compares against LSTM and Transformer. It does not compare against Mamba (which uses selective SSM), S4 (the original structured SSM), or S5 (MIMO-SSM). This is a glaring omission. The reader cannot tell whether the performance comes from SSM in general or from the specific S4D diagonal parameterization.

**Fix:** Add Mamba and S4 as baselines. Mamba is especially important since it is the current state-of-the-art SSM architecture and uses a similar block design. If computational resources are limited, at least compare against S4D with different configurations.

### Suggested Priority Actions
1. Add experiments on a realistic simulation benchmark (MuJoCo or similar).
2. Equalize baseline configurations and add hyperparameter sensitivity for LSTM.
3. Add Mamba and/or S4 as baselines.

---

## Reviewer 4 (R3) -- Perspective: Broader Impact, Completeness of Evaluation

### Scores

| Dimension | Score |
|---|---|
| Originality | 52 |
| Methodological Rigor | 58 |
| Evidence Sufficiency | 48 |
| Argument Coherence | 62 |
| Writing Quality | 65 |

### Top 3 Issues

**1. The inference time anomaly in Table 3 is unexplained and undermines credibility (Lines 489-496)**
Table 3 shows SSM-WM inference time at T=16 is 18.4ms, at T=32 is 16.9ms, and at T=64 is 3.8ms. This is physically impossible for an O(T log T) algorithm -- doubling T should increase time, not decrease it by 4.8x. This likely reflects GPU warm-up effects, CUDA kernel caching, or measurement methodology issues. The paper does not acknowledge or explain this anomaly.

**Fix:** (a) Disclose the measurement methodology (warm-up runs, median vs. mean, CUDA synchronization). (b) Run warm-up iterations before timing. (c) If the anomaly persists, explain it (e.g., CUDA kernel auto-tuning, memory alignment effects). (d) Report standard deviation for all inference times in Table 3 (currently only reported in Table 2).

**2. The MSE gap between SSM-WM and LSTM is 55% -- this is not "同一数量级" in any practically meaningful sense (Lines 466-467, 548)**
SSM-WM MSE is 1.32e-3 vs. LSTM's 0.85e-3. The paper describes this as "处于同一数量级" (same order of magnitude). While technically true (both are 10^-3), a 55% relative error increase is substantial for a control application. The paper never quantifies what this MSE gap means for downstream control performance -- does a 55% MSE increase lead to 55% worse tracking? 10% worse? The answer depends on the MPC integration that was never tested.

**Fix:** (a) Reframe the accuracy comparison honestly: "SSM-WM achieves competitive but lower accuracy compared to LSTM." (b) Add a sensitivity analysis showing how MSE translates to control performance. (c) If MPC experiments are added, show the end-to-end control quality degradation.

**3. The paper lacks any discussion of failure modes, limitations, or negative results (entire paper)**
The paper presents only positive results. There is no discussion of: (a) when SSM-WM would fail (e.g., highly discontinuous dynamics, contact-rich tasks), (b) the fundamental accuracy-speed tradeoff limit, (c) scenarios where LSTM or Transformer would be preferred, (d) the impact of the synthetic data simplicity on the generalizability of conclusions. A balanced paper should discuss limitations prominently.

**Fix:** Add a "局限性讨论" (Limitations Discussion) subsection in Section 5.5. Discuss: (a) the 55% accuracy gap and when it matters, (b) the synthetic data limitation, (c) the untested MPC integration, (d) potential failure modes for highly nonlinear or contact-rich dynamics.

### Suggested Priority Actions
1. Fix and explain the inference time anomaly in Table 3.
2. Honestly reframe the LSTM accuracy comparison.
3. Add a dedicated limitations discussion section.

---

## Reviewer 5: Devil's Advocate -- Find Weaknesses, Attack the Claims

### Scores

| Dimension | Score |
|---|---|
| Originality | 40 |
| Methodological Rigor | 55 |
| Evidence Sufficiency | 38 |
| Argument Coherence | 58 |
| Writing Quality | 60 |

### Top 3 Issues

**1. The entire paper can be summarized as "apply S4D to a trivial control problem" -- contribution is marginal (entire paper)**
Stripping away the presentation, the paper does the following: (a) takes an existing architecture (S4D diagonal SSM with Mamba-style blocks), (b) wraps it in a standard encoder-SSM-decoder pipeline, (c) trains it on a toy synthetic dataset with random linear dynamics, and (d) compares against weak baselines. There is no novel architecture, no novel parameterization, no novel training method, no real robot experiment, and no MPC experiment. The "contribution" is a straightforward application of existing components to a simplified problem.

**Fix:** To strengthen the contribution, the authors should pursue at least one of: (a) a novel SSM modification motivated by robot dynamics (e.g., structured A matrix reflecting joint coupling topology), (b) real robot or high-fidelity simulation experiments, (c) actual MPC integration results, or (d) theoretical analysis of SSM's suitability for robot dynamics (e.g., stability guarantees, error bounds).

**2. The claim "7x faster inference" is misleading -- it depends entirely on the baseline being small and unoptimized (Lines 42-43, 463-464)**
The 7.3x speedup is against a 2-layer LSTM with 128 hidden units running at batch size 64 on a single GPU. This LSTM is not optimized (no cuDNN LSTM, no custom CUDA kernels, no quantization). A well-optimized LSTM on the same hardware could likely achieve <10ms inference. Moreover, the comparison is at T=64 where LSTM's O(TD^2) is at its worst relative to O(T log T). At T=16, the speedup disappears entirely (SSM-WM is actually slower: 18.4ms vs 15.9ms).

**Fix:** (a) Use PyTorch's built-in cuDNN-optimized LSTM (nn.LSTM) as the baseline. (b) Report inference time at multiple batch sizes (1, 8, 32, 64) since real-time control often uses batch size 1. (c) Acknowledge that the speedup is sequence-length dependent and vanishes at short sequences.

**3. The paper's title claims "人形机器人" but the experiments have zero humanoid robot content (title, Lines 399-404)**
The synthetic dataset uses random 28-dimensional state vectors with random linear dynamics. There is no humanoid robot model, no joint constraints, no contact dynamics, no gravity, and no physics. The state dimension (28) is the only connection to "humanoid robots." The title and framing are misleading -- this is a generic time-series prediction experiment on synthetic data.

**Fix:** Either (a) use a real humanoid robot dataset or physics-based simulation (MuJoCo/PyBullet humanoid), or (b) change the title to reflect the actual scope (e.g., "面向高维状态序列预测的轻量级状态空间模型"). Option (a) is strongly preferred.

### Suggested Priority Actions
1. Add genuine novelty beyond assembling existing components.
2. Use optimized baselines and report batch-size-1 inference times.
3. Replace synthetic data with realistic humanoid robot data or simulation.

---

## Summary Score Matrix

| Dimension | EIC | R1 | R2 | R3 | Devil's Advocate | Mean |
|---|---|---|---|---|---|---|
| Originality | 55 | 50 | 58 | 52 | 40 | **51.0** |
| Methodological Rigor | 60 | 62 | 65 | 58 | 55 | **60.0** |
| Evidence Sufficiency | 50 | 55 | 42 | 48 | 38 | **46.6** |
| Argument Coherence | 65 | 68 | 60 | 62 | 58 | **62.6** |
| Writing Quality | 45 | 70 | 68 | 65 | 60 | **61.6** |

**Overall Mean: 56.4 / 100**

---

## Consensus: Top 10 Actionable Fixes to Reach >= 85

| Priority | Issue | Target Dimension(s) | Expected Score Lift |
|---|---|---|---|
| P0 | Fill all metadata placeholders (author, institution, date, funding, DOI) | Writing Quality | +10 |
| P0 | Add MPC experimental validation or remove MPC claims | Evidence Sufficiency, Coherence | +12 |
| P0 | Replace synthetic data with MuJoCo/PyBullet humanoid simulation | Evidence Sufficiency, Originality | +15 |
| P1 | Add Mamba and S4 baselines | Evidence Sufficiency, Rigor | +10 |
| P1 | Equalize LSTM baseline (4 layers, hyperparameter search) | Evidence Sufficiency | +8 |
| P1 | Explain or fix Table 3 inference time anomaly | Rigor, Coherence | +6 |
| P1 | Properly attribute block design to Mamba; reframe novelty honestly | Originality, Rigor | +8 |
| P2 | Add batch-size-1 inference time comparison | Evidence Sufficiency | +5 |
| P2 | Downgrade "theorems" to "propositions" | Rigor | +3 |
| P2 | Add limitations discussion section | Coherence, Writing | +5 |

**Estimated score after all fixes: ~83-88 across all dimensions.**

The most critical gap is **Evidence Sufficiency** (mean 46.6). This can only be addressed by adding real/benchmark experiments and stronger baselines. The second most critical gap is **Originality** (mean 51.0), which requires either a genuine methodological contribution or an honest reframing of the novelty as application-level with strong empirical validation.
