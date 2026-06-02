# Round 13 EIC Review -- CTA (控制理论与应用)

**Paper Title:** 面向人形机器人状态预测的轻量级状态空间世界模型 (Lightweight State Space World Model for Humanoid Robot State Prediction)
**Manuscript:** /mnt/e/Project/SSM-World-Model/paper/main.tex (~1037 lines)
**Review Round:** Round 13 (EIC-only final assessment)
**Previous Decision:** Accept with Minor Revisions (Round 12, Grand Mean 69.8)

---

## 1. Status of Round 12 Issues

All issues raised in Round 12 have been addressed:

| # | Issue | Status | Evidence |
|---|---|---|---|
| 1 | Reference [22] author list | FIXED | Line 945: "GU A, GOEL K, RE C" -- correct |
| 2 | Linear regression baseline in Table 1 | FIXED | Line 475: row "线性回归" with MSE=2.50, MAE=0.035, R^2=0.890, params=0.01 |
| 3 | Table 8 all 4 baselines | FIXED | Lines 788-792: LSTM, Transformer, Mamba, SSM-WM all present |
| 4 | MuJoCo MPC frequency projection | FIXED | Line 888: "~2.1Hz" with derivation (1/(50*9.5ms)) |
| 5 | Multi-step loss ablation (Table 6b) | FIXED | Lines 639-664: lambda in {0, 0.1, 0.5, 1.0}, H in {4, 8, 16} |
| 6 | Training convergence analysis | FIXED | Lines 666-738: convergence description + Figure 2 training curve |
| 7 | Abstract specifies "compared to LSTM-2L" | FIXED | Line 46: "参数量减少约17%（相比LSTM-2L）" |
| 8 | Statistical significance on MuJoCo | FIXED | Line 800: SSM-WM vs LSTM-WM p<0.01; Line 801: vs Mamba-WM p=0.27 |

This is a clean sweep. Every item from the Round 12 consolidated issues table has been resolved. The authors have been responsive and thorough.

---

## 2. Dimension-by-Dimension Assessment

### 2.1 Originality -- Score: 76/100

The paper assembles established components -- S4D diagonal SSM parameterization [22], Mamba-style gated blocks [7], and FFT-based convolution -- into a pipeline targeted at humanoid robot state prediction with MPC integration. The individual building blocks are not novel; their combination and application domain constitute the contribution.

**What works well:**
- The application of SSM to robot world models addresses a genuine gap. Prior SSM work focused on language modeling and long-range benchmarks; this paper brings SSM into the control-theoretic domain of CTA.
- The MPC integration (Section 4.4) is well-motivated: the SSM's O(T log T) training and O(1) recurrent inference make it a natural fit for the inner loop of gradient-based MPC.
- The paper now correctly positions itself relative to Mamba policy [40] and SSM trajectory prediction [41], clarifying the distinct contribution (world model for state prediction rather than policy learning).

**What limits the score:**
- The architecture is essentially S4D backbone + Mamba block structure + standard encoder-decoder. There is no novel architectural mechanism proposed.
- The term "world model" is used in a restricted sense (next-state predictor), not in the richer RL sense of latent dynamics with imagination-based planning (cf. Dreamer [12], DreamerV3 [26]). The paper acknowledges this implicitly but does not explicitly discuss the conceptual distinction.
- The insight that SSM's linear recurrence may be more robust to contact discontinuities (line 853) is interesting but remains a post-hoc hypothesis without theoretical support.

For CTA, where the readership values control-theoretic contributions with practical robotics relevance, this level of originality is adequate and above the threshold for publication.

### 2.2 Methodological Rigor -- Score: 74/100

The experimental methodology has improved substantially over the revision rounds.

**Strengths:**
- Two complementary datasets (synthetic near-linear, MuJoCo nonlinear) provide a meaningful evaluation landscape.
- Comprehensive ablation study covering architecture components (Table 6), training loss design (Table 6b), and MuJoCo-specific ablation (Table 8b).
- Statistical significance testing with paired t-tests, p-values, and 95% confidence intervals on key comparisons.
- Hyperparameter grid search for lambda and H documented (line 366).
- Inference time measurement protocol is well-specified: GPU warmup, 20 measurements, median reporting (line 457).
- Training convergence analysis with epoch-level description and Figure 2.

**Remaining concerns:**
1. **Small dataset scale:** The synthetic dataset has only 100 episodes x 150 steps = 15,000 transitions. MuJoCo has 200 episodes x 200 steps = 40,000 transitions. These are small by modern standards. The paper should acknowledge this more prominently in the limitations.
2. **Limited random seeds:** Three seeds (42, 123, 456) provide basic reproducibility but limited statistical power. For claims of statistical significance, more seeds would strengthen confidence.
3. **Fixed compute budget MPC comparison:** The MPC experiment (Table 7) uses 50 Adam iterations for all methods -- a fixed compute budget. A complementary comparison fixing accuracy targets would provide additional insight.
4. **Recurrent vs. convolutional mode:** The paper explains the choice of convolutional mode for MPC (lines 245-248), but the B=1 inference time (0.9ms, Table 3) is reported in convolutional mode. For single-step online prediction, recurrent mode with O(1) latency would be more appropriate. This ambiguity should be clarified.
5. **MuJoCo inference time characterization:** Table 8 reports 9.5ms median but the footnote mentions 19ms first-run due to CUDA compilation. Standard deviation (0.5ms) is provided. For real-time control claims, P95 latency would be informative.

Despite these concerns, the methodology is solid for a journal contribution. The authors have been transparent about limitations.

### 2.3 Evidence Sufficiency -- Score: 74/100

The evidence base has improved significantly since earlier rounds.

**What the evidence shows:**
- On synthetic data: SSM-WM is ~7x faster than LSTM with ~55% higher MSE, but R^2=0.945 (explains 94.5% of variance). Linear regression baseline (MSE=2.50) confirms nonlinear modeling necessity.
- On MuJoCo: SSM-WM outperforms LSTM-WM by ~6% (p<0.01) and Transformer-WM by ~13%, with performance comparable to Mamba-WM (gap <2%, p=0.27).
- MPC: SSM-WM-MPC achieves 5.1Hz with tracking accuracy comparable to LSTM-MPC (p=0.12).
- Multi-step prediction: SSM-WM error accumulation is slower than LSTM (38-47% gap vs 55% single-step gap).
- Ablation: Gate mechanism (2.3-4.6% contribution), residual connections (1.5-2.6%), depth scaling, and loss design all validated.

**What remains incomplete:**
- MuJoCo MPC experiment is absent, replaced by a frequency projection (~2.1Hz). While honest, this means the paper's central claim (SSM enables real-time MPC for humanoid control) is only fully validated on synthetic data.
- The R^2=0.592 on MuJoCo (41% unexplained variance) is acknowledged but the paper could provide more guidance on which control tasks this accuracy level can support.
- No cross-dataset generalization test (train on synthetic, test on MuJoCo, or vice versa).

For CTA, the evidence is sufficient. The two-dataset evaluation, comprehensive ablation, and MPC demonstration provide a solid foundation. The gaps are clearly acknowledged.

### 2.4 Argument Coherence -- Score: 78/100

The paper tells a coherent story that has improved with each revision round.

**The argument chain:**
1. Problem statement: Lightweight world models needed for humanoid robot real-time control (Section 1).
2. Solution: SSM-WM with S4D parameterization + Mamba gating + FFT convolution (Section 4).
3. Speed validation: 7x speedup on synthetic, <10ms on MuJoCo (Tables 2, 3, 8).
4. Accuracy validation: Competitive with LSTM on MuJoCo, superior to Transformer (Table 8).
5. MPC integration: 5.1Hz control on synthetic, projected 2.1Hz on MuJoCo (Tables 7).
6. Component validation: Systematic ablation (Tables 6, 6b, 8b).
7. Limitations: Honest acknowledgment of MuJoCo MPC gap, R^2 limitation, dataset scale (Section 5.8).

**Strengths:**
- The limitations section (Section 5.8) is unusually thorough for a revised paper. It covers 6 specific limitations with quantitative detail.
- The dataset-dependent performance discussion (lines 848-854) provides a plausible mechanistic explanation.
- The SSM-WM vs Mamba-WM comparison (lines 862-867) is fair and well-argued: SSM-WM trades ~2% accuracy for simpler implementation and faster inference.
- The future work section (lines 909-914) provides a concrete 5-point roadmap.

**What could be stronger:**
- The speed-accuracy narrative spans two different datasets, creating a structural gap. The paper acknowledges this but a unified experiment would close the loop.
- The claim "满足1kHz高频控制要求" (line 44) is based on B=1 inference time of 0.9ms. But this is single-inference latency without the MPC optimization loop. The MPC loop time is 195ms (5.1Hz), not 1kHz. The abstract should be more precise about what achieves 1kHz.

### 2.5 Writing Quality -- Score: 80/100

The writing quality is good and has improved consistently through the revision process.

**Strengths:**
- The Chinese academic writing style is appropriate for CTA.
- Table and figure labeling is clean (no duplicate labels, all properly captioned).
- The English abstract is fluent and accurately represents the Chinese content.
- The related work section (Section 2) is comprehensive with 4 well-organized subsections.
- Mathematical notation is consistent throughout (boldface for vectors/matrices, proper use of operators).
- The appendix provides useful supplementary details (hyperparameters, dataset generation, compute environment).

**Minor issues:**
- The abstract's "满足1kHz高频控制要求" refers to single-sample inference (0.9ms), but the MPC loop achieves 5.1Hz. These are different claims and should be disambiguated.
- Line 888: "约2.1Hz" -- the derivation is clear but the sentence structure could be cleaner.
- The reference list has some entries with incomplete publication information (e.g., [7] and [11] are arXiv preprints without journal/conference confirmation).
- Some figure captions could be more descriptive (e.g., Figure 1 is minimal).

---

## 3. Scoring Summary

| Dimension | Weight | Score | Justification |
|---|---|---|---|
| Originality | 20% | 76 | Assembly of known components for a new application domain; no novel mechanism but fills a genuine gap |
| Methodological Rigor | 25% | 74 | Comprehensive experiments with ablations and significance tests; limited by small datasets and 3 seeds |
| Evidence Sufficiency | 25% | 74 | Two datasets, 4 baselines, MPC demo; MuJoCo MPC gap acknowledged but unresolved |
| Argument Coherence | 15% | 78 | Coherent narrative with honest limitations; speed-accuracy split across datasets |
| Writing Quality | 15% | 80 | Good Chinese academic writing; minor precision issues in abstract claims |
| **Weighted Overall** | **100%** | **75.8** | |

---

## 4. Journal Fit Assessment

**Fit with CTA readership: Strong.**

CTA (控制理论与应用) covers control theory, systems engineering, and their applications. This paper sits at the intersection of:
- State space models (rooted in control theory, connecting to Kalman's original formulation [29])
- Model prediction control (a core CTA topic)
- Robot state estimation and prediction (a growing application area)

The paper's framing in terms of SSM as a control-theoretic construct (Section 3.2-3.3) and MPC integration (Section 4.4) aligns well with the journal's scope. The computational efficiency focus (real-time inference, parameter count, latency) is relevant to practical control system design.

The paper does not compete with vision-language-action models or large-scale RL; it occupies a well-defined niche of lightweight dynamics models for state prediction. This scope is appropriate for CTA.

---

## 5. Comparison with Round 12

| Metric | Round 12 | Round 13 | Delta |
|---|---|---|---|
| Grand Mean (all reviewers) | 69.8 | -- | -- |
| EIC Overall | 71.6 | 75.8 | +4.2 |

The improvement of +4.2 EIC points is driven by:
1. **Table 8 completeness (+2 evidence):** All 4 baselines now present on MuJoCo, enabling fair comparison.
2. **Table 6b multi-step loss ablation (+1.5 rigor):** Key training design choice now validated.
3. **Training convergence analysis (+1 rigor):** Convergence within 20 epochs verified with curves.
4. **Statistical significance on MuJoCo (+1 evidence):** p-values reported for key comparisons.
5. **MuJoCo MPC frequency projection (+1 coherence):** Argument chain more complete.
6. **Reference [22] fix (+0.5 writing):** Eliminates a persistent irritant.
7. **Abstract precision (+0.5 writing):** "compared to LSTM-2L" clarifies the parameter reduction claim.

---

## 6. Remaining Issues (Minor)

These are minor issues that should not delay publication but should be addressed in the final camera-ready version:

| # | Priority | Issue |
|---|---|---|
| 1 | MINOR | Abstract "满足1kHz高频控制要求" refers to B=1 latency, not MPC loop. Clarify. |
| 2 | MINOR | Synthetic dataset scale (15,000 transitions) should be noted as a limitation more prominently. |
| 3 | MINOR | References [7] and [11] are arXiv preprints; verify if published versions are available. |
| 4 | MINOR | Figure 1 architecture diagram is minimal; consider a more detailed version. |

None of these are blocking issues.

---

## 7. Final Assessment

### What the paper achieves:
- Demonstrates that SSM-based world models can achieve competitive prediction accuracy with significantly faster inference for humanoid robot state prediction.
- Provides a clean, well-scoped contribution that bridges SSM research and robotics control.
- Offers honest, quantitative assessment of strengths and limitations.
- Integrates SSM-WM with MPC and demonstrates real-time control capability.

### What the paper does not achieve:
- Full MuJoCo MPC validation (acknowledged as future work).
- State-of-the-art prediction accuracy (LSTM is more accurate on synthetic data).
- Novel architectural contribution (the components are established).

### Decision Rationale:
The paper has crossed the threshold for acceptance at CTA. The contribution is well-defined, the experiments are adequate, the writing is good, and the limitations are honestly discussed. The Round 12 issues have all been resolved. The remaining issues are minor and can be addressed in the camera-ready version.

The paper would be stronger with MuJoCo MPC results and larger datasets, but these are reasonable items for future work given the paper's scope. The 12-round revision process has produced a significantly improved manuscript that makes a genuine contribution to the intersection of SSM and robotics control.

---

## 8. Recommendation

**Decision: Accept**

The paper is recommended for publication in 控制理论与应用 (Control Theory & Applications). All substantive issues from Round 12 have been resolved. The remaining items are minor editorial polish suitable for the camera-ready stage.

**To the authors:** The paper has improved substantially over 13 rounds. The contribution is now well-supported, clearly presented, and appropriately scoped. Congratulations on a thorough revision process.

---

*Review completed: Round 13 (EIC Final Assessment)*
*Previous round: Round 12 (Grand Mean 69.8, Decision: Accept with Minor Revisions)*
*This round: Round 13 (EIC Score: 75.8)*
*Threshold for acceptance at CTA: ~70*
