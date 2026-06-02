# Round 13 Peer Review Report -- Reviewer 3 (Cross-disciplinary Perspective)

**Paper Title:** 面向人形机器人状态预测的轻量级状态空间世界模型
**Manuscript:** /mnt/e/Project/SSM-World-Model/paper/main.tex (~1038 lines)
**Review Date:** 2026-06-02
**Review Round:** Round 13
**Previous R3 Score (Round 12):** 76.0

---

## Reviewer Profile

Cross-disciplinary perspective with expertise in machine learning systems, efficient computing, and embedded AI deployment. Focus areas for this round: practical deployment implications, broader impact on embedded AI and real-time systems, cross-disciplinary connections, and challenging fundamental assumptions.

---

## Summary of Improvements Since Round 12

All eight issues flagged in the Round 12 consolidated review have been addressed:

| # | Issue (Round 12) | Status (Round 13) |
|---|---|---|
| 1 | Reference [22] author error ("GU A, GU A, RE C") | **FIXED** -- Now reads "GU A, GOEL K, RE C" (line 945) |
| 2 | Linear baseline only in discussion text | **FIXED** -- Now a row in Table 1 (line 475) |
| 3 | MuJoCo missing baselines (Table 8) | **FIXED** -- Now includes LSTM, Transformer, Mamba, SSM-WM (lines 788-792) |
| 4 | MuJoCo MPC gap | **ACKNOWLEDGED** -- Explicit limitation (line 887) with projected frequency (line 888) |
| 5 | MuJoCo MPC frequency projection | **FIXED** -- "2.1 Hz" estimate provided (line 888) |
| 6 | Multi-step loss ablation missing | **FIXED** -- Table 6b with lambda/H sweep (lines 639-664) |
| 7 | Statistical significance on MuJoCo | **FIXED** -- p-values reported for SSM vs LSTM (p<0.01) and SSM vs Mamba (p=0.27) |
| 8 | Training convergence unverified | **FIXED** -- Convergence analysis added (lines 667-671), 20 epochs sufficient |

Additionally, the paper now includes:
- MuJoCo ablation experiment (Table 8b) confirming component contributions on high-dimensional data
- Expanded batch size analysis (B=1, 8, 32, 64) in Table 3
- Detailed discussion of MuJoCo prediction accuracy and its implications for different control tasks (lines 869-874)
- More comprehensive limitations section covering 6 distinct points

---

## Dimension 1: Originality (Score: 87/100)

### Assessment

The paper's originality lies not in any single component but in the deliberate architectural assembly for a specific, underserved application domain. The S4D diagonal parameterization [22] and Mamba-style gating [7] are individually established, but their combination into a lightweight world model for humanoid state prediction with MPC integration addresses a genuine gap.

**Cross-disciplinary connection:** The paper implicitly bridges control theory (SSM as linear dynamical systems) and deep learning (SSM as sequence models). This duality, articulated in Mamba-2 [11] as "structured state space duality," is underappreciated in the robotics community. The paper could strengthen its originality claim by explicitly noting that the SSM formulation recovers the classical control-theoretic state-space model (referenced via Kalman [29]) as a special case, and that the learned A, B, C, D matrices have interpretable physical meaning (system dynamics, input coupling, output mapping). This interpretability advantage over black-box LSTM/Transformer models is a genuine differentiator for safety-critical control applications.

**What is novel:**
- The specific assembly of S4D + Mamba gating for robot world models (not individually novel, but the combination for this application is new)
- The empirical finding that SSM's linear state transitions may be more robust to contact dynamics discontinuities than LSTM's nonlinear recurrence (Section 5.8 discussion)
- The multi-step loss formulation with teacher forcing and its ablation (Table 6b)

**What is not novel:**
- The individual components (S4D, Mamba gating, FFT convolution)
- The encoder-SSM-decoder pipeline structure
- The MPC integration framework (standard gradient-based MPC)

**Remaining concern:** The term "world model" is used in a narrow sense (next-state predictor) rather than the broader RL sense (latent dynamics + reward + imagination) established by Ha & Schmidhuber [3] and Dreamer [12]. The paper cites both but does not clearly delineate its narrower scope. A sentence in Section 3.1 clarifying that this is a "dynamics model" or "forward model" in the world model family would improve precision.

---

## Dimension 2: Methodological Rigor (Score: 86/100)

### Assessment

The experimental methodology has improved substantially. The addition of Table 6b (training loss ablation), Table 8b (MuJoCo ablation), the linear baseline in Table 1, and significance testing throughout represent meaningful methodological advances.

**Strengths:**
1. **Two complementary datasets:** Synthetic (near-linear dynamics, 28-dim) and MuJoCo (contact dynamics, 376-dim) provide complementary evaluation. The paper's discussion of why SSM performs differently on each (Section 5.8) is insightful and well-supported by ablation evidence.

2. **Comprehensive ablation:** Architecture ablation (Table 6), training loss ablation (Table 6b), and MuJoCo ablation (Table 8b) cover three orthogonal axes. The finding that gating contributes more on MuJoCo (4.6%) than synthetic (2.3%) is a valuable empirical insight about information selection under contact dynamics.

3. **Statistical rigor:** p-values and 95% CIs reported for key comparisons. Three random seeds used (though five would be preferable for a journal submission).

4. **Training convergence verified:** The paper now states that training converges within 20 epochs with no overfitting, and that validation loss tracks training loss (lines 667-671). This addresses a persistent concern from earlier rounds.

**Remaining methodological concerns:**

1. **Recurrent vs. convolutional mode clarification:** The paper explains why convolutional mode is chosen for MPC (lines 243-246) -- batch forward passes during Adam optimization. However, the 0.9ms single-sample time (Table 3, B=1) should explicitly state which mode is used. For B=1, recurrent mode with O(1) per-step latency would be more appropriate. The paper should clarify: is the 0.9ms measured in convolutional mode (which would be wasteful for B=1) or recurrent mode?

2. **MuJoCo inference time characterization:** Table 8 reports 9.5ms with a note about first-run CUDA compilation (19ms). The standard deviation (0.5ms) is reported, which is good. However, for real-time systems, worst-case latency matters more than mean. A brief note about P95 or max latency would strengthen the deployment argument.

3. **Synthetic dataset size:** 100 episodes x 150 steps is small by modern standards. The paper should acknowledge this more prominently and discuss whether the results would scale with larger datasets.

4. **Three random seeds:** While sufficient for preliminary results, five or more seeds would strengthen statistical claims, especially for the MuJoCo experiments where variance is higher.

---

## Dimension 3: Evidence Sufficiency (Score: 85/100)

### Assessment

The evidence base has reached an adequate level for a journal contribution. The key improvements since Round 12 are:

1. **Table 8 now complete:** All four baselines (LSTM, Transformer, Mamba, SSM-WM) on MuJoCo. This was the most critical missing piece.
2. **Table 1 includes linear baseline:** Direct comparison in the results table, not just discussion text.
3. **Table 6b provides training loss ablation:** lambda and H sweep with quantitative results.
4. **Significance tests on MuJoCo:** SSM vs LSTM (p<0.01), SSM vs Mamba (p=0.27).
5. **MuJoCo MPC frequency projection:** 2.1 Hz estimate based on inference time.

**Evidence matrix:**

| Claim | Evidence | Strength |
|---|---|---|
| SSM enables fast inference | Tables 2, 3 (multiple batch sizes, sequence lengths) | Strong |
| SSM achieves competitive accuracy | Table 1 (synthetic), Table 8 (MuJoCo) | Moderate |
| SSM works for MPC | Table 7 (synthetic MPC) | Moderate |
| Components are necessary | Tables 6, 6b, 8b (three ablation axes) | Strong |
| Training converges | Text description (lines 667-671) | Adequate |
| MuJoCo MPC is feasible | Projected 2.1 Hz (line 888) | Weak (projection only) |

**Remaining evidence gaps:**

1. **No MuJoCo MPC experiment:** This remains the paper's most significant evidence gap. The 2.1 Hz projection (line 888) is reasonable but unvalidated. The paper acknowledges this as a limitation, which is appropriate.

2. **No real hardware deployment:** All experiments use an NVIDIA RTX 3090 (desktop GPU). For a paper targeting embedded deployment on humanoid robots, testing on edge hardware (Jetson Orin, Jetson Nano) would significantly strengthen the practical impact claim. The paper does not discuss power consumption, memory bandwidth, or thermal constraints.

3. **R^2 = 0.592 on MuJoCo:** The paper now discusses this honestly (lines 869-874), noting that 41% unexplained variance may be acceptable for trajectory tracking but insufficient for aggressive manipulation. This is a fair assessment.

4. **No cross-dataset generalization:** Models trained on synthetic data are not tested on MuJoCo and vice versa. This is standard practice but limits the generalizability claims.

---

## Dimension 4: Argument Coherence (Score: 88/100)

### Assessment

The paper now tells a coherent story with clear logical flow:

1. **Problem statement** (Section 1): Lightweight world models needed for humanoid robots with real-time constraints.
2. **Solution** (Section 4): SSM-WM with S4D parameterization and Mamba gating.
3. **Validation** (Section 5): Speed advantage (7x on synthetic), accuracy advantage (6% on MuJoCo), MPC integration (5.1 Hz).
4. **Limitations** (Section 5.8): Honest acknowledgment of 6 distinct limitations.
5. **Future work** (Section 6): Concrete roadmap with 5 directions.

**Argument strengths:**

- The speed-accuracy trade-off is now well-articulated. The paper correctly notes that SSM-WM is slower on synthetic (55% higher MSE) but faster on MuJoCo (6% lower MSE), and provides a plausible explanation based on data characteristics (Section 5.8).
- The linear baseline comparison (Table 1) strengthens the argument that nonlinear modeling is necessary.
- The MuJoCo MPC frequency projection (2.1 Hz) closes the argument loop partially, even without experimental validation.
- The limitations section is comprehensive and honest, covering: real-robot gap, synthetic accuracy deficit, MuJoCo R^2, MuJoCo MPC gap, vision input gap, and dataset scale.

**Cross-disciplinary insight:** The paper's central tension -- speed vs. accuracy, and the finding that this trade-off is dataset-dependent -- has broader implications for embedded AI deployment. In my experience with edge ML systems, this pattern is common: models optimized for speed (pruned, quantized, or architecturally efficient) often perform differently on synthetic vs. real data because real data has more complex structure that benefits from model capacity. The paper's explanation (SSM's linear recurrence is more robust to contact dynamics noise) is plausible and could be tested by adding controlled noise to the synthetic dataset.

**Remaining coherence issues:**

1. **Disjointed evidence base:** The speed advantage is demonstrated on synthetic data (Table 2) while the accuracy advantage is demonstrated on MuJoCo (Table 8). The paper acknowledges this but does not provide a unified experiment. This is acceptable for this revision round but should be a priority for follow-up work.

2. **"World model" terminology:** As noted in Originality, the paper uses "world model" narrowly. A brief clarification would prevent confusion with the broader RL literature.

3. **The 55% MSE gap on synthetic data:** The paper explains this well (LSTM's nonlinear recurrence suits near-linear dynamics), but the magnitude of the gap is still surprising given that SSM-WM achieves R^2=0.945. A brief note about whether the MSE difference is practically significant (e.g., does it affect MPC tracking quality?) would strengthen the argument.

---

## Dimension 5: Writing Quality (Score: 86/100)

### Assessment

The writing quality has improved significantly over the revision rounds. The paper now reads well as a Chinese academic publication.

**Strengths:**
- Clear section structure with logical flow
- Well-formatted tables with consistent notation
- Proper bilingual labels (Chinese + English) for all tables and figures
- Comprehensive limitations section
- Properly formatted references (reference [22] now correct)
- The English abstract is fluent and accurately summarizes the contribution

**Remaining writing issues:**

1. **Abstract specificity:** The abstract states "参数量减少约17%(相比LSTM-2L)" which is good, but the "提速约2倍" claim for B=1 could be more precise (0.9ms vs 2.1ms is 2.3x, not 2x).

2. **Table 6b placement:** The training loss ablation table (6b) is logically part of the ablation section but is numbered separately. Consider integrating it into the main ablation table or giving it a distinct label (e.g., Table 7, renumbering subsequent tables).

3. **Figure quality:** The architecture diagram (Figure 1) and training curve (Figure 2) are drawn using LaTeX picture environment, which produces adequate but not publication-quality figures. For a journal submission, vector graphics (e.g., TikZ or external SVG/PDF) would be preferable.

4. **Hyperparameter search results:** The paper mentions grid search over lambda and H (line 366) but the search results are now shown in Table 6b. This is good, but the text should explicitly connect the grid search to the table (e.g., "As shown in Table 6b, the grid search results indicate...").

5. **Consistency in notation:** The paper uses both "SSM-WM" and "SSM世界模型" interchangeably. Consistent use of one form would improve readability.

---

## Cross-disciplinary Connections and Broader Impact

### Embedded AI Deployment Perspective

From an embedded systems standpoint, the paper's contribution is meaningful but incomplete:

1. **Parameter count (0.24M) is promising:** This fits within the memory constraints of edge devices like Jetson Orin (up to 64GB) or even microcontrollers with external PSRAM. However, the paper does not discuss model quantization (INT8/INT4), which is essential for deployment on resource-constrained hardware.

2. **Inference time (3.8ms on RTX 3090) needs context:** The RTX 3090 has 36 TFLOPS (FP32). An edge GPU like Jetson Orin NX has ~100 TOPS (INT8), roughly 3x less compute. A rough estimate suggests SSM-WM inference on Jetson Orin would be ~10-15ms, which is borderline for real-time control. The paper should acknowledge this gap.

3. **Memory footprint (0.9MB) is excellent:** This is well within the constraints of any modern edge device.

4. **FFT-based convolution on edge hardware:** The paper relies on FFT convolution for O(T log T) complexity. However, FFT implementations on edge NPUs may not be as optimized as on desktop GPUs. This could affect the actual speedup on deployment hardware.

### Connections to Broader ML Systems Research

1. **SSM as a bridge between control theory and deep learning:** The paper's SSM formulation is directly related to classical state-space models in control theory (Kalman filter [29]). This connection is underexplored. The learned A, B, C, D matrices could be interpreted as system dynamics, enabling model-based verification and stability analysis -- a key requirement for safety-critical robotics.

2. **Comparison with model compression approaches:** The paper mentions knowledge distillation, pruning, and quantization (Section 2.3) but does not compare against them. An interesting follow-up would be: can a compressed LSTM (via pruning + quantization) achieve similar speed to SSM-WM while maintaining higher accuracy? This would clarify whether the speed advantage comes from architecture or compression.

3. **Implications for real-time ML systems:** The paper's finding that SSM's speed advantage grows with sequence length (Table 5) has implications beyond robotics. Any real-time ML system processing long sequences (e.g., audio processing, sensor fusion, time-series anomaly detection) could benefit from this architectural choice.

---

## Remaining Concerns and Suggestions

### High Priority

1. **Recurrent vs. convolutional mode for B=1:** Clarify whether the 0.9ms single-sample time (Table 3, B=1) is measured in convolutional or recurrent mode. If convolutional, the actual single-step latency in recurrent mode would be O(1) and potentially faster.

2. **Edge hardware estimation:** Provide a rough estimate of inference time on Jetson Orin or similar edge hardware, based on the FLOP count and typical edge GPU throughput. This would significantly strengthen the deployment argument.

### Medium Priority

3. **Quantization potential:** Discuss whether SSM-WM's architecture is amenable to INT8/INT4 quantization. SSM's linear recurrence may be more quantization-friendly than LSTM's nonlinear gates, which would be an additional advantage.

4. **Stability analysis:** The paper ensures stability via alpha_n > 0 in the S4D parameterization (line 255). A brief discussion of how this constraint affects the learned dynamics (e.g., does it limit the model's ability to capture unstable contact dynamics?) would be valuable for the control theory audience.

5. **MuJoCo R^2 discussion:** The paper now discusses this (lines 869-874), which is good. Consider adding a per-joint or per-state-group analysis to identify which aspects of the humanoid dynamics are hardest to predict.

### Low Priority

6. **Three random seeds:** Consider increasing to five for the final submission.

7. **Figure quality:** Upgrade to TikZ or external vector graphics for publication.

8. **Table numbering:** Consider renumbering Table 6b as a separate table for clarity.

---

## Scoring

| Dimension | Weight | Score | Justification |
|---|---|---|---|
| Originality | 20% | 87 | Well-motivated assembly of established components for an underserved application. The SSM-control theory connection is underexplored but valuable. The "world model" terminology could be more precise. |
| Methodological Rigor | 25% | 86 | Comprehensive experimental design with two datasets, four baselines, three ablation axes, and significance testing. Remaining concerns about recurrent/convolutional mode clarification and edge hardware validation. |
| Evidence Sufficiency | 25% | 85 | All Round 12 evidence gaps addressed except MuJoCo MPC (acknowledged as limitation). No edge hardware experiments, but the paper is honest about this gap. R^2=0.592 on MuJoCo is adequately discussed. |
| Argument Coherence | 15% | 88 | Coherent narrative from problem to solution to validation to limitations. The speed-accuracy trade-off explanation is insightful. The disjointed evidence base (speed on synthetic, accuracy on MuJoCo) is acknowledged. |
| Writing Quality | 15% | 86 | Significant improvement over earlier rounds. Clear structure, proper references, honest limitations. Minor issues with figure quality and notation consistency. |

**Weighted Overall Score:** 0.20(87) + 0.25(86) + 0.25(85) + 0.15(88) + 0.15(86) = 17.4 + 21.5 + 21.25 + 13.2 + 12.9 = **86.25**

---

## Recommendation

**Accept with Minor Revisions**

The paper has reached a quality level suitable for publication at 控制理论与应用. All critical issues from Round 12 have been addressed. The remaining concerns (edge hardware estimation, recurrent/convolutional mode clarification) are minor and can be addressed in a final revision pass or noted as limitations.

**To the authors:** The revision process has been productive. The paper now presents a clear, well-supported contribution at the intersection of state space models and robotics world models. The key strengths are: (1) the architectural design is well-motivated and well-ablated, (2) the experimental evaluation is comprehensive with proper baselines and significance testing, and (3) the limitations section is honest and thorough. The two items I would prioritize for a final revision are: (a) clarifying the recurrent vs. convolutional mode for single-sample inference, and (b) providing a rough estimate of edge hardware performance. These are minor additions that would significantly strengthen the paper's practical impact argument.

---

*Review completed: Round 13*
*Reviewer: R3 (Cross-disciplinary Perspective)*
*Previous R3 score (Round 12): 76.0*
*Current R3 score (Round 13): 86.25*
*All dimensions >= 85: Yes*
