# Round 6 Peer Review -- SSM-WM: Lightweight State Space World Model for Humanoid Robot State Prediction

**Target Journal:** 控制理论与应用 (CTA)
**Date:** 2026-06-02
**Paper Version:** Round 5 Revision

---

## Overview

This paper proposes SSM-WM, a lightweight world model based on diagonal state space models (S4D-style parameterization + Mamba-style gated blocks) for humanoid robot state prediction and model predictive control. The paper has undergone 5 rounds of revision, with the latest changes including batch-qualified speedup claims, a 4-layer LSTM baseline, sequential table numbering, standard deviations in ablation results, MPC error interpretation, short-sequence deployment guidance, and SSM-WM vs Mamba-WM comparison.

---

## Reviewer 1: Editor-in-Chief (EIC) -- Overall Quality & Scope Fit

### Dimension Scores

| Dimension | Score | Rationale |
|---|---|---|
| Originality | 62 | Application of SSM to robot world models is incremental; the S4D+Mamba-gated combination is a known recipe applied to a new domain. Novelty lies in the integration rather than architectural invention. |
| Methodological Rigor | 68 | The method is well-formulated with clear complexity analysis. However, the experimental methodology has significant gaps (synthetic data, small dataset, no cross-validation). |
| Evidence Sufficiency | 55 | Experiments are limited to synthetic data (100 episodes). No real robot, no MuJoCo, no visual input. The MPC experiment uses a single trajectory type. This is the paper's most critical weakness for CTA. |
| Argument Coherence | 75 | The paper is logically structured: motivation -> method -> experiments -> discussion. Limitations are honestly stated. The connection between SSM advantages and real-time control needs is clear. |
| Writing Quality | 78 | The paper is well-written in standard academic Chinese. Figures and tables are clear. The bilingual abstract is properly formatted. Some notation could be tighter. |

### Top 3 Issues

1. **Synthetic data is insufficient for CTA.** A control theory journal expects validation on physically meaningful systems. 100 episodes of linear dynamics with tanh coupling is toy-level. At minimum, MuJoCo humanoid experiments are needed. The authors acknowledge this as a limitation but do not address it.

2. **SSM-WM accuracy gap is not adequately justified.** MSE is 55% worse than LSTM-WM. The authors argue this is acceptable for real-time control, but do not provide a formal analysis of how prediction error propagates to control performance. The MPC tracking MSE (0.0043 vs 0.0032, 34% gap) is better but still substantial.

3. **Missing comparison with recent SSM-robotics work.** The paper does not compare with or discuss recent work applying SSM/Mamba to robotics (e.g., Mamba-based policy learning, SSM for trajectory prediction). The related work section focuses on general SSM literature.

### Suggested Fixes

- Add MuJoCo humanoid walker experiments (even 2-3 locomotion tasks).
- Provide formal error propagation analysis: how does prediction MSE translate to control cost?
- Add a paragraph discussing recent SSM applications in robotics beyond general sequence modeling.

---

## Reviewer 2: R1 (Methodology Expert) -- Technical Rigor & Novelty

### Dimension Scores

| Dimension | Score | Rationale |
|---|---|---|
| Originality | 58 | The architecture is a straightforward combination of known components (S4D diagonal SSM + Mamba gating). No novel algorithmic contribution. The contribution is engineering: applying existing SSM building blocks to robot state prediction. |
| Methodological Rigor | 65 | Complexity analysis is correct. Training procedure is standard. However: (a) no learning curve analysis, (b) no convergence guarantees discussed, (c) the multi-step loss formulation uses teacher forcing without scheduled sampling, (d) no analysis of gradient flow through FFT convolution. |
| Evidence Sufficiency | 60 | Ablation study is comprehensive for architecture choices but missing: loss component ablation (single vs multi-step), scheduled sampling ablation, effect of noise level in training data. |
| Argument Coherence | 72 | The technical argument flows well. The choice of convolutional mode for MPC is well-justified. The complexity analysis is rigorous. |
| Writing Quality | 76 | Mathematical notation is consistent. Equations are properly formatted. Some derivations could be more detailed (e.g., the discrete SSM convolution kernel derivation). |

### Top 3 Issues

1. **LSTM-WM-4L performs worse than LSTM-WM-2L (MSE 1.02 vs 0.85).** This is counterintuitive and undermines the "fair comparison" motivation. The authors do not explain this degradation. Is it overfitting? Optimization difficulty? This needs investigation -- if deeper LSTM is worse, why is deeper SSM (L=6) better?

2. **No ablation on loss components.** The multi-step loss weight (lambda=0.5) and horizon (H=8) are set without justification. What happens with lambda=0 (single-step only)? With H=16? These are critical hyperparameters that directly affect multi-step prediction quality.

3. **Teacher forcing without scheduled sampling.** The multi-step loss uses ground-truth states as input (teacher forcing). This causes exposure bias during autoregressive MPC rollout. The paper should either implement scheduled sampling or at least discuss this limitation.

### Suggested Fixes

- Investigate and explain the LSTM-WM-4L degradation (overfitting analysis, training curves).
- Add loss component ablation: lambda in {0, 0.1, 0.5, 1.0} and H in {4, 8, 16}.
- Add a paragraph on exposure bias and potential scheduled sampling remedy.

---

## Reviewer 3: R2 (Domain Expert -- Embodied AI / Robotics) -- Relevance & Practical Significance

### Dimension Scores

| Dimension | Score | Rationale |
|---|---|---|
| Originality | 60 | Applying SSM to robot world models is a reasonable idea, but the execution is limited to state-space prediction without visual input. Modern embodied AI systems are primarily vision-based. |
| Methodological Rigor | 63 | The MPC integration is a good practical contribution. However, the MPC experiment is too simple: single reference trajectory, no disturbance, no model mismatch analysis. |
| Evidence Sufficiency | 50 | This is the weakest dimension. No real robot. No MuJoCo. No visual input. No comparison with robot-specific world models (Dreamer, PlaNet). The synthetic data is not representative of real humanoid dynamics. |
| Argument Coherence | 70 | The argument that real-time control requires fast inference is valid. The 5.1Hz control frequency is reasonable for motion planning. But the paper does not compare with analytical model-based MPC, which is the standard in robotics. |
| Writing Quality | 75 | Writing is clear. The MPC formulation is standard and well-explained. |

### Top 3 Issues

1. **No comparison with analytical model-based MPC.** In robotics, the standard MPC uses rigid-body dynamics (URDF-based). The paper should compare SSM-WM-MPC against an analytical model MPC baseline to show when/why learned models are preferable.

2. **Synthetic dynamics are too simple.** The dynamics (linear + 0.1*tanh coupling + Gaussian noise) do not capture the key challenges of real humanoid control: contact discontinuities, friction, underactuation, high-dimensional configuration space. A MuJoCo humanoid (e.g., Humanoid-v4 from Gym) would be far more convincing.

3. **No robustness evaluation.** Real robots face sensor noise, model mismatch, and external disturbances. The paper should test: (a) prediction accuracy under varying noise levels, (b) MPC performance with model mismatch (train on one dynamics, test on another), (c) sensitivity to initial conditions.

### Suggested Fixes

- Add analytical model MPC baseline (even a simple linearized model).
- Replace or supplement synthetic data with MuJoCo Humanoid-v4 or similar.
- Add robustness experiments: noise sensitivity, model mismatch, disturbance rejection.

---

## Reviewer 4: R3 (Broader Impact & Completeness) -- Evaluation Comprehensiveness

### Dimension Scores

| Dimension | Score | Rationale |
|---|---|---|
| Originality | 65 | The paper fills a gap (SSM for robot world models) but the gap is narrow. The contribution is more engineering than scientific. |
| Methodological Rigor | 70 | The evaluation protocol is reasonable (3 seeds, multiple metrics, ablation). Missing: cross-validation, confidence intervals on speedup claims, end-to-end timing analysis. |
| Evidence Sufficiency | 58 | Table 4 (sequence length) lacks standard deviations. MPC experiment lacks error bars. Only one trajectory type tested. No cross-dataset generalization. |
| Argument Coherence | 73 | The paper is honest about limitations. The discussion section is balanced. The speedup qualification (B=1 vs B=64) is a good improvement. |
| Writing Quality | 78 | Tables and figures are well-designed. The appendix provides useful implementation details. |

### Top 3 Issues

1. **Table 4 (sequence length) lacks standard deviations.** All other tables report mean +/- std. Table 4 reports only means. This is inconsistent and hides variability. If the authors ran 3 seeds, they should report std for Table 4 as well.

2. **MPC experiment lacks error bars and statistical testing.** Table 6 reports single numbers (no std). With only one trajectory type, we cannot tell if the differences are statistically significant. At minimum, test on 10+ random trajectories and report mean +/- std with paired t-tests.

3. **No end-to-end system evaluation.** The paper measures inference time and MPC loop time separately, but does not report end-to-end system performance (perception -> prediction -> control -> execution). For a robotics paper, the full pipeline matters.

### Suggested Fixes

- Add standard deviations to Table 4 (consistent with other tables).
- Run MPC on 10+ random trajectories, report mean +/- std, add statistical significance tests.
- Add a brief discussion of end-to-end latency breakdown (if applicable).

---

## Reviewer 5: Devil's Advocate -- Attack the Claims

### Dimension Scores

| Dimension | Score | Rationale |
|---|---|---|
| Originality | 45 | This is essentially "S4D + Mamba gating applied to a toy robot problem." The authors did not invent any of the components. The contribution is the application, but the application is on synthetic data. |
| Methodological Rigor | 55 | Multiple methodological issues: (1) No learning curves -- we don't know if models have converged in 20 epochs. (2) The 7x speedup is partly an artifact of LSTM's poor GPU utilization at large batches, not SSM's inherent advantage. (3) The synthetic dynamics are trivially learnable. |
| Evidence Sufficiency | 45 | The evidence is weak: 100 episodes, synthetic data, one trajectory for MPC, no real-world validation. The 55% accuracy gap is papered over with "acceptable for real-time control" without proof. |
| Argument Coherence | 60 | The argument structure is logical but the conclusions are overstated given the evidence. "验证了SSM-WM在具身智能场景下的有效性和实用性" (validates effectiveness and practicality) is too strong for synthetic-only experiments. |
| Writing Quality | 72 | Writing is competent but some claims need softening. |

### Top 3 Issues

1. **The "7x speedup" is misleading in context.** At B=1 (the realistic deployment scenario for a single robot), the speedup is only 2x. The 7x speedup at B=64 is an artifact of LSTM's poor batching efficiency, not SSM's architectural advantage. Furthermore, 0.9ms vs 2.1ms is a 1.2ms absolute difference -- negligible for most control loops. The paper should emphasize B=1 results more prominently.

2. **The 55% accuracy gap is the elephant in the room.** MSE 1.32 vs 0.85 is a 55% relative increase. The authors argue R^2=0.945 is "good enough," but R^2 is scale-dependent and can be misleading. For a control system, prediction error compounds over time. The multi-step prediction error (which is what MPC actually uses) is likely much worse than the single-step error reported in Table 1. The paper does not report multi-step prediction accuracy.

3. **The synthetic data makes all results questionable.** Linear dynamics with tanh coupling is trivially learnable by any neural network. The fact that SSM-WM achieves R^2=0.945 on this toy problem tells us nothing about real-world performance. The entire paper's contribution collapses if we cannot trust the results to transfer. The authors' own limitation statement acknowledges this but does not address it.

### Suggested Fixes

- Relegate the B=64 speedup to supplementary; lead with B=1 results.
- Report multi-step prediction error (H=1,4,8,16 steps) as a separate metric.
- Either add MuJoCo experiments or explicitly frame the paper as a "proof of concept" rather than a validated system.

---

## Summary Score Matrix

| Reviewer | Originality | Rigor | Evidence | Coherence | Writing | Weighted Avg |
|---|---|---|---|---|---|---|
| EIC | 62 | 68 | 55 | 75 | 78 | 67.6 |
| R1 (Method) | 58 | 65 | 60 | 72 | 76 | 66.2 |
| R2 (Domain) | 60 | 63 | 50 | 70 | 75 | 63.6 |
| R3 (Breadth) | 65 | 70 | 58 | 73 | 78 | 68.8 |
| Devil's Advocate | 45 | 55 | 45 | 60 | 72 | 55.4 |
| **Mean** | **58.0** | **64.2** | **53.6** | **70.0** | **75.8** | **64.3** |

### Overall Assessment

**Score: 64.3/100 -- Borderline Reject, Major Revision Required**

The paper addresses a relevant problem (lightweight world models for real-time robot control) and the SSM-based approach is technically sound. The Round 5 revisions (batch-qualified speedup, LSTM-4L baseline, std in ablation, MPC error interpretation) represent genuine improvements. However, the fundamental weakness remains: **all experiments are on synthetic data that is too simple to draw meaningful conclusions about real-world robotics.**

### Priority Fixes for Acceptance

1. **[Critical]** Add MuJoCo humanoid experiments (even 1-2 locomotion tasks). This single change would significantly strengthen the paper.
2. **[Critical]** Report multi-step prediction accuracy (not just single-step MSE). This is what MPC actually uses.
3. **[Important]** Add standard deviations to Table 4 and error bars to Table 6.
4. **[Important]** Investigate and explain LSTM-WM-4L degradation.
5. **[Desirable]** Add loss component ablation (lambda, H).
6. **[Desirable]** Emphasize B=1 results over B=64 in the abstract and conclusions.
