# Devil's Advocate Review Report

## Manuscript Information
- **Title**: 面向人形机器人状态预测的轻量级状态空间世界模型
- **Manuscript ID**: CTA-2026-XXXX
- **Review Date**: 2026-06-03
- **Review Round**: Round 14

---

## Reviewer Information

### Reviewer Role
Devil's Advocate Reviewer

### Reviewer Identity
Dr. adversarial_reviewer — A critical thinker tasked with challenging the paper's core arguments, detecting logical fallacies, and identifying the strongest counter-arguments. No domain loyalty; pure skepticism.

### Review Focus
Core argument challenges, logical fallacy detection, cherry-picking detection, confirmation bias detection, and identification of the strongest counter-arguments against the paper's claims.

---

## Strongest Counter-Argument (250 words)

The paper's central claim is that SSM-WM provides a favorable speed-accuracy trade-off for robot world models. However, this claim is undermined by three fundamental issues:

**First, the speed advantage is overstated.** The paper reports 7.3x speedup over LSTM on RTX 3090 GPU (Table 2), but this comparison is misleading. LSTM is notoriously difficult to parallelize on GPUs, while SSM's FFT-based computation is embarrassingly parallel. On CPU or embedded hardware (where most robots actually run), the advantage would likely be much smaller or even reversed. The paper does not provide any non-GPU benchmarks, making the speed claims hardware-specific and potentially non-generalizable.

**Second, the accuracy gap is conveniently downplayed.** On the synthetic dataset, SSM-WM has 157% higher MSE than LSTM (2.72 vs 1.06). The paper dismisses this by saying the synthetic dataset has "near-linear dynamics" (line 834), but this is circular reasoning: the paper designed the synthetic dataset, and now uses its characteristics to explain away poor performance. If the synthetic dataset is not representative, why use it at all? The MuJoCo results (6% better than LSTM) are more favorable, but the R² of 0.592 means 41% of variance is unexplained—hardly a ringing endorsement.

**Third, the "first" claim is misleading.** The paper claims to be "首次将S4D风格的对角SSM参数化与Mamba风格的门控块结构相结合" (line 36, abstract). But S4D + Mamba gating is a straightforward combination of two existing techniques. Calling this "first" implies novelty where there is only engineering. The real novelty would be if the paper demonstrated something new about SSM's capabilities for robot dynamics, which it does not—the paper only shows that SSM works reasonably well, not that it works better than alternatives in any fundamental way.

---

## Issue List

### CRITICAL Issues

**C1: Data Inconsistency Between Text and Table**
- **Dimension**: Evidence Sufficiency
- **Location**: Section 5.2, lines 481-483 vs Table 1
- **Description**: The text reports SSM-WM MSE=1.32×10⁻³ and LSTM-WM MSE=0.85×10⁻³, but Table 1 shows 2.72×10⁻³ and 1.06×10⁻³ respectively. These are fundamentally different values.
- **Impact**: This is a data integrity issue that would likely lead to rejection. If the text values are from a different experiment, this must be clearly stated. If they are errors, the paper's conclusions may need to be revised.
- **Counter-argument**: If the text values are correct, SSM-WM's accuracy gap on synthetic data is only 55% (1.32 vs 0.85), not 157% (2.72 vs 1.06). This would significantly change the paper's narrative about the speed-accuracy trade-off.

**C2: Hardware-Specific Speed Claims**
- **Dimension**: Methodological Rigor
- **Location**: Section 5.1, line 435; Tables 2-3
- **Description**: All inference time measurements are on NVIDIA RTX 3090 GPU. The 7.3x speedup claim is specific to this hardware and may not generalize.
- **Impact**: Real robots typically use embedded processors (Jetson, ARM), not desktop GPUs. The paper's speed claims may be irrelevant for actual deployment.
- **Counter-argument**: The paper should either (1) provide embedded hardware benchmarks or (2) explicitly state that the speed claims are GPU-specific and may not transfer to embedded platforms.

### MAJOR Issues

**M1: Circular Reasoning on Synthetic Dataset**
- **Dimension**: Argument Coherence
- **Location**: Section 5.8, lines 832-839
- **Description**: The paper designed the synthetic dataset with near-linear dynamics (coefficient 0.1), then uses this design choice to explain away SSM-WM's poor performance (157% higher MSE than LSTM). This is circular reasoning.
- **Impact**: If the synthetic dataset is not representative of real robot dynamics, the paper should not use it as a primary validation. The MuJoCo results should be the primary focus.
- **Counter-argument**: The paper should either (1) validate on multiple synthetic datasets with varying nonlinearity or (2) acknowledge that the synthetic dataset results are not generalizable.

**M2: Low R² on MuJoCo is Glossed Over**
- **Dimension**: Evidence Sufficiency
- **Location**: Table 8, lines 860-864
- **Description**: SSM-WM achieves R²=0.592 on MuJoCo, meaning 41% of variance is unexplained. The paper briefly discusses this (lines 860-864) but does not adequately address the implications.
- **Impact**: For a world model intended for MPC control, 41% unexplained variance is a significant limitation. The paper should more honestly assess whether this accuracy level is sufficient for practical applications.
- **Counter-argument**: The paper claims SSM-WM is "基本可用" for trajectory tracking (line 862), but does not provide evidence that this accuracy level is sufficient for real-world control tasks.

**M3: "First" Claim is Overstated**
- **Dimension**: Originality
- **Location**: Abstract, line 36; Introduction, line 126
- **Description**: The paper claims to be "首次将S4D风格的对角SSM参数化与Mamba风格的门控块结构相结合." But S4D + Mamba gating is a straightforward combination of two existing techniques.
- **Impact**: Calling this "first" implies novelty where there is only engineering. The real novelty would be demonstrating something new about SSM's capabilities for robot dynamics.
- **Counter-argument**: The paper should reframe the contribution as "the first systematic study of SSM-based world models for humanoid robot state prediction" rather than claiming novelty in the architecture itself.

### MINOR Issues

**m1: Missing Ablation of SSM vs. Other Sequence Models**
- **Dimension**: Methodological Rigor
- **Location**: Section 5.6
- **Description**: The ablation studies test architecture components (gating, residual, depth) but do not test whether SSM is the right choice for the sequence modeling component. An ablation comparing SSM with 1D convolution or attention within the same architecture would strengthen the contribution.

**m2: No Discussion of Failure Modes**
- **Dimension**: Argument Coherence
- **Location**: Section 5.8
- **Description**: The paper does not discuss when SSM-WM might fail or underperform. For example, what happens with highly discontinuous dynamics, adversarial inputs, or distribution shift?

**m3: Training Data Efficiency Not Analyzed**
- **Dimension**: Evidence Sufficiency
- **Location**: Section 5.1
- **Description**: The paper uses 400 training episodes for synthetic and 160 for MuJoCo. The data efficiency of SSM-WM compared to LSTM is not analyzed. If SSM-WM requires more data to achieve comparable accuracy, this is a practical limitation.

---

## Ignored Alternative Explanations/Paths

1. **Why not use a hybrid approach?** Instead of choosing between SSM and LSTM, a hybrid model (e.g., SSM for fast inference + LSTM for accuracy-critical predictions) might achieve better speed-accuracy trade-off.

2. **Why not use model compression on LSTM?** If the goal is speed, applying pruning or quantization to LSTM might achieve similar speedup without sacrificing accuracy.

3. **Why not use neural ODE?** For continuous dynamics, neural ODE might be more natural than discrete SSM. The paper does not discuss why SSM is preferred over neural ODE.

4. **Why not use graph neural networks?** Robot dynamics have a natural graph structure (joints and links). GNNs might capture this structure better than sequence models.

---

## Missing Stakeholder Perspectives

1. **Robot practitioners**: The paper focuses on inference speed but does not discuss deployment complexity (e.g., library dependencies, CUDA requirements, model size on disk).

2. **Safety engineers**: For safety-critical applications, the 41% unexplained variance on MuJoCo is a concern. The paper does not discuss how to handle prediction uncertainty.

3. **Embedded systems engineers**: The paper does not discuss memory bandwidth, cache efficiency, or energy consumption, which are critical for embedded deployment.

---

## Observations (Non-Defects)

1. **The MPC integration is a genuine strength.** The paper goes beyond typical ML papers by demonstrating real-time control capability. This is valuable for the control community.

2. **The ablation studies are thorough.** The dual ablation (architecture + training loss) provides useful guidance for practitioners.

3. **The statistical reporting is commendable.** The use of 5 seeds, paired t-tests, CIs, and effect sizes is rare in the robotics literature and sets a good example.

---

## Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 58 | Weak | "First" claim is overstated; architecture is combination of existing techniques |
| Methodological Rigor (25%) | 62 | Adequate | Systematic experiments but text-table inconsistency and hardware-specific claims |
| Evidence Sufficiency (25%) | 60 | Adequate | Good experiments but low MuJoCo R² and missing embedded benchmarks |
| Argument Coherence (15%) | 68 | Adequate | Circular reasoning on synthetic dataset; low R² glossed over |
| Writing Quality (15%) | 68 | Adequate | Data inconsistency is a serious flaw |
| **Weighted Average** | **62.8** | **Major Revision** | |

---

## Devil's Advocate Verdict

**Overall Assessment**: The paper has genuine contributions (MPC integration, systematic experiments, statistical rigor) but is undermined by data inconsistency, hardware-specific speed claims, and circular reasoning on the synthetic dataset. The "first" claim is overstated, and the low R² on MuJoCo limits practical applicability.

**Recommendation**: **Major Revision** — The data inconsistency must be fixed, the speed claims must be qualified (GPU-specific), and the synthetic dataset discussion must be revised to avoid circular reasoning. The low R² on MuJoCo should be more honestly discussed.

**If Not Fixed**: If the data inconsistency is not resolved and the speed claims are not qualified, the paper should be rejected. These are fundamental issues that undermine the paper's credibility.

---

*End of Devil's Advocate Review Report*
