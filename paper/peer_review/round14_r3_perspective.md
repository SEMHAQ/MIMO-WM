# Peer Review Report — Reviewer 3 (Perspective)

## Manuscript Information
- **Title**: 面向人形机器人状态预测的轻量级状态空间世界模型
- **Manuscript ID**: CTA-2026-XXXX
- **Review Date**: 2026-06-03
- **Review Round**: Round 14

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 3 (Cross-Disciplinary Perspective)

### Reviewer Identity
Dr. Wang, Associate Professor at a computer science department with joint appointment in a robotics institute. Expertise in efficient deep learning, edge computing, and embedded AI systems. Published 30+ papers on model compression, neural architecture search, and TinyML. Brings a cross-disciplinary perspective from the efficient computing community.

### Review Focus
Cross-disciplinary connections (efficient computing, edge AI), practical deployment implications, broader impact of the research, and whether the paper's claims are relevant to real-world applications.

---

## Overall Assessment

### Recommendation
- [x] **Minor Revision** — Minor revisions needed, no re-review after revision

### Confidence Score
4 — Mostly within my area of expertise, high confidence

### Summary Assessment
This paper proposes SSM-WM for humanoid robot state prediction, with a focus on computational efficiency for real-time control. From an efficient computing perspective, the paper makes a solid contribution: the 7.3x speedup over LSTM and 17% parameter reduction are meaningful for resource-constrained robot platforms. The MPC integration demonstrates practical applicability. However, the paper could be strengthened by: (1) discussing deployment on actual embedded hardware (not just GPU), (2) comparing with model compression techniques (pruning, quantization, distillation), and (3) analyzing memory bandwidth and energy consumption, which are critical for battery-powered robots. The data inconsistency between text and table is a serious issue that must be fixed.

---

## Strengths

### S1: Computational Efficiency for Real-Time Control
The paper demonstrates that SSM-WM achieves 7.3x speedup over LSTM in batch mode (Table 2) and 2.3x speedup in single-sample mode (Table 3, B=1). For robot control where 1kHz sampling is required, the 0.9ms inference time (Table 3) is a strong result. The paper correctly identifies that computational efficiency is a hard constraint for real-time control (line 392).

### S2: Parameter Efficiency
SSM-WM has only 0.14M parameters (Table 1), which is 17% fewer than LSTM (0.29M) and 77% fewer than Transformer (0.62M). This parameter efficiency is important for deployment on resource-constrained robot platforms where memory is limited.

### S3: Practical MPC Integration
The MPC integration (Section 4.4) demonstrates that SSM-WM can be used in a real control loop, achieving 5.1Hz on synthetic data and 2.1Hz on MuJoCo. This is a practical result that goes beyond typical machine learning papers that only report offline metrics.

### S4: Systematic Architecture Exploration
The ablation studies (Tables 6, 6b) provide useful guidance for practitioners: (1) gating mechanism is important (2.3% MSE increase when removed), (2) deeper networks (L=6) improve accuracy but increase latency, (3) hidden dimension D=128 is a good default. This helps practitioners make informed design choices.

---

## Weaknesses

### W1: Data Inconsistency
**Problem**: Text reports SSM-WM MSE=1.32×10⁻³ (line 483) but Table 1 shows 2.72×10⁻³. This is a fundamental data integrity issue.
**Why it matters**: From an efficient computing perspective, the speed-accuracy trade-off analysis depends on accurate accuracy numbers. If the text values are wrong, the trade-off analysis is also wrong.
**Suggestion**: Fix the data inconsistency and re-analyze the speed-accuracy trade-off.
**Severity**: Critical

### W2: Missing Embedded Hardware Evaluation
**Problem**: All inference time measurements are on NVIDIA RTX 3090 GPU (line 435). Real humanoid robots typically use embedded processors (e.g., NVIDIA Jetson, ARM Cortex, or custom ASICs), not desktop GPUs.
**Why it matters**: The 7.3x speedup on RTX 3090 may not translate to embedded hardware. GPU parallelism benefits batch processing, but robot control often requires single-sample inference (B=1).
**Suggestion**: (1) Add inference time measurements on NVIDIA Jetson or similar embedded hardware. (2) If embedded hardware is not available, discuss the expected performance characteristics (e.g., memory bandwidth, compute density) and how SSM-WM's architecture is suited for embedded deployment.
**Severity**: Major

### W3: Missing Comparison with Model Compression Techniques
**Problem**: The paper compares SSM-WM with other architectures (LSTM, Transformer, Mamba) but does not compare with model compression techniques applied to these baselines (e.g., pruned LSTM, quantized Transformer, distilled Mamba).
**Why it matters**: If a compressed LSTM achieves similar speed to SSM-WM with better accuracy, the motivation for SSM-WM is weakened.
**Suggestion**: Add a brief discussion of how SSM-WM compares with compressed baselines, or at least acknowledge this as a limitation.
**Severity**: Minor

### W4: Energy Consumption Not Analyzed
**Problem**: The paper reports inference time and memory usage but not energy consumption. For battery-powered robots, energy efficiency is as important as speed.
**Why it matters**: SSM-WM's FFT-based computation may have different energy characteristics than LSTM's sequential computation. This could be an advantage or disadvantage.
**Suggestion**: Add a brief discussion of expected energy efficiency characteristics, or note this as future work.
**Severity**: Minor

### W5: Single-Hardware Platform
**Problem**: All experiments are on a single GPU (RTX 3090). Results may vary on different hardware (e.g., AMD GPU, CPU-only, TPU).
**Why it matters**: The paper's claims about computational efficiency are hardware-specific. Different hardware architectures may favor different model architectures.
**Suggestion**: Acknowledge this limitation and discuss how the results might generalize to other platforms.
**Severity**: Minor

---

## Detailed Comments

### Title & Abstract
- Title is appropriate
- Abstract could lead with the efficiency result (7x speedup)

### Introduction
- Problem motivation is clear
- The "three unique challenges" (lines 120-122) are well-articulated

### Literature Review
- SSM coverage is good
- **Gap**: Missing efficient computing literature (model compression, TinyML, edge AI)

### Methodology
- Architecture is clearly described
- Training procedure is well-documented
- **Suggestion**: Add a discussion of computational complexity in terms of FLOPs, not just Big-O notation

### Results
- Main results are clear
- **Critical issue**: Text-table inconsistency (see W1)
- MPC results are strong

### Discussion
- Speed-accuracy trade-off is well-discussed
- **Missing**: Discussion of deployment on embedded hardware
- **Missing**: Comparison with model compression techniques

### Conclusion
- Conclusion is accurate
- Future work should mention embedded deployment

---

## Questions for Authors

1. **Embedded hardware**: Have you tested SSM-WM on NVIDIA Jetson or similar embedded hardware? If not, what is the expected performance?

2. **Model compression**: Have you considered applying pruning or quantization to SSM-WM to further reduce its computational cost?

3. **Energy efficiency**: Can you discuss the expected energy efficiency of SSM-WM compared to LSTM for battery-powered robots?

4. **CPU inference**: For robots without GPUs, can you provide CPU inference time estimates?

---

## Minor Issues

### Literature
- Add references to efficient computing literature (e.g., TinyML, edge AI)
- Add references to model compression techniques

### Figures and Tables
- Table 2: Add a column for FLOPs to complement inference time
- Table 3: Add CPU inference times if available

### Terminology
- "轻量级" (lightweight) should be quantified more precisely (e.g., "17% fewer parameters than LSTM")

---

## Dimension Scores

| Dimension | Score (0-100) | Descriptor | Notes |
|-----------|--------------|------------|-------|
| Originality (20%) | 70 | Adequate | Novel application domain; architecture is combination of existing techniques |
| Methodological Rigor (25%) | 72 | Adequate | Systematic experiments but text-table inconsistency and missing embedded evaluation |
| Evidence Sufficiency (25%) | 72 | Adequate | Good GPU results but missing embedded hardware and energy evaluation |
| Argument Coherence (15%) | 76 | Strong | Clear logical flow; practical relevance well-motivated |
| Writing Quality (15%) | 72 | Adequate | Generally clear but has data inconsistency |
| Significance & Impact | 74 | Strong | Practical relevance for real-time robot control; clear speed advantage |
| **Weighted Average** | **72.4** | **Minor Revision** | |

---

*End of R3 Perspective Review Report*
