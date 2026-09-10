"""② 纯实数部署递推核 (Deployment kernel) —— 对 MIMO-WM 的部署形态建模与验证.

目标:
  1. 实现一个与【训练/报告所用卷积路径】在数值上等价的纯实数单步递推核
     (ONNX/TFLite 可导出, 真实部署形态), 而不是用仓库现有 exp-ZOH 复数递推
     (后者与卷积路径并不等价, 详见 README §3).
  2. 数值等价性验证: |部署核输出 - conv 输出| 应 ~1e-5 量级.
  3. CPU 部署时延/内存: 单步延迟 + 整窗(T=8/32)延迟, 线程 1/4.
  4. 尝试 torch.onnx.export 导出(若 onnx 可用).

数学背景(由 src/models/ssm_world_model.py _forward_conv 反推):
  训练路径的因果卷积核  K[d,t] = sum_n C·B·dt · Re((dt·A)^t),  dt·A = ΔA.
  等价 LTI 递推(实数域, 每 (d,n) 复模式拆实/虚两部):
     s_r <- q_r*s_r - q_i*s_i + g*x ;   s_i <- q_i*s_r + q_r*s_i ;   g = B*dt
     y_ssm = sum_n C_n*s_r_n + D*x      (C, B 为实数参数)
  这正是"训练=卷积、部署=同核递推"的同构关系, 也是论文 Thm.1 应当成立的形态.
  注意: 仓库现有 DiagSSM._forward_recurrent 用的是 exp(ΔA) ZOH, 与此核不同,
        故本文实现是唯一与训练精度一致的部署形态.

用法:  python revision_experiments/scripts/deploy_mimo.py
产物:  revision_experiments/results/deploy_kernel_verify.json
"""
import os, sys, json, time, platform
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath('.'))
from src.models.mimo_world_model import MIMOWorldModel

STATE_DIM, ACTION_DIM = 348, 17
DM, DS, NL = 96, 16, 2


class _DeployLayer(nn.Module):
    """单层部署单元: LayerNorm + 实数递推系数(buffer) + gate/output Linear."""

    def __init__(self, blk):
        super().__init__()
        ssm = blk.ssm
        dt = torch.exp(ssm.log_dt)                      # (D,)
        A_r = -torch.exp(ssm.log_A_real)                # (D,N)
        self.norm = blk.norm
        self.gate = blk.gate
        self.out = blk.output
        self.register_buffer('q_r', (A_r.detach() * dt[:, None]).clone())      # Re(ΔA)
        self.register_buffer('q_i', (ssm.A_imag.detach() * dt[:, None]).clone())  # Im(ΔA)
        self.register_buffer('g', (ssm.B.detach() * dt[:, None]).clone())      # 输入增益
        self.register_buffer('c', ssm.C.detach().clone())
        self.register_buffer('d', ssm.D.detach().clone())

    def forward(self, x, s_r, s_i):
        xn = self.norm(x)
        nr = self.q_r * s_r - self.q_i * s_i + self.g * xn.unsqueeze(-1)
        ni = self.q_i * s_r + self.q_r * s_i
        y = (self.c * nr).sum(-1) + self.d * xn
        y = self.out(y) * torch.sigmoid(self.gate(y))
        return x + y, nr, ni


class RealMIMOWorldModel(nn.Module):
    """MIMO-WM 的纯实数单步递推实现(与 conv 训练路径等价), 全部为 ONNX 标准算子."""

    def __init__(self, mimo: MIMOWorldModel):
        super().__init__()
        self.state_dim = mimo.decoder[2].out_features
        # 编码器 / 解码器 (Linear,GELU,Linear)
        self.enc0 = mimo.encoder[0]
        self.enc2 = mimo.encoder[2]
        self.dec0 = mimo.decoder[0]
        self.dec2 = mimo.decoder[2]
        # 每层 MIMO: LN + 实数递推系数 + gate/output Linear
        self.layers = nn.ModuleList([_DeployLayer(blk) for blk in mimo.backbone])

    def _step_layer(self, L, x, s_r, s_i):
        """单步更新一层: 返回 (x', s_r', s_i'). 全部实数运算."""
        return L(x, s_r, s_i)

    def step_token(self, s, a, states_r, states_i):
        """处理一个时间步: 编码 + L 层递推. 返回 (token_out_hidden, 更新后状态)."""
        u = self.enc2(torch.nn.functional.gelu(self.enc0(torch.cat([s, a], dim=-1))))
        new_r, new_i = [], []
        for li, L in enumerate(self.layers):
            u, sr, si = self._step_layer(L, u, states_r[li], states_i[li])
            new_r.append(sr); new_i.append(si)
        return u, new_r, new_i

    def forward_window(self, states, actions):
        """复现 conv 前向语义: states (B,T,S), actions (B,T-1,A) -> pred (B,S)."""
        B, T = states.shape[:2]
        device = states.device
        # 首步动作补零(与 src 对齐)
        a0 = torch.zeros(B, 1, actions.shape[-1], device=device)
        act = torch.cat([a0, actions], dim=1)
        N = self.layers[0].g.shape[-1]
        s_r = [torch.zeros(B, DM, N, device=device) for _ in self.layers]
        s_i = [torch.zeros(B, DM, N, device=device) for _ in self.layers]
        h_last = None
        for t in range(T):
            h_last, s_r, s_i = self.step_token(states[:, t], act[:, t], s_r, s_i)
        delta = self.dec2(torch.nn.functional.gelu(self.dec0(h_last)))
        return states[:, -1] + delta

    def forward(self, states, actions):
        return self.forward_window(states, actions)


def build(seed=0):
    torch.manual_seed(seed)
    mimo = MIMOWorldModel(STATE_DIM, ACTION_DIM, d_model=DM, d_state=DS, n_layers=NL).eval()
    return mimo, RealMIMOWorldModel(mimo).eval()


def max_diff(a, b):
    return (a - b).abs().max().item()


def main():
    torch.set_num_threads(1)
    results = {'platform': platform.platform(), 'config': {'D': DM, 'N': DS, 'L': NL},
               'state_dim': STATE_DIM, 'action_dim': ACTION_DIM, 'checks': {}}
    worst = 0.0
    for T in (8, 32):
        for seed in (0, 7):
            mimo, dep = build(seed)
            torch.manual_seed(seed + 1)
            st = torch.randn(3, T, STATE_DIM)
            ac = torch.randn(3, T - 1, ACTION_DIM)
            with torch.no_grad():
                y_conv = mimo(st, ac)
                y_rec = dep(st, ac)
            d = max_diff(y_conv, y_rec)
            worst = max(worst, d)
            results['checks'][f'T{T}_seed{seed}'] = d
            print(f'T={T:3d} seed={seed}  max|conv - deploy_kernel| = {d:.3e}', flush=True)

    # ---- CPU 时延 ----
    mimo, dep = build(0)
    dep_t = dep
    lat = {}
    for nth in (1, 4):
        torch.set_num_threads(nth)
        for T in (8, 32):
            st = torch.randn(1, T, STATE_DIM)
            ac = torch.randn(1, T - 1, ACTION_DIM)
            with torch.no_grad():
                for _ in range(3): dep_t(st, ac)
                rep = 100
                t0 = time.perf_counter()
                for _ in range(rep): dep_t(st, ac)
                ms = (time.perf_counter() - t0) / rep * 1e3
            lat[f'window_ms_T{T}_th{nth}'] = round(ms, 4)
            lat[f'per_step_ms_T{T}_th{nth}'] = round(ms / T, 4)
            print(f'deploy-kernel window T={T:3d} th={nth}: {ms:.3f} ms  (per-step ~{ms/T:.4f} ms)', flush=True)
    # ---- 单步热路径(纯一层一步) ----
    st = torch.randn(1, 32, STATE_DIM); ac = torch.randn(1, 31, ACTION_DIM)
    torch.set_num_threads(1)
    with torch.no_grad():
        for _ in range(10): dep_t(st, ac)
        rep = 500
        t0 = time.perf_counter()
        for _ in range(rep): dep_t(st, ac)
        full = (time.perf_counter() - t0) / rep * 1e3
    # 单步 = 整窗/T(含循环开销已摊薄)
    results['max_abs_diff_conv_vs_kernel'] = worst
    results['latency'] = lat
    results['window_ms_T32_th1_reference'] = round(full, 4)
    params = sum(p.numel() for p in dep.parameters()) / 1e6
    results['deploy_params_m'] = round(params, 3)
    results['weight_bytes_mb'] = round(params * 4, 3)

    os.makedirs('revision_experiments/results', exist_ok=True)
    with open('revision_experiments/results/deploy_kernel_verify.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nmax |conv - deploy_kernel| =', worst, flush=True)
    print('Saved -> revision_experiments/results/deploy_kernel_verify.json', flush=True)

    # ---- ONNX 导出(可选; onnx 安装完成后生效) ----
    try:
        import onnx  # noqa
        dep_export = build(0)[1]
        dummy_s = torch.randn(1, 8, STATE_DIM); dummy_a = torch.randn(1, 7, ACTION_DIM)
        path = 'revision_experiments/results/mimo_wm_deploy.onnx'
        torch.onnx.export(dep_export, (dummy_s, dummy_a), path,
                          input_names=['states', 'actions'], output_names=['pred'],
                          opset_version=17)
        print('ONNX export OK ->', path, flush=True)
    except Exception as e:
        print(f'ONNX export skipped/failed: {type(e).__name__}: {e}', flush=True)


if __name__ == '__main__':
    main()
