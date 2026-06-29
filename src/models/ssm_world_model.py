import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DiagSSM(nn.Module):
    """
    S4D风格的对角SSM: 用对角A矩阵 + FFT卷积
    极快且数值稳定
    支持两种模式: 'conv' (FFT卷积, 适合批量训练) 和 'recurrent' (递推, 适合单步推理)
    """

    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # 对角A: log空间, 负值确保稳定
        self.log_A_real = nn.Parameter(torch.randn(d_model, d_state) * 0.01 - 1.0)
        self.A_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.1)

        # B, C
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)

        # D (skip)
        self.D = nn.Parameter(torch.ones(d_model))

        # dt
        self.log_dt = nn.Parameter(torch.randn(d_model) * 0.01)

    def forward(self, x, mode='conv'):
        """
        x: (B, L, D) -> (B, L, D)
        mode: 'conv' for FFT convolution, 'recurrent' for recursive inference
        """
        if mode == 'recurrent':
            return self._forward_recurrent(x)
        else:
            return self._forward_conv(x)

    def _forward_conv(self, x):
        """FFT卷积模式: O(T log T) 训练复杂度"""
        batch, L, D = x.shape
        N = self.d_state

        dt = torch.exp(self.log_dt)  # (D,)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (D, N) complex

        # 计算卷积核 K[t] = sum_n C[d,n] * (exp(A[d,n]*dt[d])^t) * B[d,n] * dt[d]
        # 先计算 dt*A 的幂
        dtA = dt.unsqueeze(-1) * A  # (D, N)

        # K[t] = C * (dtA)^t * B * dt, 对t=0..L-1
        powers = torch.arange(L, device=x.device, dtype=x.dtype)  # (L,)
        # (dtA)^t = exp(t * log(dtA))
        # 但dtA是复数, 直接用幂
        dtA_pow = dtA.unsqueeze(-1) ** powers.unsqueeze(0).unsqueeze(0)  # (D, N, L)

        # K[d, t] = sum_n C[d,n] * B[d,n] * dt[d] * dtA_pow[d,n,t]
        CB = self.C * self.B * dt.unsqueeze(-1)  # (D, N)
        K = (CB.unsqueeze(-1) * dtA_pow).sum(dim=1)  # (D, L)
        K = K.real  # 取实部

        # 因果卷积 (FFT)
        K_fft = torch.fft.rfft(K, n=2*L)  # (D, L+1)
        x_fft = torch.fft.rfft(x.permute(0, 2, 1), n=2*L)  # (B, D, L+1)
        y = torch.fft.irfft(K_fft.unsqueeze(0) * x_fft, n=2*L)[:, :, :L]  # (B, D, L)
        y = y.permute(0, 2, 1)  # (B, L, D)

        # skip connection
        y = y + x * self.D

        return y

    def _forward_recurrent(self, x):
        """递推模式: O(T*N) 时间复杂度, O(1) 单步延迟"""
        batch, L, D = x.shape
        N = self.d_state

        dt = torch.exp(self.log_dt)  # (D,)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (D, N) complex

        # 离散化系数
        dtA = dt.unsqueeze(-1) * A  # (D, N)
        A_bar = torch.exp(dtA)  # (D, N) complex
        B_bar = (A_bar - 1) / A * self.B  # (D, N) complex, 近似

        # 初始化隐状态
        h = torch.zeros(batch, D, N, device=x.device, dtype=torch.cfloat)  # (B, D, N)

        outputs = []
        for t in range(L):
            # 状态更新: h_t = A_bar * h_{t-1} + B_bar * x_t
            h = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * x[:, t, :].unsqueeze(-1)
            # 输出: y_t = C * h_t + D * x_t
            y_t = (self.C.unsqueeze(0) * h).sum(dim=-1).real + self.D * x[:, t, :]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, L, D)


class SSMBlock(nn.Module):
    """SSM块: LayerNorm + SSM + 残差（S4D原版，无门控）"""

    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = DiagSSM(d_model, d_state)

    def forward(self, x, mode='conv'):
        residual = x
        x_norm = self.norm(x)
        ssm_out = self.ssm(x_norm, mode=mode)
        return residual + ssm_out


class SSMWorldModel(nn.Module):
    """
    SSM世界模型 (SSM-WM)

    用于人形机器人状态预测:
    输入: 历史状态序列 s_0:T 和动作序列 a_0:T-1
    输出: 预测的下一状态 s_{t+1}
    """

    def __init__(
        self,
        state_dim: int = 28,
        action_dim: int = 7,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 4,
        d_conv: int = 4,    # 未使用, 保持接口兼容
        expand: int = 2,     # 未使用, 保持接口兼容
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # 状态-动作编码器
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # SSM主干
        self.backbone = nn.ModuleList([
            SSMBlock(d_model, d_state)
            for _ in range(n_layers)
        ])

        # 归一化
        self.norm = nn.LayerNorm(d_model)

        # 状态解码器
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, mode='conv'):
        """
        单步预测:
        states:  (B, T, state_dim)
        actions: (B, T-1, action_dim)
        mode: 'conv' for FFT convolution, 'recurrent' for recursive inference
        返回: (B, state_dim)
        """
        # 对齐长度
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(
                states.shape[0], pad_len, actions.shape[-1],
                device=actions.device, dtype=actions.dtype
            )
            actions = torch.cat([pad, actions], dim=1)

        x = torch.cat([states, actions], dim=-1)
        x = self.encoder(x)

        for block in self.backbone:
            x = block(x, mode=mode)

        x = self.norm(x)
        x = x[:, -1, :]
        delta_s = self.decoder(x)
        pred_state = states[:, -1, :] + delta_s

        return pred_state

    def predict_trajectory(
        self,
        init_states: torch.Tensor,
        init_actions: torch.Tensor,
        future_actions: torch.Tensor,
    ):
        states_seq = init_states.clone()
        actions_seq = init_actions.clone()
        predictions = []

        for h in range(future_actions.shape[1]):
            pred = self.forward(states_seq, actions_seq)
            predictions.append(pred)
            states_seq = torch.cat([states_seq[:, 1:], pred.unsqueeze(1)], dim=1)
            actions_seq = torch.cat([actions_seq[:, 1:], future_actions[:, h:h+1]], dim=1)

        return torch.stack(predictions, dim=1)
