"""Mamba世界模型 — 支持mamba_ssm和纯PyTorch两种实现"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 尝试导入mamba_ssm，如果失败则使用纯PyTorch实现
try:
    import types, sys
    fake_gen = types.ModuleType('transformers.generation')
    fake_gen.GreedySearchDecoderOnlyOutput = type('x', (), {})
    fake_gen.SampleDecoderOnlyOutput = type('x', (), {})
    fake_gen.TextStreamer = type('x', (), {})
    sys.modules['transformers.generation'] = fake_gen
    from mamba_ssm import Mamba
    HAS_MAMBA_SSM = True
    print("[INFO] Using mamba_ssm CUDA implementation")
except Exception as e:
    HAS_MAMBA_SSM = False
    print(f"[INFO] mamba_ssm not available ({e}), using pure PyTorch fallback")


class SelectiveSSM(nn.Module):
    """纯PyTorch实现的选择性SSM（mamba_ssm不可用时的备用方案）"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.log_A = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        d_inner = self.d_model * 2

        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        params = self.x_proj(x_conv)
        B_param, C_param, dt = params.split([self.d_state, self.d_state, 1], dim=-1)
        dt = F.softplus(dt)

        A = -torch.exp(self.log_A)
        dA = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        dA = torch.exp(dA)
        dB = dt.unsqueeze(-1) * B_param.unsqueeze(2)

        h = torch.zeros(B, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x_conv[:, t, :].unsqueeze(-1)
            y_t = (h * C_param[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    """Mamba块: LayerNorm + Mamba/SelectiveSSM + 残差（原版，无门控）"""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if HAS_MAMBA_SSM:
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        else:
            self.mamba = SelectiveSSM(d_model, d_state)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        mamba_out = self.mamba(x_norm)
        return residual + mamba_out


class MambaWorldModel(nn.Module):
    """Mamba世界模型 — 架构与SSM-WM完全对齐"""
    def __init__(self, state_dim=28, action_dim=7, d_model=128, d_state=16, n_layers=4):
        super().__init__()
        self.state_dim = state_dim
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.backbone = nn.ModuleList([
            MambaBlock(d_model, d_state) for _ in range(n_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, states, actions):
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1],
                              device=actions.device, dtype=actions.dtype)
            actions = torch.cat([pad, actions], dim=1)
        x = torch.cat([states, actions], dim=-1)
        x = self.encoder(x)
        for block in self.backbone:
            x = block(x)
        delta_s = self.decoder(x[:, -1, :])
        return states[:, -1, :] + delta_s
