"""
Dynamics-Adaptive Diagonal SSM (DA-SSM)
=======================================
Novel contribution: Input-dependent diagonal A matrix via lightweight adaptation.

Key insight: In robot dynamics, the system properties (e.g., friction, inertia)
change with the robot's configuration (joint angles, velocities). A fixed diagonal
A matrix (as in S4D) cannot capture this configuration-dependence.

DA-SSM solves this by:
1. Learning a base diagonal A matrix (like S4D)
2. Using a lightweight adaptation network to produce input-dependent perturbations
3. The adaptation is bounded and smooth, ensuring numerical stability

This is fundamentally different from:
- S4D: Fixed A, no input adaptation
- Mamba: Full selective scanning (heavyweight, O(D*N) per-step)
- DA-SSM: Diagonal adaptation (lightweight, O(D) per-step)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DynamicsAdaptiveDiagSSM(nn.Module):
    """
    Dynamics-Adaptive Diagonal SSM.

    The core innovation: instead of a fixed diagonal A matrix, we compute
    A(x) = A_base + alpha * adapt_net(x), where:
    - A_base: learned base diagonal (like S4D)
    - adapt_net: lightweight MLP that produces bounded perturbations
    - alpha: learnable scale factor (initialized to 0 for stable training)

    This allows the SSM to adapt its dynamics based on the input,
    capturing configuration-dependent robot dynamics.
    """

    def __init__(self, d_model: int, d_state: int = 64, adapt_hidden: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Base diagonal A (like S4D)
        self.log_A_real = nn.Parameter(torch.randn(d_model, d_state) * 0.01 - 1.0)
        self.A_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.1)

        # B, C
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)

        # D (skip)
        self.D = nn.Parameter(torch.ones(d_model))

        # dt
        self.log_dt = nn.Parameter(torch.randn(d_model) * 0.01)

        # === NOVEL: Dynamics Adaptation Network ===
        # Lightweight MLP that produces perturbations to A_real
        # Input: d_model features -> adapt_hidden -> d_state outputs
        # This is much lighter than Mamba's selective scanning (O(D*H) vs O(D*N*2))
        self.adapt_net = nn.Sequential(
            nn.Linear(d_model, adapt_hidden),
            nn.GELU(),
            nn.Linear(adapt_hidden, d_state),
            nn.Tanh(),  # Bound output to [-1, 1]
        )

        # Learnable adaptation scale (initialized to 0 = no adaptation initially)
        # This ensures the model starts as S4D and learns to adapt gradually
        self.alpha = nn.Parameter(torch.zeros(d_model, 1))

    def forward(self, x, mode='conv'):
        """
        x: (B, L, D) -> (B, L, D)
        """
        if mode == 'recurrent':
            return self._forward_recurrent(x)
        else:
            return self._forward_conv(x)

    def _get_adaptive_A(self, x):
        """
        Compute input-dependent diagonal A.
        x: (B, L, D) -> A: (B, L, D, N) complex
        """
        batch, L, D = x.shape

        # Base A (same as S4D)
        A_base_real = -torch.exp(self.log_A_real)  # (D, N)

        # Adaptive perturbation
        delta = self.adapt_net(x)  # (B, L, N)
        # Apply learnable scale (per-dimension)
        delta = delta * self.alpha  # (B, L, D, N) via broadcasting

        # Adaptive A
        A_real = A_base_real.unsqueeze(0).unsqueeze(0) + delta  # (B, L, D, N)
        A_imag = self.A_imag.unsqueeze(0).unsqueeze(0)  # (D, N) -> (1, 1, D, N)

        return -torch.exp(A_real) + 1j * A_imag

    def _forward_conv(self, x):
        """FFT convolution mode with adaptive A."""
        batch, L, D = x.shape
        N = self.d_state

        dt = torch.exp(self.log_dt)  # (D,)

        # For FFT convolution, we need a single kernel per dimension
        # Use the base A for the kernel (faster) and apply adaptation as a correction
        A_base = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (D, N)

        # Compute base convolution kernel (same as S4D)
        dtA = dt.unsqueeze(-1) * A_base  # (D, N)
        powers = torch.arange(L, device=x.device, dtype=x.dtype)
        dtA_pow = dtA.unsqueeze(-1) ** powers.unsqueeze(0).unsqueeze(0)  # (D, N, L)
        CB = self.C * self.B * dt.unsqueeze(-1)  # (D, N)
        K = (CB.unsqueeze(-1) * dtA_pow).sum(dim=1)  # (D, L)
        K = K.real

        # FFT convolution
        K_fft = torch.fft.rfft(K, n=2*L)
        x_fft = torch.fft.rfft(x.permute(0, 2, 1), n=2*L)
        y_base = torch.fft.irfft(K_fft.unsqueeze(0) * x_fft, n=2*L)[:, :, :L]
        y_base = y_base.permute(0, 2, 1)  # (B, L, D)

        # === NOVEL: Adaptive correction ===
        # Compute input-dependent correction term
        delta = self.adapt_net(x)  # (B, L, N)
        correction = (delta.unsqueeze(2) * self.C.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (B, L, D)
        correction = correction * self.alpha.squeeze(-1).unsqueeze(0).unsqueeze(0)  # (B, L, D)

        # Combine base output with adaptive correction
        y = y_base + correction

        # Skip connection
        y = y + x * self.D

        return y

    def _forward_recurrent(self, x):
        """Recurrent mode with adaptive A."""
        batch, L, D = x.shape
        N = self.d_state

        dt = torch.exp(self.log_dt)
        A_base = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (D, N)

        # Adaptive A for each timestep
        delta = self.adapt_net(x)  # (B, L, N)
        A_adapt = self.alpha.unsqueeze(0).unsqueeze(0) * delta  # (B, L, D, N)

        # Initialize hidden state
        h = torch.zeros(batch, D, N, device=x.device, dtype=torch.cfloat)

        outputs = []
        for t in range(L):
            # Adaptive A at this timestep
            A_t = A_base + A_adapt[:, t]  # (B, D, N)
            dtA = dt.unsqueeze(-1) * A_t  # (B, D, N)
            A_bar = torch.exp(dtA)
            B_bar = (A_bar - 1) / A_t * self.B.unsqueeze(0)

            # State update
            h = A_bar * h + B_bar * x[:, t, :].unsqueeze(-1)
            y_t = (self.C.unsqueeze(0) * h).sum(dim=-1).real + self.D * x[:, t, :]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class DASSMBlock(nn.Module):
    """DA-SSM Block: LayerNorm + DA-SSM + Gating + Residual"""

    def __init__(self, d_model: int, d_state: int = 64, adapt_hidden: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = DynamicsAdaptiveDiagSSM(d_model, d_state, adapt_hidden)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x, mode='conv'):
        residual = x
        x_norm = self.norm(x)
        ssm_out = self.ssm(x_norm, mode=mode)
        g = torch.sigmoid(self.gate(x_norm))
        out = g * ssm_out + (1 - g) * x_norm
        return residual + out


class DASSMWorldModel(nn.Module):
    """
    DA-SSM World Model

    Novel architecture combining:
    1. Dynamics-Adaptive Diagonal SSM (DA-SSM) - input-dependent state transitions
    2. Mamba-style gated block structure
    3. Residual state prediction

    Key innovation: The diagonal A matrix adapts to the input, capturing
    configuration-dependent robot dynamics without the overhead of
    Mamba's selective scanning mechanism.
    """

    def __init__(
        self,
        state_dim: int = 28,
        action_dim: int = 7,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 4,
        adapt_hidden: int = 16,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # DA-SSM backbone
        self.backbone = nn.ModuleList([
            DASSMBlock(d_model, d_state, adapt_hidden)
            for _ in range(n_layers)
        ])

        # Norm
        self.norm = nn.LayerNorm(d_model)

        # Decoder
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
            # Initialize alpha to 0 for stable training start
            if hasattr(m, 'alpha'):
                nn.init.zeros_(m.alpha)

    def forward(self, states, actions, mode='conv'):
        """
        states: (B, T, state_dim)
        actions: (B, T-1, action_dim)
        Returns: (B, state_dim)
        """
        if actions.shape[1] < states.shape[1]:
            pad_len = states.shape[1] - actions.shape[1]
            pad = torch.zeros(states.shape[0], pad_len, actions.shape[-1],
                            device=actions.device, dtype=actions.dtype)
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

    def predict_trajectory(self, init_states, init_actions, future_actions):
        states_seq = init_states.clone()
        actions_seq = init_actions.clone()
        predictions = []

        for h in range(future_actions.shape[1]):
            pred = self.forward(states_seq, actions_seq)
            predictions.append(pred)
            states_seq = torch.cat([states_seq[:, 1:], pred.unsqueeze(1)], dim=1)
            actions_seq = torch.cat([actions_seq[:, 1:], future_actions[:, h:h+1]], dim=1)

        return torch.stack(predictions, dim=1)
