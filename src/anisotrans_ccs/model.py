"""AnisoTrans v2 model: config scaler, Fourier UTM encoding, sparse transmissibility-biased attention.

Parameter and buffer names match the R6 source implementation (``AnisoTrans-CCS/model_v2.py`` and
``AnisoTrans-CCS/model/config_scaler.py``) so that its ``best.pt`` checkpoints load with strict=True.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

N_CONFIG = 8  # theta_tilde = [k, phi, T, inj, sim_years, kz/kx, L_diff_m, inj_over_k]
_IDX_K, _IDX_T, _IDX_INJ, _IDX_KZKX, _IDX_L = 0, 2, 3, 5, 6


# --------------------------------------------------------------------------- config scaler s(theta)


def physics_init_weights(pins: Mapping[str, float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(W_phys (3x8), b_phys (3,))`` for the pinned physics coefficients."""
    W = torch.zeros(3, N_CONFIG)
    for row in range(3):
        W[row, _IDX_K] = pins["a2"]
        W[row, _IDX_T] = pins["a1"]
        W[row, _IDX_INJ] = pins["a3"]
        W[row, _IDX_L] = pins["a4"]
    W[2, _IDX_KZKX] = pins["c1"]
    b = torch.tensor([pins["b0"], pins["b0"], pins["b0"] + pins["c0"]])
    return W, b


def s_phys(theta: torch.Tensor, pins: Mapping[str, float], exp_clip: Sequence[float]) -> torch.Tensor:
    """Pinned physics target ``exp(clamp(W_phys @ theta + b_phys))`` used by the anchor loss."""
    W, b = physics_init_weights(pins)
    g = theta @ W.to(theta).T + b.to(theta)
    return torch.exp(torch.clamp(g, float(exp_clip[0]), float(exp_clip[1])))


class ConfigScaler(nn.Module):
    """8 -> 32 -> 3 ReLU MLP followed by a clamped exponential (419 parameters).

    At initialisation the module realises ``s = s_phys`` exactly (bias-shifted identity hidden layer).
    """

    SHIFT = 10.0  # keeps the hidden layer in the ReLU-active regime for normalised inputs

    def __init__(self, pins: Mapping[str, float], exp_clip: Sequence[float]):
        super().__init__()
        self.fc1 = nn.Linear(N_CONFIG, 32)
        self.fc2 = nn.Linear(32, 3)
        self.anchor = nn.Parameter(torch.zeros(32))
        self.exp_clip = (float(exp_clip[0]), float(exp_clip[1]))
        self._physics_init(pins)

    @torch.no_grad()
    def _physics_init(self, pins: Mapping[str, float]) -> None:
        W, b = physics_init_weights(pins)
        m = self.SHIFT
        self.fc1.weight.zero_()
        for i in range(N_CONFIG):
            self.fc1.weight[i, i] = 1.0
        self.fc1.bias.fill_(m)
        self.fc2.weight.zero_()
        self.fc2.weight[:, :N_CONFIG].copy_(W)
        self.fc2.bias.copy_(b - W @ torch.full((N_CONFIG,), m))
        self.anchor.zero_()

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(theta) + self.anchor)
        g = torch.clamp(self.fc2(h), self.exp_clip[0], self.exp_clip[1])
        return torch.exp(g)


# --------------------------------------------------------------------------- positional encoding


class _FourierPE(nn.Module):
    """Fourier-feature encoding of normalised UTM coordinates (Tancik et al., 2020)."""

    def __init__(self, dim: int, n_fourier: int = 16):
        super().__init__()
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(200.0), n_fourier))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(3 * 2 * n_fourier, dim)

    def forward(self, utm: torch.Tensor) -> torch.Tensor:
        x = utm.unsqueeze(-1) * self.freqs  # (N, 3, B)
        feat = torch.cat([x.sin(), x.cos()], dim=-1)  # (N, 3, 2B)
        return self.proj(feat.flatten(1))  # (N, dim)


# --------------------------------------------------------------------------- sparse attention


class _SparsePhysAttn(nn.Module):
    """Multi-head attention restricted to each cell's k nearest physical neighbours."""

    def __init__(
        self,
        cand_idx: torch.Tensor,
        cand_valid: torch.Tensor,
        dim: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} is not divisible by n_heads {n_heads}")
        self.h = n_heads
        self.dh = dim // n_heads
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.register_buffer("cand_idx", cand_idx.long().clamp(min=0).contiguous())
        self.register_buffer("cand_valid", cand_valid.bool().contiguous())

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        q = self.wq(x).view(n, self.h, self.dh)
        k = self.wk(x).view(n, self.h, self.dh)
        v = self.wv(x).view(n, self.h, self.dh)
        ci = self.cand_idx
        kg = k[ci]  # (N, k, h, dh)
        vg = v[ci]
        scores = torch.einsum("nhd,nkhd->nhk", q, kg) / (self.dh**0.5)
        scores = scores + bias.unsqueeze(1)  # (N, 1, k)
        scores = scores.masked_fill(~self.cand_valid.unsqueeze(1), float("-inf"))
        w = self.attn_drop(scores.softmax(dim=-1))
        out = torch.einsum("nhk,nkhd->nhd", w, vg)
        return self.wo(out.reshape(n, self.h * self.dh))


class _SparseBlock(nn.Module):
    """Post-LayerNorm sparse attention and feed-forward residual block."""

    def __init__(self, cand_idx, cand_valid, dim: int, n_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.attn = _SparsePhysAttn(cand_idx, cand_valid, dim, n_heads, dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        hidden = dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.attn(x, bias))
        return self.ln2(x + self.ffn(x))


# --------------------------------------------------------------------------- AnisoTrans v2


class AnisoTransV2(nn.Module):
    """Transmissibility-weighted sparse-neighbourhood transformer (AnisoTrans v2).

    ``theta`` is the (8,) normalised configuration vector and ``K`` the (N, 3) normalised
    permeability field. The output is the normalised pressure field of shape (N,).
    """

    def __init__(
        self,
        cfg: Mapping,
        utm_norm: torch.Tensor,
        cand_idx: torch.Tensor,
        cand_valid: torch.Tensor,
    ):
        super().__init__()
        self.dim = int(cfg["dim"])
        self.n_heads = int(cfg["n_heads"])
        self.n_layers = int(cfg["n_layers"])
        self.ffn_mult = int(cfg["ffn_mult"])
        dropout = float(cfg["dropout"])
        self.alpha_max = float(cfg["alpha_max"])
        alpha_init = float(cfg["alpha_init"])

        scaler_cfg = cfg["scaler"]
        self.scaler = ConfigScaler(scaler_cfg["physics_init"], scaler_cfg["exp_clip"])

        self.token_embed = nn.Linear(14, self.dim)  # [K(3); K_s(3); theta(8)]
        self.utm_pe = _FourierPE(self.dim, n_fourier=int(cfg["n_fourier"]))
        self.input_norm = nn.LayerNorm(self.dim)
        self.blocks = nn.ModuleList(
            [
                _SparseBlock(cand_idx, cand_valid, self.dim, self.n_heads, self.ffn_mult, dropout)
                for _ in range(self.n_layers)
            ]
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.register_buffer("utm_norm", utm_norm.float().contiguous())

        p0 = min(max(alpha_init / self.alpha_max, 1e-4), 1.0 - 1e-4)
        self.alpha_raw = nn.Parameter(torch.tensor(math.log(p0 / (1.0 - p0))))

    def alpha(self) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.alpha_raw)

    def forward(self, theta: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        n = K.shape[0]
        th = theta.reshape(1, N_CONFIG)
        with torch.autocast(device_type=K.device.type, enabled=False):
            s = self.scaler(th.float()).squeeze(0)  # (3,), fp32 for a finite backward pass
        K_s = K * s.to(K.dtype)  # transmissibility scaling, (N, 3)
        T = K_s.sum(dim=1) / 3.0  # (N,) scalar transmissibility per cell
        tokens = torch.cat([K, K_s, th.expand(n, N_CONFIG)], dim=1)  # (N, 14)
        x = self.input_norm(self.token_embed(tokens) + self.utm_pe(self.utm_norm))

        # Pairwise physics bias on the (N, k) neighbourhood: high-transmissibility neighbours
        # receive more attention.
        Tg = T[self.blocks[0].attn.cand_idx]  # (N, k)
        bias = self.alpha() * 0.5 * (T.unsqueeze(1) + Tg)
        for blk in self.blocks:
            if self.training:
                x = checkpoint(blk, x, bias, use_reentrant=False)
            else:
                x = blk(x, bias)
        return self.readout(x).squeeze(1)


def build_model(
    cfg: Mapping,
    utm_norm: torch.Tensor,
    cand_idx: torch.Tensor,
    cand_valid: torch.Tensor,
) -> AnisoTransV2:
    """Construct the model from the ``model`` section of the configuration."""
    return AnisoTransV2(cfg, utm_norm=utm_norm, cand_idx=cand_idx, cand_valid=cand_valid)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
