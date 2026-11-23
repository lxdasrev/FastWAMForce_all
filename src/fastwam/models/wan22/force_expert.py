from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wan_video_dit import RMSNorm, sinusoidal_embedding_1d


def compute_force_flow_loss(
    pred_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    valid: torch.Tensor,
    scheduler_weight: torch.Tensor,
    lambda_force: float,
) -> dict[str, torch.Tensor]:
    """Average valid elements per sample, then average only valid samples."""
    squared = (pred_velocity.float() - target_velocity.float()).pow(2)
    valid_float = valid.to(device=squared.device, dtype=squared.dtype)
    valid_elements = valid_float.sum(dim=(1, 2))
    valid_samples = valid_elements > 0
    sample_mse = (squared * valid_float).sum(dim=(1, 2)) / valid_elements.clamp(min=1.0)
    weight = scheduler_weight.to(device=squared.device, dtype=squared.dtype)
    if weight.ndim == 0:
        weight = weight.expand_as(sample_mse)
    if bool(valid_samples.any()):
        raw = sample_mse[valid_samples].mean()
        weighted = (sample_mse[valid_samples] * weight[valid_samples]).mean()
    else:
        raw = pred_velocity.sum() * 0.0
        weighted = pred_velocity.sum() * 0.0
    return {
        "raw_force_loss": raw,
        "scheduler_weighted_force_loss": weighted,
        "force_loss": weighted * float(lambda_force),
        "valid_ratio": valid_float.mean(),
        "valid_samples": valid_samples.sum(),
    }


def safe_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allowed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scaled dot-product attention that is defined for fully-masked rows.

    Args:
        q/k/v: [B, H, Q|K, Dh]
        allowed: boolean [B, Q, K] or [Q, K], True means visible.
    """
    if allowed is None:
        return F.scaled_dot_product_attention(q, k, v)
    if allowed.ndim == 2:
        allowed = allowed.unsqueeze(0).expand(q.shape[0], -1, -1)
    if allowed.ndim != 3:
        raise ValueError(f"allowed must be [Q,K] or [B,Q,K], got {tuple(allowed.shape)}")
    allowed = allowed.to(device=q.device, dtype=torch.bool)
    row_valid = allowed.any(dim=-1)
    safe = allowed.clone()
    invalid_rows = ~row_valid
    if bool(invalid_rows.any()):
        # SDPA may produce NaN for an all-masked row on some kernels. Temporarily
        # expose one key and zero the corresponding output afterwards.
        safe[..., 0] |= invalid_rows
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=safe.unsqueeze(1))
    return out * row_valid[:, None, :, None].to(dtype=out.dtype)


class ProjectedAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        inner = self.num_heads * self.head_dim
        self.q = nn.Linear(hidden_dim, inner)
        self.k = nn.Linear(hidden_dim, inner)
        self.v = nn.Linear(hidden_dim, inner)
        self.o = nn.Linear(inner, hidden_dim)
        self.norm_q = RMSNorm(inner, eps=eps)
        self.norm_k = RMSNorm(inner, eps=eps)

    def project_qkv(self, x: torch.Tensor):
        return self.norm_q(self.q(x)), self.norm_k(self.k(x)), self.v(x)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        q, k, v = self.project_qkv(x)
        b, q_len, _ = q.shape
        k_len = k.shape[1]
        q = q.view(b, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        out = safe_attention(q, k, v, allowed)
        return self.o(out.transpose(1, 2).reshape(b, q_len, -1))


@dataclass(frozen=True)
class ForceExpertConfig:
    force_dim: int = 2
    hidden_dim: int = 512
    encoder_hidden_dim: int = 128
    ffn_dim: int = 2048
    text_dim: int = 4096
    freq_dim: int = 256
    num_layers: int = 10
    num_heads: int = 8
    head_dim: int = 64
    bridge_num_heads: int = 24
    bridge_head_dim: int = 128
    max_history_frames: int = 16
    future_frames: int = 32
    eps: float = 1.0e-6
    dropout: float = 0.0

    def __post_init__(self):
        if self.force_dim != 2:
            raise ValueError("ForceExpert currently requires exactly thumb/index force_dim=2.")
        if self.num_layers != 10:
            raise ValueError("ForceExpert H-Bridge schedule requires num_layers=10.")
        if self.future_frames != 32:
            raise ValueError("FastWAMForce requires future_frames=32.")


class ForceTokenEncoder(nn.Module):
    def __init__(self, config: ForceExpertConfig):
        super().__init__()
        self.config = config
        self.value = nn.Sequential(
            nn.Linear(1, config.encoder_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.encoder_hidden_dim, config.hidden_dim),
        )
        self.finger_embedding = nn.Parameter(torch.randn(config.force_dim, config.hidden_dim) * 0.02)
        positions = config.max_history_frames + config.future_frames + 1
        self.position_embedding = nn.Parameter(torch.randn(positions, config.hidden_dim) * 0.02)
        self.type_embedding = nn.Parameter(torch.randn(2, config.hidden_dim) * 0.02)
        self.norm = nn.LayerNorm(config.hidden_dim, eps=config.eps)

    def forward(
        self,
        force: torch.Tensor,
        observed_frames: int,
        effective_valid: torch.Tensor,
    ) -> torch.Tensor:
        if force.ndim != 3 or force.shape[-1] != self.config.force_dim:
            raise ValueError(f"force must be [B,T,2], got {tuple(force.shape)}")
        if effective_valid.shape != force.shape:
            raise ValueError("effective_valid must have the same [B,T,2] shape as force")
        total_frames = force.shape[1]
        history_frames = observed_frames - 1
        if history_frames not in (0, self.config.max_history_frames):
            raise ValueError(
                f"observed_frames must be 1 or {self.config.max_history_frames + 1}, got {observed_frames}"
            )
        if total_frames != observed_frames + self.config.future_frames:
            raise ValueError(
                f"force T must equal observed+future={observed_frames + self.config.future_frames}, got {total_frames}"
            )
        pos_start = self.config.max_history_frames - history_frames
        pos_ids = torch.arange(pos_start, pos_start + total_frames, device=force.device)
        type_ids = torch.cat(
            [
                torch.zeros(observed_frames, dtype=torch.long, device=force.device),
                torch.ones(self.config.future_frames, dtype=torch.long, device=force.device),
            ]
        )
        tokens = self.value(force.unsqueeze(-1))
        tokens = (
            tokens
            + self.finger_embedding[None, None]
            + self.position_embedding[pos_ids][None, :, None]
            + self.type_embedding[type_ids][None, :, None]
        )
        tokens = self.norm(tokens)
        return tokens * effective_valid.unsqueeze(-1).to(dtype=tokens.dtype)


class ForceBlock(nn.Module):
    """Temporal -> spatial -> optional bridge -> context cross-attn -> FFN."""

    def __init__(self, config: ForceExpertConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.temporal_norm = nn.LayerNorm(d, eps=config.eps, elementwise_affine=False)
        self.spatial_norm = nn.LayerNorm(d, eps=config.eps, elementwise_affine=False)
        self.bridge_norm = nn.LayerNorm(d, eps=config.eps, elementwise_affine=False)
        self.ffn_norm = nn.LayerNorm(d, eps=config.eps, elementwise_affine=False)
        self.cross_norm = nn.LayerNorm(d, eps=config.eps)
        self.temporal_attn = ProjectedAttention(d, config.num_heads, config.head_dim, config.eps)
        self.spatial_attn = ProjectedAttention(d, config.num_heads, config.head_dim, config.eps)
        self.bridge_attn = ProjectedAttention(
            d, config.bridge_num_heads, config.bridge_head_dim, config.eps
        )
        self.cross_attn = nn.MultiheadAttention(
            d, config.num_heads, dropout=config.dropout, batch_first=True, kdim=d, vdim=d
        )
        self.ffn = nn.Sequential(
            nn.Linear(d, config.ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.ffn_dim, d),
        )
        # temporal, spatial, bridge and FFN each use shift/scale/gate.
        self.modulation = nn.Parameter(torch.randn(1, 1, 12, d) / d**0.5)

    @staticmethod
    def _adaln(x, norm, shift, scale):
        return norm(x) * (1 + scale) + shift

    @staticmethod
    def _zero_invalid(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return x * valid.unsqueeze(-1).to(dtype=x.dtype)

    def local_attention(
        self,
        tokens: torch.Tensor,
        valid: torch.Tensor,
        temporal_mask: torch.Tensor,
        modulation: torch.Tensor,
    ) -> torch.Tensor:
        b, t, f, d = tokens.shape
        parts = (self.modulation.to(modulation) + modulation).chunk(12, dim=3)

        # Temporal: each finger is an independent sequence.
        x = tokens.permute(0, 2, 1, 3).reshape(b * f, t, d)
        v = valid.permute(0, 2, 1).reshape(b * f, t)
        temporal_allowed = temporal_mask[None] & v[:, None, :]
        shift, scale, gate = (p.squeeze(3).permute(0, 2, 1, 3).reshape(b * f, t, d) for p in parts[:3])
        attn_in = self._adaln(x, self.temporal_norm, shift, scale)
        x = x + gate * self.temporal_attn(attn_in, temporal_allowed)
        x = self._zero_invalid(x, v)
        tokens = x.reshape(b, f, t, d).permute(0, 2, 1, 3)

        # Spatial: thumb/index at the same time, no causal restriction.
        x = tokens.reshape(b * t, f, d)
        v = valid.reshape(b * t, f)
        spatial_allowed = v[:, None, :].expand(-1, f, -1)
        shift, scale, gate = (p.squeeze(3).reshape(b * t, f, d) for p in parts[3:6])
        attn_in = self._adaln(x, self.spatial_norm, shift, scale)
        x = x + gate * self.spatial_attn(attn_in, spatial_allowed)
        return self._zero_invalid(x, v).reshape(b, t, f, d)

    def bridge_qkv(self, tokens: torch.Tensor, modulation: torch.Tensor):
        b, t, f, d = tokens.shape
        parts = (self.modulation.to(modulation) + modulation).chunk(12, dim=2)
        shift, scale, gate = (p.squeeze(2).reshape(b, t * f, d) for p in parts[6:9])
        flat = tokens.reshape(b, t * f, d)
        bridge_in = self._adaln(flat, self.bridge_norm, shift, scale)
        q, k, v = self.bridge_attn.project_qkv(bridge_in)
        return q, k, v, flat, gate

    def bridge_residual(
        self,
        flat_tokens: torch.Tensor,
        mixed: torch.Tensor,
        gate: torch.Tensor,
        valid_flat: torch.Tensor,
    ) -> torch.Tensor:
        x = flat_tokens + gate * self.bridge_attn.o(mixed)
        return self._zero_invalid(x, valid_flat)

    def post_attention(
        self,
        flat_tokens: torch.Tensor,
        valid_flat: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        modulation: torch.Tensor,
    ) -> torch.Tensor:
        b, s, d = flat_tokens.shape
        if context_mask.ndim != 2:
            raise ValueError(f"context_mask must be [B,L], got {tuple(context_mask.shape)}")
        cross, _ = self.cross_attn(
            self.cross_norm(flat_tokens),
            context,
            context,
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        x = self._zero_invalid(flat_tokens + cross, valid_flat)
        parts = (self.modulation.to(modulation) + modulation).chunk(12, dim=2)
        shift, scale, gate = (p.squeeze(2).reshape(b, s, d) for p in parts[9:12])
        ffn_in = self._adaln(x, self.ffn_norm, shift, scale)
        x = x + gate * self.ffn(ffn_in)
        return self._zero_invalid(x, valid_flat)


class ForceOutputHead(nn.Module):
    def __init__(self, config: ForceExpertConfig):
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(config.hidden_dim, eps=config.eps, elementwise_affine=False)
        self.modulation = nn.Parameter(torch.randn(1, 1, 2, config.hidden_dim) / config.hidden_dim**0.5)
        self.proj = nn.Linear(config.hidden_dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: torch.Tensor, time_embedding: torch.Tensor, observed_frames: int):
        shift, scale = (self.modulation.to(time_embedding) + time_embedding.unsqueeze(2)).chunk(2, dim=2)
        x = self.proj(self.norm(tokens) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        x = x.reshape(tokens.shape[0], -1, self.config.force_dim)
        return torch.cat([torch.zeros_like(x[:, :observed_frames]), x[:, observed_frames:]], dim=1)


class ForceExpert(nn.Module):
    def __init__(self, config: ForceExpertConfig | dict):
        super().__init__()
        if isinstance(config, dict):
            config = ForceExpertConfig(**config)
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.encoder = ForceTokenEncoder(config)
        self.context_embedding = nn.Sequential(
            nn.Linear(config.text_dim, config.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(config.freq_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_dim, config.hidden_dim * 12)
        )
        self.blocks = nn.ModuleList([ForceBlock(config) for _ in range(config.num_layers)])
        self.head = ForceOutputHead(config)

    @staticmethod
    def build_temporal_mask(total_frames: int, observed_frames: int, device=None):
        mask = torch.zeros(total_frames, total_frames, dtype=torch.bool, device=device)
        mask[:observed_frames, :observed_frames] = True
        mask[observed_frames:, :] = True
        return mask

    def prepare(
        self,
        force: torch.Tensor,
        timestep: torch.Tensor,
        observed_frames: int,
        effective_valid: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | int]:
        b, t, f = force.shape
        if timestep.shape != (b,):
            raise ValueError(f"timestep must be [B], got {tuple(timestep.shape)}")
        tokens = self.encoder(force, observed_frames, effective_valid)
        slice_t = torch.cat(
            [
                torch.zeros(b, observed_frames, device=force.device, dtype=timestep.dtype),
                timestep[:, None].expand(-1, self.config.future_frames),
            ],
            dim=1,
        )
        token_t = slice_t.repeat_interleave(self.config.force_dim, dim=1)
        time = self.time_embedding(
            sinusoidal_embedding_1d(self.config.freq_dim, token_t.reshape(-1)).reshape(b, t * f, -1)
        )
        modulation = self.time_projection(time).reshape(b, t * f, 12, self.config.hidden_dim)
        return {
            "tokens": tokens,
            "valid": effective_valid,
            "valid_flat": effective_valid.reshape(b, t * f),
            "context": self.context_embedding(context),
            "context_mask": context_mask,
            "time": time,
            "modulation": modulation,
            "temporal_mask": self.build_temporal_mask(t, observed_frames, force.device),
            "observed_frames": observed_frames,
        }

    def run_unbridged_block(self, state: dict, layer_idx: int) -> dict:
        block = self.blocks[layer_idx]
        b, t, f, _ = state["tokens"].shape
        modulation = state["modulation"].reshape(b, t, f, 12, self.hidden_dim)
        tokens = block.local_attention(
            state["tokens"], state["valid"], state["temporal_mask"], modulation
        )
        flat = block.post_attention(
            tokens.reshape(b, t * f, self.hidden_dim),
            state["valid_flat"],
            state["context"],
            state["context_mask"],
            state["modulation"],
        )
        state = dict(state)
        state["tokens"] = flat.reshape(b, t, f, self.hidden_dim)
        return state

    def post(self, state: dict) -> torch.Tensor:
        flat = state["tokens"].reshape(state["tokens"].shape[0], -1, self.hidden_dim)
        pred = self.head(flat, state["time"], int(state["observed_frames"]))
        return pred * state["valid"].to(dtype=pred.dtype)

