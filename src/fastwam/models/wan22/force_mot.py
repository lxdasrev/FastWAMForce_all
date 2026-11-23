from __future__ import annotations

import torch
import torch.nn as nn

from .force_expert import ForceExpert, safe_attention
from .mot import MoT
from .wan_video_dit import flash_attention


class ForceBridgeMoT(MoT):
    """Thirty-layer FastWAM backbone with a ten-layer sparse Force H-Bridge."""

    DEFAULT_FORCE_TO_BACKBONE = {1: 8, 2: 10, 3: 13, 4: 15, 5: 17, 6: 20, 7: 22}

    def __init__(
        self,
        mixtures: dict[str, nn.Module],
        force_expert: ForceExpert,
        force_to_backbone: dict[int, int] | None = None,
        mot_checkpoint_mixed_attn: bool = False,
    ):
        super().__init__(mixtures, mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn)
        self.force_expert = force_expert
        mapping = dict(force_to_backbone or self.DEFAULT_FORCE_TO_BACKBONE)
        if sorted(mapping) != list(range(1, 8)):
            raise ValueError("force_to_backbone must map Force blocks 1..7 exactly once")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("backbone bridge layers must be unique")
        if min(mapping.values()) < 0 or max(mapping.values()) >= self.num_layers:
            raise ValueError("backbone bridge layer is outside the Video/Action depth")
        self.force_to_backbone = mapping
        self.backbone_to_force = {backbone: force for force, backbone in mapping.items()}
        self.bridge_gates = nn.ParameterDict(
            {str(layer): nn.Parameter(torch.zeros(())) for layer in sorted(mapping)}
        )
        self._bridge_residual_rms: dict[str, torch.Tensor] = {}
        self.action_reads_future_force = True
        cfg = force_expert.config
        if cfg.bridge_num_heads != self.num_heads or cfg.bridge_head_dim != self.attn_head_dim:
            raise ValueError(
                "Force bridge heads must match Wan MoT heads: "
                f"force={cfg.bridge_num_heads}x{cfg.bridge_head_dim}, "
                f"wan={self.num_heads}x{self.attn_head_dim}"
            )

    @staticmethod
    def _common_attention(q, k, v, num_heads: int, allowed: torch.Tensor):
        b, q_len, inner = q.shape
        head_dim = inner // num_heads
        k_len = k.shape[1]
        qh = q.view(b, q_len, num_heads, head_dim).transpose(1, 2)
        kh = k.view(b, k_len, num_heads, head_dim).transpose(1, 2)
        vh = v.view(b, k_len, num_heads, head_dim).transpose(1, 2)
        out = safe_attention(qh, kh, vh, allowed)
        return out.transpose(1, 2).reshape(b, q_len, inner)

    def _force_masks(
        self,
        force_valid: torch.Tensor,
        observed_frames: int,
        video_seq_len: int,
        first_frame_tokens: int,
        action_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return bridge masks for [force queries] and [action queries].

        Key order is [video, force, action]. Force-to-action remains disabled.
        """
        b, force_seq_len = force_valid.shape
        total_keys = video_seq_len + force_seq_len + action_seq_len
        force_allowed = torch.zeros(
            b, force_seq_len, total_keys, dtype=torch.bool, device=force_valid.device
        )
        action_allowed = torch.zeros(
            b, action_seq_len, total_keys, dtype=torch.bool, device=force_valid.device
        )
        v0 = min(first_frame_tokens, video_seq_len)
        force_allowed[:, :, :v0] = True
        action_allowed[:, :, :v0] = True

        fingers = force_seq_len // (observed_frames + 32)
        if fingers != 2:
            raise ValueError("Force bridge expects two time-major finger tokens per frame")
        observed_tokens = observed_frames * fingers
        key_valid = force_valid[:, None, :]
        # F0 -> V0,F0; F+ -> V0,F0,F+.
        force_allowed[:, :observed_tokens, video_seq_len : video_seq_len + observed_tokens] = True
        force_allowed[:, observed_tokens:, video_seq_len : video_seq_len + force_seq_len] = True
        force_allowed[:, :, video_seq_len : video_seq_len + force_seq_len] &= key_valid
        # Invalid force queries are hard-zeroed and receive no attention rows.
        force_allowed &= force_valid[:, :, None]

        # A -> V0,F0,F+,A. It never sees V+.
        action_force_tokens = force_seq_len if self.action_reads_future_force else observed_tokens
        action_allowed[:, :, video_seq_len : video_seq_len + action_force_tokens] = key_valid[:, :, :action_force_tokens]
        action_allowed[:, :, video_seq_len + force_seq_len :] = True
        return force_allowed, action_allowed

    def _run_force_local(self, state: dict, force_layer: int):
        block = self.force_expert.blocks[force_layer]
        b, t, f, _ = state["tokens"].shape
        modulation = state["modulation"].reshape(
            b, t, f, 12, self.force_expert.hidden_dim
        )
        state = dict(state)
        state["tokens"] = block.local_attention(
            state["tokens"], state["valid"], state["temporal_mask"], modulation
        )
        return state

    def _finish_force_block(self, state: dict, force_layer: int, flat: torch.Tensor):
        block = self.force_expert.blocks[force_layer]
        flat = block.post_attention(
            flat,
            state["valid_flat"],
            state["context"],
            state["context_mask"],
            state["modulation"],
        )
        state = dict(state)
        state["tokens"] = flat.reshape_as(state["tokens"])
        return state

    def _forward_bridge_layer(
        self,
        *,
        layer_idx: int,
        force_layer: int,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        force_state: dict,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        base_attention_mask: torch.Tensor,
        first_frame_tokens: int,
    ):
        video_block = self.mixtures["video"].blocks[layer_idx]
        action_block = self.mixtures["action"].blocks[layer_idx]
        force_block = self.force_expert.blocks[force_layer]
        force_state = self._run_force_local(force_state, force_layer)

        v_io = self._build_expert_attention_io(
            self.mixtures["video"], video_block, video_tokens, video_freqs, video_t_mod
        )
        a_io = self._build_expert_attention_io(
            self.mixtures["action"], action_block, action_tokens, action_freqs, action_t_mod
        )
        qv, kv, vv, rv, gmv, smv, scv, gfv, _ = v_io
        qa, ka, va, ra, gma, sma, sca, gfa, _ = a_io
        qf, kf, vf, rf, gmf = force_block.bridge_qkv(
            force_state["tokens"], force_state["modulation"]
        )

        base = flash_attention(
            q=torch.cat([qv, qa], dim=1),
            k=torch.cat([kv, ka], dim=1),
            v=torch.cat([vv, va], dim=1),
            num_heads=self.num_heads,
            ctx_mask=base_attention_mask.to(qv.device),
        )
        sv, sf, sa = video_tokens.shape[1], rf.shape[1], action_tokens.shape[1]
        force_mask, action_mask = self._force_masks(
            force_state["valid_flat"],
            int(force_state["observed_frames"]),
            sv,
            first_frame_tokens,
            sa,
        )
        tri_allowed = torch.cat([force_mask, action_mask], dim=1)
        tri = self._common_attention(
            torch.cat([qf, qa], dim=1),
            torch.cat([kv, kf, ka], dim=1),
            torch.cat([vv, vf, va], dim=1),
            self.num_heads,
            tri_allowed,
        )
        tri_force, tri_action = tri[:, :sf], tri[:, sf:]
        base_video, base_action = base[:, :sv], base[:, sv:]
        gate = torch.tanh(self.bridge_gates[str(force_layer)]).to(base_action.dtype)
        has_force = force_state["valid_flat"].any(dim=1)[:, None, None]
        action_mixed = base_action + gate * (tri_action - base_action) * has_force
        self._bridge_residual_rms[str(force_layer)] = (
            (tri_action.detach().float() - base_action.detach().float()).pow(2).mean().sqrt()
        )

        video_tokens = self._apply_expert_post_block_tensor(
            video_block, rv, base_video, gmv, smv, scv, gfv, video_context, video_context_mask
        )
        action_tokens = self._apply_expert_post_block_tensor(
            action_block, ra, action_mixed, gma, sma, sca, gfa, action_context, action_context_mask
        )
        force_flat = force_block.bridge_residual(
            rf, tri_force, gmf, force_state["valid_flat"]
        )
        force_state = self._finish_force_block(force_state, force_layer, force_flat)
        return video_tokens, action_tokens, force_state

    def forward_joint_force_core(
        self,
        *,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        force_state: dict,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        first_frame_tokens: int,
    ):
        force_state = self.force_expert.run_unbridged_block(force_state, 0)
        xv, xa = video_tokens, action_tokens
        tail_done = False
        for layer_idx in range(self.num_layers):
            force_layer = self.backbone_to_force.get(layer_idx)
            if force_layer is None:
                xv, xa = self._forward_joint_layer(
                    self.mixtures["video"].blocks[layer_idx],
                    self.mixtures["action"].blocks[layer_idx],
                    xv,
                    xa,
                    video_freqs,
                    action_freqs,
                    video_t_mod,
                    action_t_mod,
                    video_context,
                    video_context_mask,
                    action_context,
                    action_context_mask,
                    attention_mask,
                )
            else:
                xv, xa, force_state = self._forward_bridge_layer(
                    layer_idx=layer_idx,
                    force_layer=force_layer,
                    video_tokens=xv,
                    action_tokens=xa,
                    force_state=force_state,
                    video_freqs=video_freqs,
                    action_freqs=action_freqs,
                    video_t_mod=video_t_mod,
                    action_t_mod=action_t_mod,
                    video_context=video_context,
                    video_context_mask=video_context_mask,
                    action_context=action_context,
                    action_context_mask=action_context_mask,
                    base_attention_mask=attention_mask,
                    first_frame_tokens=first_frame_tokens,
                )
                if force_layer == 7:
                    force_state = self.force_expert.run_unbridged_block(force_state, 8)
                    force_state = self.force_expert.run_unbridged_block(force_state, 9)
                    tail_done = True
        if not tail_done:
            raise RuntimeError("Force tail blocks were not scheduled; bridge mapping is invalid")
        return xv, xa, force_state

    def _cached_action_layer(
        self,
        layer_idx,
        action_tokens,
        action_freqs,
        action_t_mod,
        action_context,
        action_context_mask,
        video_k,
        video_v,
        action_attention_mask,
    ):
        block = self.mixtures["action"].blocks[layer_idx]
        io = self._build_expert_attention_io(
            self.mixtures["action"], block, action_tokens, action_freqs, action_t_mod
        )
        qa, ka, va, ra, gma, sma, sca, gfa, _ = io
        mixed = flash_attention(
            q=qa,
            k=torch.cat([video_k, ka], dim=1),
            v=torch.cat([video_v, va], dim=1),
            num_heads=self.num_heads,
            ctx_mask=action_attention_mask.to(qa.device),
        )
        return self._apply_expert_post_block_tensor(
            block, ra, mixed, gma, sma, sca, gfa, action_context, action_context_mask
        )

    def forward_action_force_with_video_cache(
        self,
        *,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        force_state: dict,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
    ):
        force_state = self.force_expert.run_unbridged_block(force_state, 0)
        xa = action_tokens
        video_seq_len = video_cache_k[0].shape[1]
        tail_done = False
        for layer_idx in range(self.num_layers):
            force_layer = self.backbone_to_force.get(layer_idx)
            if force_layer is None:
                xa = self._cached_action_layer(
                    layer_idx,
                    xa,
                    action_freqs,
                    action_t_mod,
                    action_context,
                    action_context_mask,
                    video_cache_k[layer_idx],
                    video_cache_v[layer_idx],
                    action_attention_mask,
                )
                continue

            force_state = self._run_force_local(force_state, force_layer)
            force_block = self.force_expert.blocks[force_layer]
            action_block = self.mixtures["action"].blocks[layer_idx]
            a_io = self._build_expert_attention_io(
                self.mixtures["action"], action_block, xa, action_freqs, action_t_mod
            )
            qa, ka, va, ra, gma, sma, sca, gfa, _ = a_io
            qf, kf, vf, rf, gmf = force_block.bridge_qkv(
                force_state["tokens"], force_state["modulation"]
            )
            base_action = flash_attention(
                q=qa,
                k=torch.cat([video_cache_k[layer_idx], ka], dim=1),
                v=torch.cat([video_cache_v[layer_idx], va], dim=1),
                num_heads=self.num_heads,
                ctx_mask=action_attention_mask.to(qa.device),
            )
            sf, sa = rf.shape[1], xa.shape[1]
            force_mask, action_mask = self._force_masks(
                force_state["valid_flat"],
                int(force_state["observed_frames"]),
                video_seq_len,
                video_seq_len,
                sa,
            )
            tri = self._common_attention(
                torch.cat([qf, qa], dim=1),
                torch.cat([video_cache_k[layer_idx], kf, ka], dim=1),
                torch.cat([video_cache_v[layer_idx], vf, va], dim=1),
                self.num_heads,
                torch.cat([force_mask, action_mask], dim=1),
            )
            tri_force, tri_action = tri[:, :sf], tri[:, sf:]
            gate = torch.tanh(self.bridge_gates[str(force_layer)]).to(base_action.dtype)
            has_force = force_state["valid_flat"].any(dim=1)[:, None, None]
            action_mixed = base_action + gate * (tri_action - base_action) * has_force
            self._bridge_residual_rms[str(force_layer)] = (
                (tri_action.detach().float() - base_action.detach().float()).pow(2).mean().sqrt()
            )
            xa = self._apply_expert_post_block_tensor(
                action_block,
                ra,
                action_mixed,
                gma,
                sma,
                sca,
                gfa,
                action_context,
                action_context_mask,
            )
            force_flat = force_block.bridge_residual(
                rf, tri_force, gmf, force_state["valid_flat"]
            )
            force_state = self._finish_force_block(force_state, force_layer, force_flat)
            if force_layer == 7:
                force_state = self.force_expert.run_unbridged_block(force_state, 8)
                force_state = self.force_expert.run_unbridged_block(force_state, 9)
                tail_done = True
        if not tail_done:
            raise RuntimeError("Force tail blocks were not scheduled")
        return xa, force_state

    def bridge_diagnostics(self):
        return {
            int(layer): {
                "raw": param.detach(),
                "tanh": torch.tanh(param.detach()),
                "grad": None if param.grad is None else param.grad.detach(),
                "residual_rms": self._bridge_residual_rms.get(layer),
            }
            for layer, param in self.bridge_gates.items()
        }

