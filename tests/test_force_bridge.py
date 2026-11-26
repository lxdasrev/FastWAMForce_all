import torch

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.force_expert import ForceExpert, ForceExpertConfig
from fastwam.models.wan22.force_mot import ForceBridgeMoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT


def _experts():
    video = WanVideoDiT(
        hidden_dim=32, in_dim=4, ffn_dim=64, out_dim=4, text_dim=24, freq_dim=16,
        eps=1e-6, patch_size=(1, 1, 1), num_heads=4, attn_head_dim=8,
        num_layers=30, has_image_input=False, seperated_timestep=True,
        require_vae_embedding=False, require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True, video_attention_mask_mode="first_frame_causal",
    )
    action = ActionDiT(
        hidden_dim=32, action_dim=7, ffn_dim=64, text_dim=24, freq_dim=16,
        eps=1e-6, num_heads=4, attn_head_dim=8, num_layers=30,
    )
    force = ForceExpert(ForceExpertConfig(
        hidden_dim=32, encoder_hidden_dim=8, ffn_dim=64, text_dim=24, freq_dim=16,
        num_heads=4, head_dim=8, bridge_num_heads=4, bridge_head_dim=8,
    ))
    return video, action, force


def test_zero_bridge_gate_strictly_preserves_va_path():
    torch.manual_seed(4)
    video, action, force = _experts()
    mot = ForceBridgeMoT({"video": video, "action": action}, force).eval()
    context = torch.randn(1, 5, 24)
    context_mask = torch.ones(1, 5, dtype=torch.bool)
    video_pre = video.pre_dit(
        x=torch.randn(1, 4, 1, 2, 2), timestep=torch.tensor([0.0]), context=context,
        context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=True,
    )
    action_pre = action.pre_dit(
        action_tokens=torch.randn(1, 3, 7), timestep=torch.tensor([500.0]),
        context=context, context_mask=context_mask,
    )
    force_state = force.prepare(
        torch.randn(1, 33, 2), torch.tensor([500.0]), 1,
        torch.ones(1, 33, 2, dtype=torch.bool), context, context_mask,
    )
    sv, sa = video_pre["tokens"].shape[1], action_pre["tokens"].shape[1]
    mask = torch.ones(sv + sa, sv + sa, dtype=torch.bool)

    base_v, base_a = video_pre["tokens"], action_pre["tokens"]
    for layer in range(30):
        base_v, base_a = mot._forward_joint_layer(
            video.blocks[layer], action.blocks[layer], base_v, base_a,
            video_pre["freqs"], action_pre["freqs"], video_pre["t_mod"], action_pre["t_mod"],
            video_pre["context"], video_pre["context_mask"],
            action_pre["context"], action_pre["context_mask"], mask,
        )
    bridge_v, bridge_a, _ = mot.forward_joint_force_core(
        video_tokens=video_pre["tokens"], action_tokens=action_pre["tokens"],
        force_state=force_state, video_freqs=video_pre["freqs"], action_freqs=action_pre["freqs"],
        video_t_mod=video_pre["t_mod"], action_t_mod=action_pre["t_mod"],
        video_context=video_pre["context"], video_context_mask=video_pre["context_mask"],
        action_context=action_pre["context"], action_context_mask=action_pre["context_mask"],
        attention_mask=mask, first_frame_tokens=sv,
    )
    assert torch.allclose(bridge_v, base_v, atol=1e-6, rtol=1e-6)
    assert torch.allclose(bridge_a, base_a, atol=1e-6, rtol=1e-6)

    # With active gates, changing an invalid force scalar still cannot change
    # the action output through the three-expert bridge.
    for gate in mot.bridge_gates.values():
        gate.data.fill_(0.4)
    valid = torch.ones(1, 33, 2, dtype=torch.bool)
    valid[:, 4, 0] = False
    force_values = torch.randn(1, 33, 2)

    def run(values):
        state = force.prepare(values, torch.tensor([500.0]), 1, valid, context, context_mask)
        return mot.forward_joint_force_core(
            video_tokens=video_pre["tokens"], action_tokens=action_pre["tokens"], force_state=state,
            video_freqs=video_pre["freqs"], action_freqs=action_pre["freqs"],
            video_t_mod=video_pre["t_mod"], action_t_mod=action_pre["t_mod"],
            video_context=video_pre["context"], video_context_mask=video_pre["context_mask"],
            action_context=action_pre["context"], action_context_mask=action_pre["context_mask"],
            attention_mask=mask, first_frame_tokens=sv,
        )[1]

    action_a = run(force_values)
    force_values[:, 4, 0] = 1.0e9
    action_b = run(force_values)
    assert torch.allclose(action_a, action_b, atol=1e-6, rtol=1e-6)

    cache_k, cache_v = mot.prefill_video_cache_tensor(
        video_tokens=video_pre["tokens"], video_freqs=video_pre["freqs"],
        video_t_mod=video_pre["t_mod"], video_context=video_pre["context"],
        video_context_mask=video_pre["context_mask"],
        video_attention_mask=torch.ones(sv, sv, dtype=torch.bool),
    )
    cached_action, cached_force = mot.forward_action_force_with_video_cache(
        action_tokens=action_pre["tokens"], action_freqs=action_pre["freqs"],
        action_t_mod=action_pre["t_mod"], action_context=action_pre["context"],
        action_context_mask=action_pre["context_mask"],
        force_state=force.prepare(force_values, torch.tensor([500.0]), 1, valid, context, context_mask),
        video_cache_k=cache_k, video_cache_v=cache_v,
        action_attention_mask=torch.ones(sa, sv + sa, dtype=torch.bool),
    )
    assert cached_action.shape == action_pre["tokens"].shape
    assert cached_force["tokens"].shape == (1, 33, 2, 32)
