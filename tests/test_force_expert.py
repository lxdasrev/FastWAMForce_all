import torch

from fastwam.models.wan22.force_expert import (
    compute_force_flow_loss,
    ForceExpert,
    ForceExpertConfig,
    safe_attention,
)


def tiny_config():
    return ForceExpertConfig(
        hidden_dim=32,
        encoder_hidden_dim=8,
        ffn_dim=64,
        text_dim=24,
        freq_dim=16,
        num_heads=4,
        head_dim=8,
        bridge_num_heads=4,
        bridge_head_dim=8,
    )


def test_temporal_mask_contract():
    mask = ForceExpert.build_temporal_mask(49, 17)
    assert mask[:17, :17].all()
    assert not mask[:17, 17:].any()
    assert mask[17:, :].all()


def test_safe_attention_all_invalid_is_zero_and_finite():
    q = torch.randn(2, 4, 3, 8, requires_grad=True)
    k = torch.randn(2, 4, 5, 8, requires_grad=True)
    v = torch.randn(2, 4, 5, 8, requires_grad=True)
    allowed = torch.zeros(2, 3, 5, dtype=torch.bool)
    out = safe_attention(q, k, v, allowed)
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros_like(out))
    out.sum().backward()
    assert torch.isfinite(q.grad).all()


def test_encoder_33_and_49_shapes_and_invalid_zero():
    model = ForceExpert(tiny_config())
    context = torch.randn(2, 5, 24)
    context_mask = torch.ones(2, 5, dtype=torch.bool)
    timestep = torch.tensor([250.0, 750.0])
    for observed, total in ((1, 33), (17, 49)):
        force = torch.randn(2, total, 2)
        valid = torch.ones_like(force, dtype=torch.bool)
        valid[0, 3, 1] = False
        state = model.prepare(force, timestep, observed, valid, context, context_mask)
        assert state["tokens"].shape == (2, total, 2, 32)
        assert torch.equal(state["tokens"][0, 3, 1], torch.zeros(32))
        state = model.run_unbridged_block(state, 0)
        assert torch.equal(state["tokens"][0, 3, 1], torch.zeros(32))
        pred = model.post(state)
        assert pred.shape == force.shape
        assert torch.equal(pred[:, :observed], torch.zeros_like(pred[:, :observed]))
        assert torch.equal(pred[0, 3, 1], torch.zeros_like(pred[0, 3, 1]))


def test_invalid_values_cannot_change_valid_output():
    torch.manual_seed(7)
    model = ForceExpert(tiny_config()).eval()
    force = torch.randn(1, 33, 2)
    valid = torch.ones_like(force, dtype=torch.bool)
    valid[:, 4, 0] = False
    kwargs = dict(
        timestep=torch.tensor([500.0]),
        observed_frames=1,
        effective_valid=valid,
        context=torch.randn(1, 3, 24),
        context_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    a = model.run_unbridged_block(model.prepare(force, **kwargs), 0)["tokens"]
    force[:, 4, 0] = 1.0e9
    b = model.run_unbridged_block(model.prepare(force, **kwargs), 0)["tokens"]
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_force_loss_per_sample_normalization_and_empty_sample_skip():
    pred = torch.tensor([[[1.0], [3.0]], [[100.0], [100.0]]], requires_grad=True)
    target = torch.zeros_like(pred)
    valid = torch.tensor([[[True], [True]], [[False], [False]]])
    terms = compute_force_flow_loss(pred, target, valid, torch.tensor([2.0, 9.0]), 0.5)
    assert torch.allclose(terms["raw_force_loss"], torch.tensor(5.0))
    assert torch.allclose(terms["scheduler_weighted_force_loss"], torch.tensor(10.0))
    assert torch.allclose(terms["force_loss"], torch.tensor(5.0))
    terms["force_loss"].backward()
    assert torch.isfinite(pred.grad).all()


def test_force_loss_all_invalid_is_graph_zero():
    pred = torch.randn(2, 3, 2, requires_grad=True)
    terms = compute_force_flow_loss(
        pred, torch.zeros_like(pred), torch.zeros_like(pred, dtype=torch.bool), torch.ones(2), 1.0
    )
    assert terms["force_loss"].item() == 0.0
    terms["force_loss"].backward()
    assert torch.equal(pred.grad, torch.zeros_like(pred))

