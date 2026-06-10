"""Smoke tests for the cloud-probability head (Part 1, B1 ablation).

Run locally (CPU, synthetic tensors, no dataset needed):

    python test_cloud_mask_head.py

Checks:
  1. mask logits shape is [B,1,H,W] and zero at init (zero-init head);
  2. enabling the head does NOT change the velocity output (warm-start safe);
  3. disabled head leaves last_mask_logits = None;
  4. BCE supervision produces finite loss and non-zero gradients on the head;
  5. sampler mask composition restores clear pixels from mu.
"""
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

from sgm.modules.diffusionmodules.k_diffusion.image_transformer import (
    ImageTransformerDenoiserModelInterface,
)

COMMON = dict(
    in_channels=8,
    out_channels=4,
    patch_size=[2, 2],
    widths=[32, 64],
    depths=[1, 1],
    d_ffs=[64, 128],
    self_attns=[
        {"type": "shifted-window", "d_head": 16, "window_size": 8},
        {"type": "global", "d_head": 16},
    ],
    dropout_rate=[0.0, 0.0],
    mapping_depth=1,
    mapping_width=64,
    mapping_d_ff=128,
    mapping_dropout_rate=0.0,
    use_aux_cond=True,
    aux_channels=5,
    aux_hidden_channels=16,
    aux_global_mode="tmm",
    aux_tmm_gate_hidden_channels=8,
)


def build(predict_cloud_mask):
    torch.manual_seed(0)
    return ImageTransformerDenoiserModelInterface(
        predict_cloud_mask=predict_cloud_mask, **COMMON
    )


def test_head_shape_and_warm_start():
    net_a = build(False)
    net_b = build(True)

    x = torch.randn(2, 8, 32, 32)
    t = torch.rand(2)
    aux = torch.randn(2, 5, 32, 32)
    va = net_a(x, t, aux_cond=aux)
    vb = net_b(x, t, aux_cond=aux)

    assert torch.allclose(va, vb), "velocity output changed by enabling the head"
    assert net_b.last_mask_logits.shape == (2, 1, 32, 32)
    assert bool((net_b.last_mask_logits == 0).all()), "head is not zero-init"
    assert net_a.last_mask_logits is None
    print("[OK] head shape / zero-init / velocity warm-start equivalence")


def test_bce_supervision_gradients():
    net = build(True)
    x = torch.randn(2, 8, 32, 32)
    t = torch.rand(2)
    aux = torch.randn(2, 5, 32, 32)
    target_mask = (torch.rand(2, 1, 32, 32) > 0.7).float()

    _ = net(x, t, aux_cond=aux)
    loss = F.binary_cross_entropy_with_logits(
        net.last_mask_logits.float(), target_mask
    )
    loss.backward()
    assert torch.isfinite(loss), "BCE loss is not finite"
    grad = net.mask_out.proj.weight.grad
    assert grad is not None and grad.abs().sum() > 0, "head got no gradient"
    print(f"[OK] BCE supervision: loss={loss.item():.4f}, head grad flows")


def test_mask_composition():
    # x_final = (1-m)*mu + m*x_pred with confident clear mask must return mu.
    mu = torch.randn(2, 4, 32, 32)
    x_pred = torch.randn(2, 4, 32, 32)
    logits = torch.full((2, 1, 32, 32), -20.0)  # m ~= 0 everywhere (clear)
    m = torch.sigmoid(logits)
    x_final = (1.0 - m) * mu + m * x_pred
    assert torch.allclose(x_final, mu, atol=1e-5)
    logits[:, :, :16] = 20.0  # top half cloudy -> prediction passes through
    m = torch.sigmoid(logits)
    x_final = (1.0 - m) * mu + m * x_pred
    assert torch.allclose(x_final[:, :, :16], x_pred[:, :, :16], atol=1e-5)
    assert torch.allclose(x_final[:, :, 16:], mu[:, :, 16:], atol=1e-5)
    print("[OK] mask composition restores clear pixels / passes cloudy pixels")


if __name__ == "__main__":
    test_head_shape_and_warm_start()
    test_bce_supervision_gradients()
    test_mask_composition()
    print("all cloud-mask-head smoke tests passed")
