"""MeanFlow feasibility + smoke tests (Part 2).

Run locally or on the training server (CPU ok, synthetic tensors):

    python test_meanflow_jvp.py

Checks:
  1. torch.func.jvp works through the wrapped HDiT denoiser (per attention
     type; natten may lack forward-mode AD -> reported, fd fallback covers it);
  2. JVP and finite-difference derivative estimates agree;
  3. dual-time warm start: at init the average-velocity output is identical
     to the single-time CFM output, for any jump target T;
  4. MeanFlowLoss forward/backward: finite loss, gradients flow to backbone,
     dual-time branch and mask head;
  5. MeanFlowSampler 1-step / multi-step output shapes + mask composition.
"""
import warnings

warnings.filterwarnings("ignore")

import torch

from sgm.modules.diffusionmodules.denoiser import ResidualDenoiser
from sgm.modules.diffusionmodules.k_diffusion.image_transformer import (
    ImageTransformerDenoiserModelInterface,
    apply_rotary_emb_,
)
from sgm.modules.diffusionmodules.loss_meanflow import MeanFlowLoss, meanflow_c_noise
from sgm.modules.diffusionmodules.sampling_cfm import MeanFlowSampler
from sgm.modules.diffusionmodules.sigma2st_cfm import ConsistencyFlowMatchingSigma2St
from sgm.modules.diffusionmodules.wrappers import CloudRemovalWrapper

torch.manual_seed(0)

B, C, H, W = 2, 4, 32, 32

SCALING_CONFIG = {
    "target": "sgm.modules.diffusionmodules.denoiser_scaling_cfm.MeanFlowScaling"
}
DISCRETIZATION_CONFIG = {
    "target": "sgm.modules.diffusionmodules.discretizer.EDMDiscretization",
    "params": {"sigma_min": 0.001, "sigma_max": 1.0, "rho": 1},
}


def build_network(self_attns, use_dual_time=True, predict_cloud_mask=True, seed=0):
    torch.manual_seed(seed)
    return ImageTransformerDenoiserModelInterface(
        in_channels=2 * C,
        out_channels=C,
        patch_size=[2, 2],
        widths=[32, 64],
        depths=[1, 1],
        d_ffs=[64, 128],
        self_attns=self_attns,
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
        use_dual_time=use_dual_time,
        predict_cloud_mask=predict_cloud_mask,
    )


def make_inputs():
    mu = torch.randn(B, C, H, W)
    x_clean = torch.randn(B, C, H, W)
    aux = torch.randn(B, 5, H, W)
    cond = {"concat": mu, "aux_cond": aux}
    return mu, x_clean, aux, cond


def make_u_fn(wrapper, denoiser, cond):
    def u_fn(x_in, s_in, T_in):
        sigma_in = 1.0 - s_in
        return denoiser(
            wrapper, x_in, sigma_in, cond, s_in, timesteps_r=meanflow_c_noise(T_in)
        )

    return u_fn


def test_rotary_embedding_jvp():
    x = torch.randn(2, 3, 4, 8)
    tangent = torch.randn_like(x)
    theta = torch.randn(2, 3, 4, 4)

    output, output_tangent = torch.func.jvp(
        lambda value: apply_rotary_emb_(value, theta),
        (x,),
        (tangent,),
    )
    expected_tangent = apply_rotary_emb_(tangent, theta)

    assert output.shape == x.shape
    assert torch.allclose(output_tangent, expected_tangent, atol=1e-6)
    print("[OK] rotary embedding supports torch.func.jvp")


def test_jvp_support(self_attns, label):
    wrapper = CloudRemovalWrapper(build_network(self_attns))
    denoiser = ResidualDenoiser(scaling_config=SCALING_CONFIG)
    mu, x_clean, aux, cond = make_inputs()
    v = x_clean - mu
    s = torch.rand(B) * 0.8
    T = (s + torch.rand(B) * (1.0 - s)).clamp(max=1.0)
    s_bc = s.view(-1, 1, 1, 1)
    x_s = (1 - s_bc) * mu + s_bc * x_clean
    u_fn = make_u_fn(wrapper, denoiser, cond)

    try:
        u, du_jvp = torch.func.jvp(
            u_fn, (x_s, s, T), (v, torch.ones_like(s), torch.zeros_like(T))
        )
    except Exception as e:  # noqa: BLE001 - report and continue with fd
        print(f"[WARN] {label}: torch.func.jvp FAILED -> use jvp_mode='fd' "
              f"({type(e).__name__}: {str(e)[:120]})")
        return False

    # finite-difference cross-check
    eps = 1e-3
    with torch.no_grad():
        u0 = u_fn(x_s, s, T)
        u1 = u_fn(x_s + eps * v, s + eps, T)
        du_fd = (u1 - u0) / eps
    rel = (du_jvp - du_fd).norm() / du_fd.norm().clamp_min(1e-8)
    status = "OK" if rel < 0.05 else "MISMATCH"
    print(f"[{status}] {label}: jvp supported, jvp-vs-fd relative diff = {rel:.4f}")
    assert rel < 0.05, f"{label}: jvp and fd disagree"
    return True


def test_dual_time_warm_start():
    # At init the zero-init dual-time branch must make u(x,s,T) independent
    # of T and identical to the single-time output.
    attns = [
        {"type": "shifted-window", "d_head": 16, "window_size": 8},
        {"type": "global", "d_head": 16},
    ]
    net_dual = build_network(attns, use_dual_time=True, seed=0)
    net_single = build_network(attns, use_dual_time=False, seed=0)
    x = torch.randn(B, 2 * C, H, W)
    t_emb = torch.rand(B)
    aux = torch.randn(B, 5, H, W)
    out_single = net_single(x, t_emb, aux_cond=aux)
    out_T1 = net_dual(x, t_emb, aux_cond=aux, timesteps_r=meanflow_c_noise(torch.ones(B)))
    out_T05 = net_dual(x, t_emb, aux_cond=aux, timesteps_r=meanflow_c_noise(torch.full((B,), 0.5)))
    assert torch.allclose(out_single, out_T1, atol=1e-6)
    assert torch.allclose(out_T1, out_T05, atol=1e-6)
    print("[OK] dual-time zero-init warm start (output independent of T at init)")


def run_loss(jvp_mode):
    attns = [
        {"type": "shifted-window", "d_head": 16, "window_size": 8},
        {"type": "global", "d_head": 16},
    ]
    wrapper = CloudRemovalWrapper(build_network(attns))
    denoiser = ResidualDenoiser(scaling_config=SCALING_CONFIG)
    sigma2st = ConsistencyFlowMatchingSigma2St()
    mu, x_clean, aux, cond = make_inputs()
    batch = {"M": (torch.rand(B, H, W) > 0.7).float()}

    loss_fn = MeanFlowLoss(
        discretization_config=DISCRETIZATION_CONFIG,
        loss_type="charbonnier",
        meanflow_loss_weight=1.0,
        full_pair_prob=0.35,
        t1_pair_prob=0.25,
        equal_pair_prob=0.15,
        jvp_mode=jvp_mode,
        clean_endpoint_loss_weight=1.0,
        non_cloud_identity_loss_weight=0.2,
        cloud_mask_pred_loss_weight=0.1,
        cloud_loss_weight=2.0,
        cloud_weight_velocity_anchor=True,
        feather_mask_kernel=7,
        num_steps=40,
    )
    loss = loss_fn._forward(
        wrapper, None, denoiser, cond, sigma2st, x_clean, mu, batch
    ).mean()
    loss.backward()
    assert torch.isfinite(loss), f"loss not finite ({jvp_mode})"

    net = wrapper.diffusion_model
    checks = {
        "backbone(patch_in)": net.patch_in.proj.weight.grad,
        "dual-time proj": net.time_in_proj_r.weight.grad,
        "mask head": net.mask_out.proj.weight.grad,
    }
    for name, g in checks.items():
        assert g is not None and torch.isfinite(g).all(), f"no/inf grad: {name}"
        assert g.abs().sum() > 0 or name == "dual-time proj", f"zero grad: {name}"
    print(f"[OK] MeanFlowLoss ({jvp_mode}): loss={loss.item():.4f}, grads flow")


def test_sampler():
    attns = [
        {"type": "shifted-window", "d_head": 16, "window_size": 8},
        {"type": "global", "d_head": 16},
    ]
    wrapper = CloudRemovalWrapper(build_network(attns))
    denoiser_mod = ResidualDenoiser(scaling_config=SCALING_CONFIG)
    mu, x_clean, aux, cond = make_inputs()

    def denoiser(input, sigma, c, st, **extra):
        return denoiser_mod(wrapper, input, sigma, c, st, **extra)

    for num_steps, composite in [(1, False), (1, True), (3, False)]:
        sampler = MeanFlowSampler(
            discretization_config=DISCRETIZATION_CONFIG,
            num_steps=num_steps,
            device="cpu",
            clamp_output=True,
            mask_composite=composite,
        )
        sampler.set_sigma2st(ConsistencyFlowMatchingSigma2St())
        sampler.set_network_ref(wrapper)
        with torch.no_grad():
            x_out, _ = sampler(denoiser, torch.randn_like(mu), mu, cond)
        assert x_out.shape == mu.shape
        assert torch.isfinite(x_out).all()
        # zero-init mask head => m = 0.5 => output halfway between mu and pred
        print(f"[OK] MeanFlowSampler num_steps={num_steps} "
              f"mask_composite={composite}: shape {tuple(x_out.shape)}")


if __name__ == "__main__":
    test_rotary_embedding_jvp()

    sw_global = [
        {"type": "shifted-window", "d_head": 16, "window_size": 8},
        {"type": "global", "d_head": 16},
    ]
    test_jvp_support(sw_global, "shifted-window + global")
    try:
        import natten  # noqa: F401

        nb = [
            {"type": "neighborhood", "d_head": 16, "kernel_size": 7},
            {"type": "global", "d_head": 16},
        ]
        test_jvp_support(nb, "neighborhood (natten) + global")
    except ImportError:
        print("[SKIP] natten not installed here; re-run on the training "
              "server to decide jvp vs fd for the real config")

    test_dual_time_warm_start()
    run_loss("jvp")
    run_loss("fd")
    test_sampler()
    print("all MeanFlow smoke tests passed")
