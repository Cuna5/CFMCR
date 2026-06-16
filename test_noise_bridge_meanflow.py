"""Smoke tests for the P1 noise-bridge MeanFlow implementation.

Run on the project training environment:

    python test_noise_bridge_meanflow.py
"""

import torch

from sgm.modules.diffusionmodules.denoiser import ResidualDenoiser
from sgm.modules.diffusionmodules.k_diffusion.image_transformer import (
    ImageTransformerDenoiserModelInterface,
)
from sgm.modules.diffusionmodules.loss_noise_bridge_meanflow import (
    NoiseBridgeMeanFlowLoss,
)
from sgm.modules.diffusionmodules.sampling_noise_bridge_meanflow import (
    NoiseBridgeMeanFlowSampler,
)
from sgm.modules.diffusionmodules.sampling_cfm import MeanFlowSampler
from sgm.modules.diffusionmodules.sigma2st_cfm import (
    ConsistencyFlowMatchingSigma2St,
)
from sgm.modules.diffusionmodules.wrappers import CloudRemovalWrapper


torch.manual_seed(0)

BATCH, CHANNELS, HEIGHT, WIDTH = 2, 3, 16, 16

SCALING_CONFIG = {
    "target": "sgm.modules.diffusionmodules.denoiser_scaling_cfm.MeanFlowScaling"
}
DISCRETIZATION_CONFIG = {
    "target": "sgm.modules.diffusionmodules.discretizer.EDMDiscretization",
    "params": {"sigma_min": 0.001, "sigma_max": 1.0, "rho": 1},
}


def build_network():
    model = ImageTransformerDenoiserModelInterface(
        in_channels=2 * CHANNELS,
        out_channels=CHANNELS,
        patch_size=[1, 1],
        widths=[16, 32],
        depths=[1, 1],
        d_ffs=[32, 64],
        self_attns=[
            {"type": "shifted-window", "d_head": 8, "window_size": 4},
            {"type": "global", "d_head": 8},
        ],
        dropout_rate=[0.0, 0.0],
        mapping_depth=1,
        mapping_width=32,
        mapping_d_ff=64,
        mapping_dropout_rate=0.0,
        use_dual_time=True,
        predict_cloud_mask=False,
    )
    return CloudRemovalWrapper(model)


def build_loss(noise_sigma=0.1, noise_ramp_steps=0, jvp_mode="fd"):
    return NoiseBridgeMeanFlowLoss(
        discretization_config=DISCRETIZATION_CONFIG,
        num_steps=40,
        loss_type="charbonnier",
        meanflow_loss_weight=1.0,
        clean_endpoint_loss_weight=1.0,
        full_pair_prob=0.35,
        t1_pair_prob=0.25,
        equal_pair_prob=0.15,
        jvp_mode=jvp_mode,
        fd_eps=1e-2,
        noise_sigma=noise_sigma,
        noise_ramp_steps=noise_ramp_steps,
        endpoint_loss_weight=0.0,
        consistency_loss_weight=0.0,
        velocity_anchor_loss_weight=0.0,
        non_cloud_identity_loss_weight=0.0,
        ssim_endpoint_loss_weight=0.0,
        cloud_mask_pred_loss_weight=0.0,
    )


def test_bridge_degenerates_at_zero_noise():
    loss_fn = build_loss(noise_sigma=0.0)
    clean = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    cloudy = torch.randn_like(clean)
    noise = torch.randn_like(clean)
    s = torch.rand(BATCH)

    start, x_s, velocity = loss_fn._build_bridge(
        clean, cloudy, s, noise, noise_sigma=0.0
    )
    s_bc = s.view(BATCH, 1, 1, 1)

    assert torch.equal(start, cloudy)
    assert torch.allclose(x_s, (1.0 - s_bc) * cloudy + s_bc * clean)
    assert torch.equal(velocity, clean - cloudy)
    print("[OK] zero noise degenerates to deterministic MeanFlow")


def test_noise_ramp():
    loss_fn = build_loss(noise_sigma=0.2, noise_ramp_steps=100)
    assert loss_fn.noise_sigma_at(0) == 0.0
    assert abs(loss_fn.noise_sigma_at(50) - 0.1) < 1e-8
    assert abs(loss_fn.noise_sigma_at(100) - 0.2) < 1e-8
    assert abs(loss_fn.noise_sigma_at(1000) - 0.2) < 1e-8
    print("[OK] noise ramp reaches the configured target")


def test_finite_difference_respects_pair_boundary():
    loss_fn = build_loss(jvp_mode="fd")
    x_s = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    velocity = torch.randn_like(x_s)
    s = torch.tensor([0.2, 0.5])
    T = torch.tensor([0.8, 0.5])

    def u_fn(x_in, s_in, T_in):
        del T_in
        return 2.0 * x_in + 3.0 * s_in[:, None, None, None]

    cpu_rng_state = torch.get_rng_state()
    u = u_fn(x_s, s, T)
    du_ds = loss_fn._finite_difference_du_ds(
        u_fn,
        u,
        x_s,
        s,
        T,
        velocity,
        cpu_rng_state,
        None,
    )
    expected = 2.0 * velocity + 3.0
    assert torch.allclose(du_ds[0], expected[0], atol=1e-4)
    assert torch.equal(du_ds[1], torch.zeros_like(du_ds[1]))
    print("[OK] finite difference stays inside s <= T")


def test_oracle_sampler_and_reproducibility():
    cloudy = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    clean = torch.randn_like(cloudy)
    noise = torch.randn_like(cloudy)
    cond = {"concat": cloudy}
    noise_sigma = 0.1

    def oracle_denoiser(x, sigma, c, st, **extra):
        del sigma, st, extra
        assert torch.equal(c["concat"], cloudy)
        return clean - x

    sampler = NoiseBridgeMeanFlowSampler(
        discretization_config=DISCRETIZATION_CONFIG,
        num_steps=1,
        device="cpu",
        noise_sigma=noise_sigma,
        clamp_output=False,
        mask_composite=False,
    )
    sampler.set_sigma2st(ConsistencyFlowMatchingSigma2St())

    output_1, _ = sampler(oracle_denoiser, noise, cloudy, cond)
    output_2, _ = sampler(oracle_denoiser, noise.clone(), cloudy, cond)
    assert torch.allclose(output_1, clean, atol=1e-6)
    assert torch.equal(output_1, output_2)
    print("[OK] oracle one-step recovery and fixed-noise reproducibility")


def test_sampler_zero_noise_matches_meanflow_start():
    cloudy = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    random_noise = torch.randn_like(cloudy)
    sampler = NoiseBridgeMeanFlowSampler(
        discretization_config=DISCRETIZATION_CONFIG,
        num_steps=1,
        device="cpu",
        noise_sigma=0.0,
    )
    assert torch.equal(sampler._noise_start(random_noise, cloudy), cloudy)
    print("[OK] sampler starts exactly from cloudy input at zero noise")


def test_zero_noise_sampler_matches_deterministic_meanflow():
    cloudy = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    random_noise = torch.randn_like(cloudy)
    cond = {"concat": cloudy}

    def denoiser(x, sigma, c, st, **extra):
        del sigma, st, extra
        return 0.25 * (c["concat"] - x)

    common = {
        "discretization_config": DISCRETIZATION_CONFIG,
        "num_steps": 1,
        "device": "cpu",
        "clamp_output": False,
        "mask_composite": False,
    }
    deterministic = MeanFlowSampler(**common)
    noise_bridge = NoiseBridgeMeanFlowSampler(noise_sigma=0.0, **common)
    sigma2st = ConsistencyFlowMatchingSigma2St()
    deterministic.set_sigma2st(sigma2st)
    noise_bridge.set_sigma2st(sigma2st)

    expected, _ = deterministic(denoiser, random_noise, cloudy, cond)
    actual, _ = noise_bridge(denoiser, random_noise, cloudy, cond)
    assert torch.equal(actual, expected)
    print("[OK] zero-noise sampler matches deterministic MeanFlow")


def test_loss_forward_backward(jvp_mode):
    network = build_network()
    denoiser = ResidualDenoiser(scaling_config=SCALING_CONFIG)
    sigma2st = ConsistencyFlowMatchingSigma2St()
    loss_fn = build_loss(jvp_mode=jvp_mode)

    clean = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    cloudy = torch.randn_like(clean)
    cond = {"concat": cloudy}
    batch = {"global_step": 100}

    loss = loss_fn._forward(
        network,
        denoiser,
        cond,
        sigma2st,
        clean,
        cloudy,
        batch,
    ).mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert network.diffusion_model.patch_in.proj.weight.grad is not None
    assert network.diffusion_model.time_in_proj_r.weight.grad is not None
    print(f"[OK] loss forward/backward ({jvp_mode}): {loss.item():.4f}")


if __name__ == "__main__":
    test_bridge_degenerates_at_zero_noise()
    test_noise_ramp()
    test_finite_difference_respects_pair_boundary()
    test_oracle_sampler_and_reproducibility()
    test_sampler_zero_noise_matches_meanflow_start()
    test_zero_noise_sampler_matches_deterministic_meanflow()
    test_loss_forward_backward("fd")
    test_loss_forward_backward("jvp")
    print("all noise-bridge MeanFlow smoke tests passed")
