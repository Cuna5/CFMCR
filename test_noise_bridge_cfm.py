"""Smoke tests for noise-bridge Consistency Flow Matching.

Run in the project training environment:

    python test_noise_bridge_cfm.py
"""

import torch

from sgm.modules.diffusionmodules.denoiser import ResidualDenoiser
from sgm.modules.diffusionmodules.k_diffusion.image_transformer import (
    ImageTransformerDenoiserModelInterface,
)
from sgm.modules.diffusionmodules.loss_noise_bridge_cfm import (
    NoiseBridgeConsistencyFlowMatchingLoss,
)
from sgm.modules.diffusionmodules.sampling_cfm import (
    ConsistencyFlowMatchingSampler,
)
from sgm.modules.diffusionmodules.sampling_noise_bridge_cfm import (
    NoiseBridgeConsistencyFlowMatchingSampler,
)
from sgm.modules.diffusionmodules.sigma2st_cfm import (
    ConsistencyFlowMatchingSigma2St,
)
from sgm.modules.diffusionmodules.wrappers import CloudRemovalWrapper


torch.manual_seed(0)

BATCH, CHANNELS, HEIGHT, WIDTH = 2, 3, 16, 16
SCALING_CONFIG = {
    "target": (
        "sgm.modules.diffusionmodules.denoiser_scaling_cfm."
        "ConsistencyFlowMatchingScaling"
    )
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
        predict_cloud_mask=False,
    )
    return CloudRemovalWrapper(model)


def build_loss(noise_sigma=0.1, noise_ramp_steps=0):
    return NoiseBridgeConsistencyFlowMatchingLoss(
        discretization_config=DISCRETIZATION_CONFIG,
        num_steps=40,
        loss_type="charbonnier",
        endpoint_loss_weight=0.5,
        consistency_loss_weight=0.5,
        velocity_anchor_loss_weight=1.0,
        clean_endpoint_loss_weight=1.0,
        start_pair_prob=0.35,
        consistency_warmup_steps=0,
        noise_sigma=noise_sigma,
        noise_ramp_steps=noise_ramp_steps,
        cloud_loss_weight=1.0,
        non_cloud_identity_loss_weight=0.0,
        ssim_endpoint_loss_weight=0.0,
        cloud_mask_pred_loss_weight=0.0,
    )


def test_shared_noise_bridge_points():
    loss_fn = build_loss(noise_sigma=0.1)
    clean = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    cloudy = torch.randn_like(clean)
    noise = torch.randn_like(clean)
    t_current = torch.tensor([0.0, 0.25])
    t_next = torch.tensor([0.5, 0.75])

    start, current, next_point, velocity = loss_fn._build_bridge_points(
        clean,
        cloudy,
        t_current,
        t_next,
        noise,
        noise_sigma=0.1,
    )
    expected_start = cloudy + 0.1 * noise
    assert torch.allclose(start, expected_start)
    assert torch.allclose(velocity, clean - expected_start)
    assert torch.allclose(current[0], expected_start[0])
    expected_next = 0.25 * expected_start[1] + 0.75 * clean[1]
    assert torch.allclose(next_point[1], expected_next)
    print("[OK] student and teacher points share one noisy path")


def test_zero_noise_degenerates_to_cfm_path():
    loss_fn = build_loss(noise_sigma=0.0)
    clean = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    cloudy = torch.randn_like(clean)
    noise = torch.randn_like(clean)
    t_current = torch.rand(BATCH)
    t_next = t_current + 0.5 * (1.0 - t_current)
    start, current, next_point, velocity = loss_fn._build_bridge_points(
        clean,
        cloudy,
        t_current,
        t_next,
        noise,
        noise_sigma=0.0,
    )
    current_t = t_current[:, None, None, None]
    next_t = t_next[:, None, None, None]
    assert torch.equal(start, cloudy)
    assert torch.allclose(
        current,
        (1.0 - current_t) * cloudy + current_t * clean,
    )
    assert torch.allclose(
        next_point,
        (1.0 - next_t) * cloudy + next_t * clean,
    )
    assert torch.equal(velocity, clean - cloudy)
    print("[OK] zero noise degenerates to deterministic CFM")


def test_noise_ramp():
    loss_fn = build_loss(noise_sigma=0.2, noise_ramp_steps=100)
    assert loss_fn.noise_sigma_at(0) == 0.0
    assert abs(loss_fn.noise_sigma_at(50) - 0.1) < 1e-8
    assert abs(loss_fn.noise_sigma_at(100) - 0.2) < 1e-8
    print("[OK] CFM noise ramp reaches the configured target")


def test_oracle_sampler_and_reproducibility():
    cloudy = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    clean = torch.randn_like(cloudy)
    noise = torch.randn_like(cloudy)
    cond = {"concat": cloudy}

    def oracle_denoiser(x, sigma, c, st, **extra):
        del sigma, st, extra
        assert torch.equal(c["concat"], cloudy)
        return clean - x

    sampler = NoiseBridgeConsistencyFlowMatchingSampler(
        discretization_config=DISCRETIZATION_CONFIG,
        num_steps=1,
        device="cpu",
        noise_sigma=0.1,
        clamp_output=False,
        mask_composite=False,
    )
    sampler.set_sigma2st(ConsistencyFlowMatchingSigma2St())
    output_1, _ = sampler(oracle_denoiser, noise, cloudy, cond)
    output_2, _ = sampler(oracle_denoiser, noise.clone(), cloudy, cond)
    assert torch.allclose(output_1, clean, atol=1e-6)
    assert torch.equal(output_1, output_2)
    print("[OK] oracle CFM recovery and fixed-noise reproducibility")


def test_zero_noise_sampler_matches_cfm():
    cloudy = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    noise = torch.randn_like(cloudy)
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
    deterministic = ConsistencyFlowMatchingSampler(**common)
    noise_bridge = NoiseBridgeConsistencyFlowMatchingSampler(
        noise_sigma=0.0,
        **common,
    )
    sigma2st = ConsistencyFlowMatchingSigma2St()
    deterministic.set_sigma2st(sigma2st)
    noise_bridge.set_sigma2st(sigma2st)
    expected, _ = deterministic(denoiser, noise, cloudy, cond)
    actual, _ = noise_bridge(denoiser, noise, cloudy, cond)
    assert torch.equal(actual, expected)
    print("[OK] zero-noise sampler matches deterministic CFM")


def test_loss_forward_backward():
    network = build_network()
    denoiser = ResidualDenoiser(scaling_config=SCALING_CONFIG)
    sigma2st = ConsistencyFlowMatchingSigma2St()
    loss_fn = build_loss()
    clean = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    cloudy = torch.randn_like(clean)
    cond = {"concat": cloudy}
    batch = {"global_step": 100}

    def teacher_fn(x, sigma, st, c, **extra):
        del sigma, st, c, extra
        return torch.zeros_like(x)

    loss = loss_fn._forward(
        network,
        teacher_fn,
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
    print(f"[OK] noise-bridge CFM forward/backward: {loss.item():.4f}")


if __name__ == "__main__":
    test_shared_noise_bridge_points()
    test_zero_noise_degenerates_to_cfm_path()
    test_noise_ramp()
    test_oracle_sampler_and_reproducibility()
    test_zero_noise_sampler_matches_cfm()
    test_loss_forward_backward()
    print("all noise-bridge CFM smoke tests passed")
