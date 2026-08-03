"""Smoke tests for scalar and spatial-gamma noise-bridge MeanFlow.

Run on the project training environment:

    python test_noise_bridge_meanflow.py
"""

import torch
import torch.nn as nn

from sgm.modules.diffusionmodules.denoiser import ResidualDenoiser
from sgm.modules.diffusionmodules.k_diffusion.image_transformer import (
    ImageTransformerDenoiserModelInterface,
)
from sgm.modules.diffusionmodules.loss_noise_bridge_meanflow import (
    NoiseBridgeMeanFlowLoss,
)
from sgm.modules.diffusionmodules.loss_meanflow import meanflow_c_noise
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


def build_network(predict_cloud_mask=False):
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
        predict_cloud_mask=predict_cloud_mask,
    )
    return CloudRemovalWrapper(model)


def build_loss(
    noise_sigma=0.1,
    noise_ramp_steps=0,
    jvp_mode="fd",
    spatial_noise=False,
    noise_sigma_floor=0.08,
    gamma_delta_tau=0.5,
    gamma_delta_reduction="rms",
    gamma_smooth_kernel=0,
    gamma_head_loss_weight=0.0,
):
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
        spatial_noise=spatial_noise,
        noise_sigma_floor=noise_sigma_floor,
        spatial_noise_source="degradation",
        gamma_delta_tau=gamma_delta_tau,
        gamma_delta_reduction=gamma_delta_reduction,
        gamma_smooth_kernel=gamma_smooth_kernel,
        gamma_head_loss_weight=gamma_head_loss_weight,
        endpoint_loss_weight=0.0,
        consistency_loss_weight=0.0,
        velocity_anchor_loss_weight=0.0,
        non_cloud_identity_loss_weight=0.0,
        ssim_endpoint_loss_weight=0.0,
        cloud_mask_pred_loss_weight=0.0,
    )


class FakeGammaModel:
    def __init__(self):
        self.predict_cloud_mask = True
        self.last_mask_logits = None


class FakeGammaNetwork:
    def __init__(self):
        self.diffusion_model = FakeGammaModel()


def assert_raises(exception_type, fn, contains=None):
    try:
        fn()
    except exception_type as exc:
        if contains is not None:
            assert contains in str(exc)
        return
    raise AssertionError(f"expected {exception_type.__name__}")


def test_bridge_degenerates_at_zero_noise():
    loss_fn = build_loss(noise_sigma=0.0)
    clean = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    cloudy = torch.randn_like(clean)
    noise = torch.randn_like(clean)
    s = torch.rand(BATCH)

    start, x_s, velocity = loss_fn._build_bridge(
        clean, cloudy, s, noise, noise_scale=0.0
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


def test_spatial_noise_scale_from_paired_degradation():
    loss_fn = build_loss(
        noise_sigma=0.2,
        spatial_noise=True,
        noise_sigma_floor=0.1,
        gamma_delta_tau=0.5,
        gamma_head_loss_weight=0.1,
    )
    clean = torch.zeros(BATCH, CHANNELS, HEIGHT, WIDTH)
    cloudy = clean.clone()
    cloudy[:, :, : HEIGHT // 2] = 0.5

    gamma = loss_fn._get_degradation_prob(clean, cloudy)
    scale = loss_fn._get_noise_scale({}, clean, 0.2, cloudy)

    assert gamma.shape == (BATCH, 1, HEIGHT, WIDTH)
    assert torch.equal(
        gamma[:, :, : HEIGHT // 2],
        torch.ones_like(gamma[:, :, : HEIGHT // 2]),
    )
    assert torch.equal(
        gamma[:, :, HEIGHT // 2 :],
        torch.zeros_like(gamma[:, :, HEIGHT // 2 :]),
    )
    assert torch.allclose(
        scale[:, :, : HEIGHT // 2],
        torch.full_like(scale[:, :, : HEIGHT // 2], 0.2),
    )
    assert torch.allclose(
        scale[:, :, HEIGHT // 2 :],
        torch.full_like(scale[:, :, HEIGHT // 2 :], 0.02),
    )
    print("[OK] paired degradation produces the expected spatial noise scale")


def test_spatial_parameter_validation_and_smoothing():
    assert_raises(
        ValueError,
        lambda: build_loss(noise_sigma_floor=1.1),
        "noise_sigma_floor",
    )
    assert_raises(
        ValueError,
        lambda: build_loss(gamma_delta_reduction="bad"),
        "gamma_delta_reduction",
    )
    loss_fn = build_loss(
        spatial_noise=True,
        gamma_smooth_kernel=2,
        gamma_head_loss_weight=0.1,
    )
    clean = torch.zeros(1, CHANNELS, 5, 5)
    cloudy = clean.clone()
    cloudy[:, :, 2, 2] = 1.0
    gamma = loss_fn._get_degradation_prob(clean, cloudy)
    assert 0.0 < gamma[0, 0, 2, 1] < 1.0
    assert gamma.shape == (1, 1, 5, 5)
    print("[OK] spatial gamma parameters validate and even smoothing uses odd kernel")


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


def test_gamma_prepass_dual_time_and_saved_logits_backward():
    class TrainDiffusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.predict_cloud_mask = True
            self.gamma_logit = nn.Parameter(torch.zeros(()))
            self.last_mask_logits = None

    class TrainNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_model = TrainDiffusion()

    network = TrainNetwork()
    loss_fn = build_loss(
        noise_sigma=0.2,
        spatial_noise=True,
        noise_sigma_floor=0.1,
        gamma_delta_tau=0.5,
        gamma_head_loss_weight=1.0,
        jvp_mode="fd",
    )
    loss_fn._sample_pairs = lambda batch, device: (
        torch.full((batch,), 0.25, device=device),
        torch.ones(batch, device=device),
    )
    sigma2st = ConsistencyFlowMatchingSigma2St()
    clean = torch.zeros(BATCH, CHANNELS, HEIGHT, WIDTH)
    cloudy = torch.ones_like(clean) * 0.5
    cond = {"concat": cloudy}
    calls = []

    def fake_denoiser(net, x, sigma, c, st, **extra):
        del c
        calls.append(
            {
                "sigma": sigma.detach().clone(),
                "st": st.detach().clone(),
                "timesteps_r": extra["timesteps_r"].detach().clone(),
            }
        )
        if len(calls) == 1:
            logits = net.diffusion_model.gamma_logit.expand(
                x.shape[0], 1, x.shape[-2], x.shape[-1]
            )
        else:
            # Primary and FD forwards deliberately overwrite the module
            # attribute. Backward must still use the saved prepass tensor.
            logits = torch.zeros(
                x.shape[0], 1, x.shape[-2], x.shape[-1], device=x.device
            )
        net.diffusion_model.last_mask_logits = logits
        return torch.zeros_like(x)

    loss = loss_fn._forward(
        network,
        fake_denoiser,
        cond,
        sigma2st,
        clean,
        cloudy,
        {"global_step": 1},
        noise=torch.zeros_like(clean),
    ).mean()
    loss.backward()

    assert len(calls) == 3  # gamma prepass + primary + FD probe
    assert torch.equal(calls[0]["sigma"], torch.ones(BATCH))
    assert torch.equal(calls[0]["st"], torch.zeros(BATCH))
    assert torch.allclose(
        calls[0]["timesteps_r"],
        meanflow_c_noise(torch.ones(BATCH)),
    )
    assert network.diffusion_model.gamma_logit.grad is not None
    assert network.diffusion_model.gamma_logit.grad.abs() > 0
    assert network.diffusion_model.last_mask_logits.grad_fn is None
    print("[OK] gamma prepass uses s=0/T=1 and saved logits retain gradients")


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


def test_spatial_sampler_cfg_uses_conditional_gamma_and_dual_time():
    cloudy = torch.zeros(BATCH, CHANNELS, HEIGHT, WIDTH)
    noise = torch.ones_like(cloudy)
    cond = {"concat": cloudy}
    uc = {"concat": cloudy.clone()}
    network = FakeGammaNetwork()
    calls = []

    def denoiser(x, sigma, c, st, **extra):
        del c
        calls.append(
            {
                "x": x.detach().clone(),
                "sigma": sigma.detach().clone(),
                "st": st.detach().clone(),
                "timesteps_r": extra["timesteps_r"].detach().clone(),
            }
        )
        if len(calls) == 1:
            # VanillaCFG orders [unconditional, conditional]. Only the
            # conditional half may define gamma used by the restoration path.
            logits = torch.full(
                (2 * BATCH, 1, HEIGHT, WIDTH),
                -80.0,
                device=x.device,
            )
            logits[BATCH + 1] = 80.0
            network.diffusion_model.last_mask_logits = logits
        return torch.zeros_like(x)

    sampler = NoiseBridgeMeanFlowSampler(
        discretization_config=DISCRETIZATION_CONFIG,
        guider_config={
            "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {"scale": 2.0},
        },
        num_steps=1,
        device="cpu",
        noise_sigma=0.2,
        spatial_noise=True,
        noise_sigma_floor=0.1,
        clamp_output=False,
        mask_composite=False,
    )
    sampler.set_sigma2st(ConsistencyFlowMatchingSigma2St())
    sampler.set_network_ref(network)

    output, _ = sampler(denoiser, noise, cloudy, cond, uc)
    expected = torch.empty_like(output)
    expected[0].fill_(0.02)
    expected[1].fill_(0.2)

    assert len(calls) == 2
    assert calls[0]["x"].shape[0] == 2 * BATCH
    assert calls[1]["x"].shape[0] == 2 * BATCH
    assert calls[0]["st"].shape == (2 * BATCH,)
    assert calls[0]["timesteps_r"].shape == (2 * BATCH,)
    assert torch.allclose(
        calls[0]["timesteps_r"],
        meanflow_c_noise(torch.ones(2 * BATCH)),
    )
    assert torch.allclose(calls[1]["x"][:BATCH], expected)
    assert torch.allclose(calls[1]["x"][BATCH:], expected)
    assert torch.allclose(output, expected)
    print("[OK] CFG uses conditional gamma with aligned s=0/T=1 conditioning")


def test_spatial_sampler_tta_keeps_gamma_coordinates_aligned():
    height, width = 5, 7
    marker = torch.linspace(-0.4, 0.4, height * width).reshape(
        1, 1, height, width
    )
    cloudy = marker.expand(1, CHANNELS, height, width).clone()
    noise = torch.linspace(-1.0, 1.0, height * width).reshape(
        1, 1, height, width
    ).expand_as(cloudy).clone()
    cond = {"concat": cloudy}
    network = FakeGammaNetwork()
    calls = []

    def denoiser(x, sigma, c, st, **extra):
        calls.append(extra["timesteps_r"].detach().clone())
        logits = torch.where(
            c["concat"][:, :1] > 0,
            torch.full_like(c["concat"][:, :1], 80.0),
            torch.full_like(c["concat"][:, :1], -80.0),
        )
        network.diffusion_model.last_mask_logits = logits
        return torch.zeros_like(x)

    sampler = NoiseBridgeMeanFlowSampler(
        discretization_config=DISCRETIZATION_CONFIG,
        num_steps=1,
        device="cpu",
        noise_sigma=0.2,
        spatial_noise=True,
        noise_sigma_floor=0.1,
        tta=True,
        clamp_output=False,
        mask_composite=False,
    )
    sampler.set_sigma2st(ConsistencyFlowMatchingSigma2St())
    sampler.set_network_ref(network)

    output, _ = sampler(denoiser, noise, cloudy, cond)
    gamma = (cloudy[:, :1] > 0).to(dtype=cloudy.dtype)
    scale = 0.2 * (0.1 + 0.9 * gamma)
    expected = cloudy + scale * noise

    assert len(calls) == 8  # four transforms, each prepass + restoration
    assert all(
        torch.allclose(value, meanflow_c_noise(torch.ones_like(value)))
        for value in calls
    )
    assert torch.allclose(output, expected, atol=1e-6)
    print("[OK] TTA predicts gamma in each transformed coordinate system")


def test_spatial_sampler_zero_noise_and_multistep_guard():
    cloudy = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    noise = torch.randn_like(cloudy)
    cond = {"concat": cloudy}
    network = FakeGammaNetwork()

    def denoiser(x, sigma, c, st, **extra):
        del sigma, c, st, extra
        network.diffusion_model.last_mask_logits = torch.randn(
            x.shape[0], 1, x.shape[-2], x.shape[-1]
        )
        return torch.zeros_like(x)

    sampler = NoiseBridgeMeanFlowSampler(
        discretization_config=DISCRETIZATION_CONFIG,
        num_steps=1,
        device="cpu",
        noise_sigma=0.0,
        spatial_noise=True,
    )
    sampler.set_sigma2st(ConsistencyFlowMatchingSigma2St())
    sampler.set_network_ref(network)
    output, _ = sampler(denoiser, noise, cloudy, cond)
    assert torch.equal(output, cloudy)

    network.diffusion_model.adaptive_skip_fusion = True
    assert_raises(
        NotImplementedError,
        lambda: sampler(denoiser, noise, cloudy, cond),
        "adaptive_skip_fusion=false",
    )
    network.diffusion_model.adaptive_skip_fusion = False

    sampler.num_steps = 2
    assert_raises(
        NotImplementedError,
        lambda: sampler(denoiser, noise, cloudy, cond),
        "num_steps=1",
    )
    print("[OK] spatial zero-noise is exact and multi-step fails explicitly")


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


def test_spatial_loss_forward_backward_and_head_contract():
    denoiser = ResidualDenoiser(scaling_config=SCALING_CONFIG)
    sigma2st = ConsistencyFlowMatchingSigma2St()
    loss_fn = build_loss(
        spatial_noise=True,
        gamma_head_loss_weight=0.1,
        jvp_mode="fd",
    )
    clean = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    cloudy = torch.randn_like(clean)
    cond = {"concat": cloudy}
    batch = {"global_step": 100}

    assert_raises(
        ValueError,
        lambda: loss_fn._forward(
            build_network(predict_cloud_mask=False),
            denoiser,
            cond,
            sigma2st,
            clean,
            cloudy,
            batch,
        ),
        "predict_cloud_mask=true",
    )

    network = build_network(predict_cloud_mask=True)
    network.diffusion_model.adaptive_skip_fusion = True
    assert_raises(
        ValueError,
        lambda: loss_fn._forward(
            network,
            denoiser,
            cond,
            sigma2st,
            clean,
            cloudy,
            batch,
        ),
        "adaptive_skip_fusion=false",
    )
    network.diffusion_model.adaptive_skip_fusion = False
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
    head_grad = network.diffusion_model.mask_out.proj.weight.grad
    assert head_grad is not None and torch.isfinite(head_grad).all()
    assert head_grad.abs().sum() > 0
    assert network.diffusion_model.patch_in.proj.weight.grad is not None
    print(f"[OK] spatial loss/head forward-backward: {loss.item():.4f}")


if __name__ == "__main__":
    test_bridge_degenerates_at_zero_noise()
    test_noise_ramp()
    test_spatial_noise_scale_from_paired_degradation()
    test_spatial_parameter_validation_and_smoothing()
    test_finite_difference_respects_pair_boundary()
    test_gamma_prepass_dual_time_and_saved_logits_backward()
    test_oracle_sampler_and_reproducibility()
    test_spatial_sampler_cfg_uses_conditional_gamma_and_dual_time()
    test_spatial_sampler_tta_keeps_gamma_coordinates_aligned()
    test_spatial_sampler_zero_noise_and_multistep_guard()
    test_sampler_zero_noise_matches_meanflow_start()
    test_zero_noise_sampler_matches_deterministic_meanflow()
    test_loss_forward_backward("fd")
    test_loss_forward_backward("jvp")
    test_spatial_loss_forward_backward_and_head_contract()
    print("all noise-bridge MeanFlow smoke tests passed")
