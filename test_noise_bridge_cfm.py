"""Smoke tests for noise-bridge Consistency Flow Matching.

Run in the project training environment:

    python test_noise_bridge_cfm.py
"""

import torch

from sgm.modules.diffusionmodules.denoiser import ResidualDenoiser
from sgm.modules.diffusionmodules.k_diffusion.image_transformer import (
    GammaTimeAdaptiveTokenSplit,
    ImageTransformerDenoiserModelInterface,
    TokenSplit,
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


def build_network(
    predict_cloud_mask=False,
    adaptive_skip_fusion=False,
):
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
        predict_cloud_mask=predict_cloud_mask,
        adaptive_skip_fusion=adaptive_skip_fusion,
    )
    return CloudRemovalWrapper(model)


def test_adaptive_token_split_warm_start_and_gradient():
    original = TokenSplit(in_features=8, out_features=4)
    split = GammaTimeAdaptiveTokenSplit(
        in_features=8,
        out_features=4,
        hidden_channels=4,
        max_delta=0.25,
    )
    incompatible = split.load_state_dict(original.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {
        "gate_in.weight",
        "gate_in.bias",
        "gate_out.weight",
        "gate_out.bias",
    }
    assert not incompatible.unexpected_keys
    x = torch.randn(BATCH, 4, 4, 8)
    skip = torch.randn(BATCH, 8, 8, 4)
    gamma = torch.rand(BATCH, 1, 16, 16)
    timesteps = torch.tensor([-2.0, -0.1])

    expected = original(x, skip)
    actual = split(x, skip, skip_gamma=gamma, timesteps=timesteps)
    assert torch.equal(actual, expected)
    assert torch.equal(split(x, skip), expected)

    actual.square().mean().backward()
    grad = split.gate_out.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum() > 0

    with torch.no_grad():
        split.gate_in.weight.zero_()
        split.gate_in.bias.zero_()
        split.gate_in.weight[0, 0] = 1.0
        split.gate_in.weight[1, 1] = 1.0
        split.gate_out.weight.zero_()
        split.gate_out.bias.zero_()
        split.gate_out.weight[0, 0] = 1.0
        split.gate_out.weight[0, 1] = 1.0
    low = split(
        x,
        skip,
        skip_gamma=torch.zeros_like(gamma),
        timesteps=torch.full_like(timesteps, -2.0),
    )
    high = split(
        x,
        skip,
        skip_gamma=torch.ones_like(gamma),
        timesteps=torch.zeros_like(timesteps),
    )
    assert not torch.allclose(low, high)
    print("[OK] adaptive TokenSplit is warm-start exact and learnable")


def build_loss(
    noise_sigma=0.1,
    noise_ramp_steps=0,
    spatial_noise=False,
    noise_sigma_floor=0.08,
    cloud_mask_pred_loss_weight=0.0,
    spatial_noise_source="mask",
    gamma_delta_tau=0.5,
    gamma_head_loss_weight=0.0,
    gamma_mix_start_step=0,
    gamma_mix_end_step=0,
    gamma_mix_max_prob=0.0,
    residual_wavelet_ll_loss_weight=0.0,
    residual_wavelet_hf_loss_weight=0.0,
    residual_fft_loss_weight=0.0,
    residual_detail_start_step=0,
    residual_detail_warmup_steps=0,
    residual_fft_channels=0,
):
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
        spatial_noise=spatial_noise,
        noise_sigma_floor=noise_sigma_floor,
        spatial_noise_source=spatial_noise_source,
        gamma_delta_tau=gamma_delta_tau,
        gamma_head_loss_weight=gamma_head_loss_weight,
        gamma_mix_start_step=gamma_mix_start_step,
        gamma_mix_end_step=gamma_mix_end_step,
        gamma_mix_max_prob=gamma_mix_max_prob,
        residual_wavelet_ll_loss_weight=residual_wavelet_ll_loss_weight,
        residual_wavelet_hf_loss_weight=residual_wavelet_hf_loss_weight,
        residual_fft_loss_weight=residual_fft_loss_weight,
        residual_detail_start_step=residual_detail_start_step,
        residual_detail_warmup_steps=residual_detail_warmup_steps,
        residual_fft_channels=residual_fft_channels,
        cloud_mask_key="M",
        cloud_loss_weight=1.0,
        non_cloud_identity_loss_weight=0.0,
        ssim_endpoint_loss_weight=0.0,
        cloud_mask_pred_loss_weight=cloud_mask_pred_loss_weight,
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
        noise_scale=0.1,
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
        noise_scale=0.0,
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


def test_spatial_noise_scale_from_mask():
    loss_fn = build_loss(
        noise_sigma=0.2,
        spatial_noise=True,
        noise_sigma_floor=0.1,
    )
    clean = torch.zeros(BATCH, CHANNELS, HEIGHT, WIDTH)
    mask = torch.zeros(BATCH, 1, HEIGHT, WIDTH)
    mask[:, :, : HEIGHT // 2] = 1.0
    scale = loss_fn._get_noise_scale({"M": mask}, clean, noise_sigma=0.2)
    assert scale.shape == (BATCH, 1, HEIGHT, WIDTH)
    assert torch.allclose(
        scale[:, :, : HEIGHT // 2],
        torch.full_like(scale[:, :, : HEIGHT // 2], 0.2),
    )
    assert torch.allclose(
        scale[:, :, HEIGHT // 2 :],
        torch.full_like(scale[:, :, HEIGHT // 2 :], 0.02),
    )
    print("[OK] spatial training noise scale follows the soft cloud mask")


def test_spatial_noise_scale_from_degradation():
    loss_fn = build_loss(
        noise_sigma=0.2,
        spatial_noise=True,
        noise_sigma_floor=0.1,
        spatial_noise_source="degradation",
        gamma_delta_tau=0.5,
    )
    clean = torch.zeros(BATCH, CHANNELS, HEIGHT, WIDTH)
    cloudy = clean.clone()
    cloudy[:, :, : HEIGHT // 2] = 0.5
    scale = loss_fn._get_noise_scale({}, clean, noise_sigma=0.2, mu=cloudy)
    assert scale.shape == (BATCH, 1, HEIGHT, WIDTH)
    assert torch.allclose(
        scale[:, :, : HEIGHT // 2],
        torch.full_like(scale[:, :, : HEIGHT // 2], 0.2),
    )
    assert torch.allclose(
        scale[:, :, HEIGHT // 2 :],
        torch.full_like(scale[:, :, HEIGHT // 2 :], 0.02),
    )
    print("[OK] spatial training noise scale follows paired degradation")


def test_residual_haar_bands_and_gamma_weighting():
    loss_fn = build_loss(
        residual_wavelet_ll_loss_weight=1.0,
        residual_wavelet_hf_loss_weight=1.0,
    )

    # Replicate padding keeps a constant odd-sized image free of high bands.
    constant = torch.ones(1, 1, 5, 7)
    _, lh, hl, hh = loss_fn._haar2d(constant)
    assert torch.count_nonzero(lh) == 0
    assert torch.count_nonzero(hl) == 0
    assert torch.count_nonzero(hh) == 0

    checker = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    checker = checker.repeat(3, 4)[None, None]
    cloudy = torch.zeros_like(checker)
    clean = checker
    endpoint = torch.zeros_like(checker)
    gamma_on = torch.ones(1, 1, *checker.shape[-2:])
    gamma_off = torch.zeros_like(gamma_on)
    _, high_on, _ = loss_fn._residual_multidomain_losses(
        endpoint, clean, cloudy, gamma_on
    )
    _, high_off, _ = loss_fn._residual_multidomain_losses(
        endpoint, clean, cloudy, gamma_off
    )
    assert high_on.item() > 0.0
    assert high_off.item() == 0.0
    print("[OK] Haar high bands are weighted only by paired gamma")


def test_residual_fft_loss_and_gradient():
    loss_fn = build_loss(
        residual_wavelet_hf_loss_weight=1.0,
        residual_fft_loss_weight=1.0,
        residual_fft_channels=2,
    )
    cloudy = torch.zeros(1, 3, 7, 9)
    clean = torch.randn_like(cloudy)
    endpoint = torch.zeros_like(cloudy, requires_grad=True)
    gamma_on = torch.ones(1, 1, 7, 9)
    gamma_off = torch.zeros_like(gamma_on)

    _, _, fft_on = loss_fn._residual_multidomain_losses(
        endpoint, clean, cloudy, gamma_on
    )
    _, _, fft_off = loss_fn._residual_multidomain_losses(
        endpoint, clean, cloudy, gamma_off
    )
    _, _, fft_exact = loss_fn._residual_multidomain_losses(
        clean, clean, cloudy, gamma_on
    )
    assert torch.isfinite(fft_on).all() and fft_on.item() > 0.0
    assert abs(fft_off.item()) < 1e-7
    assert abs(fft_exact.item()) < 1e-7
    fft_on.mean().backward()
    assert endpoint.grad is not None and torch.isfinite(endpoint.grad).all()
    assert endpoint.grad.abs().sum() > 0
    print("[OK] gamma-windowed complex FFT loss is finite and differentiable")


def test_residual_detail_ramp():
    loss_fn = build_loss(
        residual_wavelet_hf_loss_weight=1.0,
        residual_detail_start_step=10,
        residual_detail_warmup_steps=20,
    )
    assert loss_fn._residual_detail_ramp({"global_step": 9}) == 0.0
    assert loss_fn._residual_detail_ramp({"global_step": 10}) == 0.0
    assert loss_fn._residual_detail_ramp({"global_step": 20}) == 0.5
    assert loss_fn._residual_detail_ramp({"global_step": 30}) == 1.0
    print("[OK] residual detail supervision follows its delayed warmup")


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


def test_spatial_sampler_predicts_noise_scale():
    class FakeDiffusionModel:
        def __init__(self):
            self.last_mask_logits = None

    class FakeWrappedNetwork:
        def __init__(self):
            self.diffusion_model = FakeDiffusionModel()

    cloudy = torch.zeros(BATCH, CHANNELS, HEIGHT, WIDTH)
    cond = {"concat": cloudy}
    logits = torch.full((BATCH, 1, HEIGHT, WIDTH), -80.0)
    logits[:, :, : HEIGHT // 2] = 80.0
    network = FakeWrappedNetwork()

    def denoiser(x, sigma, c, st, **extra):
        del x, sigma, c, st, extra
        network.diffusion_model.last_mask_logits = logits
        return torch.zeros_like(cloudy)

    sampler = NoiseBridgeConsistencyFlowMatchingSampler(
        discretization_config=DISCRETIZATION_CONFIG,
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
    sigma = torch.ones(BATCH)
    st = sampler.sigma2st(sigma)
    scale = sampler._predict_noise_scale(denoiser, cloudy, sigma, cond, st, cond)
    assert scale.shape == (BATCH, 1, HEIGHT, WIDTH)
    assert torch.allclose(
        scale[:, :, : HEIGHT // 2],
        torch.full_like(scale[:, :, : HEIGHT // 2], 0.2),
    )
    assert torch.allclose(
        scale[:, :, HEIGHT // 2 :],
        torch.full_like(scale[:, :, HEIGHT // 2 :], 0.02),
    )
    print("[OK] sampler prepass converts predicted mask logits to spatial noise")


def test_adaptive_sampler_prepass_is_non_circular():
    class FakeDiffusionModel:
        def __init__(self):
            self.last_mask_logits = None
            self.adaptive_skip_fusion = True

    class FakeWrappedNetwork:
        def __init__(self):
            self.diffusion_model = FakeDiffusionModel()

    cloudy = torch.zeros(BATCH, CHANNELS, HEIGHT, WIDTH)
    noise = torch.randn_like(cloudy)
    cond = {"concat": cloudy}
    network = FakeWrappedNetwork()
    skip_gamma_calls = []

    def denoiser(x, sigma, c, st, **extra):
        del sigma, c, st
        gamma = extra.get("skip_gamma")
        skip_gamma_calls.append(None if gamma is None else gamma.detach().clone())
        network.diffusion_model.last_mask_logits = torch.zeros(
            x.shape[0],
            1,
            HEIGHT,
            WIDTH,
        )
        return torch.zeros_like(x)

    sampler = NoiseBridgeConsistencyFlowMatchingSampler(
        discretization_config=DISCRETIZATION_CONFIG,
        num_steps=1,
        device="cpu",
        noise_sigma=0.1,
        spatial_noise=False,
        clamp_output=False,
        mask_composite=False,
    )
    sampler.set_sigma2st(ConsistencyFlowMatchingSigma2St())
    sampler.set_network_ref(network)
    sampler._one_step(denoiser, noise, cloudy, cond)

    assert len(skip_gamma_calls) == 2
    assert skip_gamma_calls[0] is None
    assert torch.equal(
        skip_gamma_calls[1],
        torch.full((BATCH, 1, HEIGHT, WIDTH), 0.5),
    )
    print("[OK] adaptive sampler uses base prepass then detached gamma")


def test_adaptive_sampler_cfg_aligns_gamma_batch():
    class FakeDiffusionModel:
        def __init__(self):
            self.last_mask_logits = None
            self.adaptive_skip_fusion = True

    class FakeWrappedNetwork:
        def __init__(self):
            self.diffusion_model = FakeDiffusionModel()

    cloudy = torch.zeros(BATCH, CHANNELS, HEIGHT, WIDTH)
    cond = {"concat": cloudy}
    network = FakeWrappedNetwork()
    skip_gamma_calls = []

    def denoiser(x, sigma, c, st, **extra):
        del sigma, c, st
        skip_gamma_calls.append(extra.get("skip_gamma"))
        logits = torch.full((x.shape[0], 1, HEIGHT, WIDTH), -80.0)
        logits[x.shape[0] // 2 :] = 80.0
        network.diffusion_model.last_mask_logits = logits
        return torch.zeros_like(x)

    sampler = NoiseBridgeConsistencyFlowMatchingSampler(
        discretization_config=DISCRETIZATION_CONFIG,
        guider_config={
            "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {"scale": 2.0},
        },
        num_steps=1,
        device="cpu",
        noise_sigma=0.1,
        spatial_noise=False,
        clamp_output=False,
        mask_composite=False,
    )
    sampler.set_sigma2st(ConsistencyFlowMatchingSigma2St())
    sampler.set_network_ref(network)
    sampler._one_step(
        denoiser,
        torch.randn_like(cloudy),
        cloudy,
        cond,
        cond,
    )

    assert len(skip_gamma_calls) == 2 and skip_gamma_calls[0] is None
    guided_gamma = skip_gamma_calls[1]
    assert guided_gamma.shape == (2 * BATCH, 1, HEIGHT, WIDTH)
    assert torch.equal(guided_gamma, torch.ones_like(guided_gamma))
    assert torch.equal(guided_gamma[:BATCH], guided_gamma[BATCH:])
    print("[OK] adaptive sampler aligns conditional gamma with CFG batches")


def test_loss_forward_backward():
    network = build_network()
    denoiser = ResidualDenoiser(scaling_config=SCALING_CONFIG)
    sigma2st = ConsistencyFlowMatchingSigma2St()
    loss_fn = build_loss()

    def unexpected_multidomain_call(*args, **kwargs):
        del args, kwargs
        raise AssertionError("zero residual weights must skip multi-domain loss")

    loss_fn._residual_multidomain_losses = unexpected_multidomain_call
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


def test_residual_multidomain_loss_forward_backward():
    network = build_network()
    denoiser = ResidualDenoiser(scaling_config=SCALING_CONFIG)
    sigma2st = ConsistencyFlowMatchingSigma2St()
    loss_fn = build_loss(
        spatial_noise=True,
        spatial_noise_source="degradation",
        residual_wavelet_ll_loss_weight=0.02,
        residual_wavelet_hf_loss_weight=0.01,
        residual_fft_loss_weight=0.005,
        residual_detail_start_step=0,
        residual_detail_warmup_steps=0,
        residual_fft_channels=3,
    )
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
    grad = network.diffusion_model.patch_in.proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum() > 0
    print(f"[OK] residual Haar/FFT forward/backward: {loss.item():.4f}")


def test_spatial_loss_with_mask_head_backward():
    network = build_network(predict_cloud_mask=True)
    denoiser = ResidualDenoiser(scaling_config=SCALING_CONFIG)
    sigma2st = ConsistencyFlowMatchingSigma2St()
    loss_fn = build_loss(
        spatial_noise=True,
        noise_sigma_floor=0.1,
        cloud_mask_pred_loss_weight=0.1,
    )
    clean = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    cloudy = torch.randn_like(clean)
    cond = {"concat": cloudy}
    batch = {
        "global_step": 100,
        "M": torch.rand(BATCH, 1, HEIGHT, WIDTH),
    }

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
    grad = network.diffusion_model.mask_out.proj.weight.grad
    assert grad is not None and grad.abs().sum() > 0
    print(f"[OK] spatial noise + mask-head backward: {loss.item():.4f}")


def test_gamma_head_backward_without_mask():
    network = build_network(predict_cloud_mask=True)
    denoiser = ResidualDenoiser(scaling_config=SCALING_CONFIG)
    sigma2st = ConsistencyFlowMatchingSigma2St()
    loss_fn = build_loss(
        spatial_noise=True,
        spatial_noise_source="degradation",
        gamma_delta_tau=0.5,
        noise_sigma_floor=0.1,
        cloud_mask_pred_loss_weight=0.0,
        gamma_head_loss_weight=0.1,
        gamma_mix_start_step=0,
        gamma_mix_end_step=0,
        gamma_mix_max_prob=1.0,
    )
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
    grad = network.diffusion_model.mask_out.proj.weight.grad
    assert grad is not None and grad.abs().sum() > 0
    print(f"[OK] gamma-head degradation P0 backward: {loss.item():.4f}")


def test_adaptive_skip_loss_forward_backward():
    network = build_network(
        predict_cloud_mask=True,
        adaptive_skip_fusion=True,
    )
    # Make the output projection non-zero so this synthetic one-shot test can
    # observe gate gradients immediately; production warm-start keeps it as-is.
    with torch.no_grad():
        network.diffusion_model.patch_out.proj.weight.normal_(std=0.01)
    denoiser = ResidualDenoiser(scaling_config=SCALING_CONFIG)
    sigma2st = ConsistencyFlowMatchingSigma2St()
    loss_fn = build_loss(
        spatial_noise=True,
        spatial_noise_source="degradation",
        gamma_head_loss_weight=0.1,
        gamma_mix_start_step=0,
        gamma_mix_end_step=0,
        gamma_mix_max_prob=1.0,
    )
    clean = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
    cloudy = torch.randn_like(clean)
    cond = {"concat": cloudy}
    batch = {"global_step": 100}

    def teacher_fn(x, sigma, st, c, **extra):
        del sigma, st, c
        gamma = extra.get("skip_gamma")
        assert gamma is not None and not gamma.requires_grad
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
    grads = [split.gate_out.weight.grad for split in network.diffusion_model.splits]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
    assert sum(grad.abs().sum() for grad in grads) > 0
    print(f"[OK] gamma/time adaptive skip forward/backward: {loss.item():.4f}")


def test_adaptive_skip_stage_c_shares_gamma_selection_with_noise():
    class FakeDiffusionModel:
        predict_cloud_mask = True
        adaptive_skip_fusion = True
        last_mask_logits = None

    class FakeNetwork:
        diffusion_model = FakeDiffusionModel()

    network = FakeNetwork()
    loss_fn = build_loss(
        spatial_noise=True,
        spatial_noise_source="degradation",
        gamma_head_loss_weight=0.1,
        gamma_mix_start_step=0,
        gamma_mix_end_step=0,
        gamma_mix_max_prob=0.5,
    )
    loss_fn.start_pair_prob = 0.0
    clean = torch.zeros(BATCH, CHANNELS, HEIGHT, WIDTH)
    cloudy = clean.clone()
    cond = {"concat": cloudy}
    captured = {}

    def denoiser(net, x, sigma, c, st, **extra):
        del sigma, c, st
        net.diffusion_model.last_mask_logits = torch.full(
            (x.shape[0], 1, HEIGHT, WIDTH),
            80.0,
        )
        if "skip_gamma" in extra:
            captured["student_gamma"] = extra["skip_gamma"].detach().clone()
        return torch.zeros_like(x)

    def teacher_fn(x, sigma, st, c, **extra):
        del sigma, st, c
        captured["teacher_gamma"] = extra["skip_gamma"].detach().clone()
        return torch.zeros_like(x)

    original_build = loss_fn._build_bridge_points

    def capture_bridge(*args, **kwargs):
        noise_scale = args[-1]
        captured["noise_scale"] = noise_scale.detach().clone()
        return original_build(*args, **kwargs)

    loss_fn._build_bridge_points = capture_bridge
    original_rand = torch.rand

    def fixed_mix_rand(*shape, **kwargs):
        assert tuple(shape) == (BATCH, 1, 1, 1)
        return torch.tensor(
            [0.25, 0.75],
            device=kwargs.get("device", clean.device),
        ).view(BATCH, 1, 1, 1)

    torch.rand = fixed_mix_rand
    try:
        loss_fn._forward(
            network,
            teacher_fn,
            denoiser,
            cond,
            ConsistencyFlowMatchingSigma2St(),
            clean,
            cloudy,
            {"global_step": 1},
            noise=torch.zeros_like(clean),
        )
    finally:
        torch.rand = original_rand

    assert torch.equal(captured["teacher_gamma"], captured["student_gamma"])
    expected_scale = loss_fn._noise_scale_from_prob(
        captured["teacher_gamma"],
        0.1,
    )
    assert torch.equal(captured["noise_scale"], expected_scale)
    assert captured["teacher_gamma"][0].mean() > 0.99
    assert captured["teacher_gamma"][1].mean() < 0.01
    print("[OK] Stage-C shares one gamma selection for skip fusion and noise")


if __name__ == "__main__":
    test_adaptive_token_split_warm_start_and_gradient()
    test_shared_noise_bridge_points()
    test_zero_noise_degenerates_to_cfm_path()
    test_noise_ramp()
    test_spatial_noise_scale_from_mask()
    test_spatial_noise_scale_from_degradation()
    test_residual_haar_bands_and_gamma_weighting()
    test_residual_fft_loss_and_gradient()
    test_residual_detail_ramp()
    test_oracle_sampler_and_reproducibility()
    test_zero_noise_sampler_matches_cfm()
    test_spatial_sampler_predicts_noise_scale()
    test_adaptive_sampler_prepass_is_non_circular()
    test_adaptive_sampler_cfg_aligns_gamma_batch()
    test_loss_forward_backward()
    test_residual_multidomain_loss_forward_backward()
    test_spatial_loss_with_mask_head_backward()
    test_gamma_head_backward_without_mask()
    test_adaptive_skip_loss_forward_backward()
    test_adaptive_skip_stage_c_shares_gamma_selection_with_noise()
    print("all noise-bridge CFM smoke tests passed")
