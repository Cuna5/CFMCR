"""Sampler for noise-bridge MeanFlow cloud-removal models."""

import torch

from ...util import append_dims, default, tools_scale
from .sampling_cfm import MeanFlowSampler, _TTA_TRANSFORMS
from .sampling_noise_bridge_cfm import (
    NoiseBridgeConsistencyFlowMatchingSampler,
)


class NoiseBridgeMeanFlowSampler(MeanFlowSampler):
    """MeanFlow sampler whose initial state is cloudy input plus noise.

    ``x_randn`` is supplied by ``ResidualDiffusionEngine.sample``. Reusing that
    tensor instead of drawing noise internally makes seeded evaluation and
    repeated uncertainty sampling controllable by the caller.

    With ``spatial_noise=True``, a one-step MeanFlow prepass predicts gamma from
    the cloudy input at the exact restoration condition ``s=0, T=1``. Gamma is
    then mapped to a per-pixel noise scale. No clean target or dataset mask is
    used by the sampler.
    """

    def __init__(
        self,
        *args,
        noise_sigma: float = 0.1,
        spatial_noise: bool = False,
        noise_sigma_floor: float = 0.08,
        **kwargs,
    ):
        if noise_sigma < 0.0:
            raise ValueError("noise_sigma must be non-negative")
        if not 0.0 <= noise_sigma_floor <= 1.0:
            raise ValueError("noise_sigma_floor must be in [0, 1]")
        super().__init__(*args, **kwargs)
        self.noise_sigma = float(noise_sigma)
        self.spatial_noise = bool(spatial_noise)
        self.noise_sigma_floor = float(noise_sigma_floor)
        if self.spatial_noise and self.mask_composite:
            raise ValueError(
                "spatial_noise uses the shared head as a degradation/gamma "
                "predictor; keep mask_composite=false."
            )

    # Reuse CFM's shape-safe spatial-noise helpers. The MeanFlow-specific
    # prepass below deliberately does not reuse the CFM prepass because it must
    # also supply the jump-target time ``timesteps_r``.
    _noise_start = NoiseBridgeConsistencyFlowMatchingSampler._noise_start
    _get_last_gamma_prob = (
        NoiseBridgeConsistencyFlowMatchingSampler._get_last_mask_prob
    )
    _noise_scale_from_prob = (
        NoiseBridgeConsistencyFlowMatchingSampler._noise_scale_from_mask
    )

    @staticmethod
    def _align_guided_batch(value, model_batch: int, name: str):
        if not torch.is_tensor(value) or value.ndim == 0:
            return value
        if value.shape[0] == model_batch:
            return value
        if model_batch % value.shape[0] != 0:
            raise ValueError(
                f"{name} batch size {value.shape[0]} cannot be aligned with "
                f"guided model batch size {model_batch}."
            )
        return torch.cat([value] * (model_batch // value.shape[0]), dim=0)

    def _denoise_jump(self, x, denoiser, sigma, sigma_next, cond, st, uc):
        """MeanFlow jump denoising with CFG-safe dual-time conditioning."""
        from .loss_meanflow import meanflow_c_noise

        prepared = self.guider.prepare_inputs(x, sigma, cond, uc)
        model_batch = prepared[0].shape[0]
        st_model = self._align_guided_batch(st, model_batch, "st")
        target_time = self.sigma2st(sigma_next)
        timesteps_r = meanflow_c_noise(target_time)
        timesteps_r = self._align_guided_batch(
            timesteps_r,
            model_batch,
            "timesteps_r",
        )
        u_pred = denoiser(
            *prepared,
            st=st_model,
            timesteps_r=timesteps_r,
        )
        return self.guider(u_pred, sigma)

    def _predict_gamma_prob(self, denoiser, mu, sigma, cond, st, uc):
        # The prepass uses the exact same dual-time state as one-step recovery:
        # current s=0 (sigma=1) and target T=1 (sigma_next=0).
        sigma_zero = torch.zeros_like(sigma)
        _ = self._denoise_jump(
            mu,
            denoiser,
            sigma,
            sigma_zero,
            cond,
            st,
            uc,
        )
        return self._get_last_gamma_prob(mu).detach()

    def _predict_noise_scale(self, denoiser, mu, sigma, cond, st, uc):
        if not self.spatial_noise:
            return None
        gamma = self._predict_gamma_prob(denoiser, mu, sigma, cond, st, uc)
        return self._noise_scale_from_prob(gamma)

    def _validate_network_scope(self):
        if self._network_ref is None:
            return
        network = getattr(
            self._network_ref,
            "diffusion_model",
            self._network_ref,
        )
        if getattr(network, "adaptive_skip_fusion", False):
            raise NotImplementedError(
                "NoiseBridgeMeanFlowSampler does not implement adaptive skip "
                "fusion; keep adaptive_skip_fusion=false."
            )

    def _prepare_loop(self, x_randn, mu, cond, uc, num_steps):
        self._validate_network_scope()
        if self.spatial_noise:
            raise NotImplementedError(
                "Noise-bridge MeanFlow spatial_noise currently supports only "
                "the one-step sampler path. Set sampler.num_steps=1."
            )
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps,
            device=self.device,
        )
        uc = default(uc, cond)
        x_init = self._noise_start(x_randn, mu)
        s_in = x_init.new_ones([x_init.shape[0]])
        return x_init, s_in, sigmas, len(sigmas), cond, uc

    def _sample_with_tta(self, denoiser, x, mu, cond, uc, num_steps):
        accumulator = torch.zeros_like(mu)

        for forward_fn, inverse_fn in _TTA_TRANSFORMS:
            noise_t = forward_fn(x)
            mu_t = forward_fn(mu)
            cond_t = self._apply_transform_to_cond(cond, forward_fn)
            uc_t = (
                self._apply_transform_to_cond(uc, forward_fn)
                if uc is not None
                else None
            )
            sample_t = self._sample_single(
                denoiser,
                noise_t,
                mu_t,
                cond_t,
                uc_t,
                num_steps,
            )
            accumulator += inverse_fn(sample_t)

        return accumulator / len(_TTA_TRANSFORMS)

    def _one_step(
        self,
        denoiser,
        x,
        mu,
        cond,
        uc=None,
        return_intermediate: bool = False,
        return_denoised: bool = False,
    ):
        self._validate_network_scope()
        uc = default(uc, cond)
        s_in = mu.new_ones([mu.shape[0]])
        sigma = s_in * float(getattr(self.discretization, "sigma_max", 1.0))
        sigma_zero = torch.zeros_like(sigma)
        st = self.sigma2st(sigma)
        noise_scale = self._predict_noise_scale(
            denoiser,
            mu,
            sigma,
            cond,
            st,
            uc,
        )
        x_init = self._noise_start(x, mu, noise_scale)

        u_pred = self._denoise_jump(
            x_init,
            denoiser,
            sigma,
            sigma_zero,
            cond,
            st,
            uc,
        )
        gap = append_dims(self.sigma2st(sigma_zero) - st, x_init.ndim)
        x_clean = self._finalize(x_init + gap * u_pred, mu)

        others = {}
        if return_intermediate:
            others["intermediates"] = [tools_scale(x_init.clone().detach())]
        if return_denoised:
            others["denoiseds"] = [tools_scale(x_clean.clone().detach())]
        return x_clean, others
