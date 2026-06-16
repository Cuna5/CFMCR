"""Sampler for noise-bridge MeanFlow cloud-removal models."""

import torch

from ...util import append_dims, default, tools_scale
from .sampling_cfm import MeanFlowSampler, _TTA_TRANSFORMS


class NoiseBridgeMeanFlowSampler(MeanFlowSampler):
    """MeanFlow sampler whose initial state is cloudy input plus noise.

    ``x_randn`` is supplied by ``ResidualDiffusionEngine.sample``. Reusing that
    tensor instead of drawing noise internally makes seeded evaluation and
    repeated uncertainty sampling controllable by the caller.
    """

    def __init__(self, *args, noise_sigma: float = 0.1, **kwargs):
        if noise_sigma < 0.0:
            raise ValueError("noise_sigma must be non-negative")
        super().__init__(*args, **kwargs)
        self.noise_sigma = float(noise_sigma)

    def _noise_start(self, x_randn, mu):
        if x_randn.shape != mu.shape:
            raise ValueError(
                f"noise shape {tuple(x_randn.shape)} does not match cloudy "
                f"input {tuple(mu.shape)}"
            )
        return mu + self.noise_sigma * x_randn.to(
            device=mu.device,
            dtype=mu.dtype,
        )

    def _prepare_loop(self, x_randn, mu, cond, uc, num_steps):
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
        uc = default(uc, cond)
        s_in = mu.new_ones([mu.shape[0]])
        sigma = s_in * float(getattr(self.discretization, "sigma_max", 1.0))
        sigma_zero = torch.zeros_like(sigma)
        x_init = self._noise_start(x, mu)
        st = self.sigma2st(sigma)

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
