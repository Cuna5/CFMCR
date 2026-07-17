"""Sampler for noise-bridge Consistency Flow Matching models."""

import torch
import torch.nn.functional as F

from ...util import append_dims, default, tools_scale
from .sampling_cfm import ConsistencyFlowMatchingSampler, _TTA_TRANSFORMS


class NoiseBridgeConsistencyFlowMatchingSampler(
    ConsistencyFlowMatchingSampler
):
    """CFM sampler initialized from cloudy input plus caller-provided noise.

    With ``spatial_noise=True`` the shared one-channel head predicts a soft
    degradation / gamma probability from the cloudy input, which is mapped to a
    per-pixel noise scale. The sampler never consumes label-derived masks.
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
                "spatial_noise uses the predicted gamma probability to "
                "modulate the input noise; keep mask_composite=false."
            )
        # Additional churn leaves the path used during training.
        self.s_churn = 0.0

    def _noise_start(self, x_randn, mu, noise_scale=None):
        if x_randn.shape != mu.shape:
            raise ValueError(
                f"noise shape {tuple(x_randn.shape)} does not match cloudy "
                f"input {tuple(mu.shape)}"
            )
        if noise_scale is None:
            noise_scale = self.noise_sigma
        elif torch.is_tensor(noise_scale):
            noise_scale = noise_scale.to(device=mu.device, dtype=mu.dtype)
            if noise_scale.ndim != mu.ndim:
                raise ValueError(
                    "noise_scale must broadcast as [B,1,H,W] or [B,C,H,W], "
                    f"got shape {tuple(noise_scale.shape)}"
                )
            if noise_scale.shape[0] != mu.shape[0]:
                raise ValueError(
                    "noise_scale batch size does not match cloudy input: "
                    f"{noise_scale.shape[0]} vs {mu.shape[0]}"
                )
            if noise_scale.shape[1] not in (1, mu.shape[1]):
                raise ValueError(
                    "noise_scale channel dimension must be 1 or match the "
                    f"cloudy input channels, got {noise_scale.shape[1]}"
                )
            if noise_scale.shape[-2:] != mu.shape[-2:]:
                raise ValueError(
                    "noise_scale spatial size does not match cloudy input: "
                    f"{tuple(noise_scale.shape[-2:])} vs {tuple(mu.shape[-2:])}"
                )
        return mu + noise_scale * x_randn.to(
            device=mu.device,
            dtype=mu.dtype,
        )

    def _get_last_mask_prob(self, mu):
        if self._network_ref is None:
            raise RuntimeError(
                "spatial noise or adaptive skip fusion requires "
                "sampler.set_network_ref(model)."
            )
        net = getattr(self._network_ref, "diffusion_model", self._network_ref)
        logits = getattr(net, "last_mask_logits", None)
        if logits is None:
            raise RuntimeError(
                "spatial noise or adaptive skip fusion requires "
                "network_config.params."
                "predict_cloud_mask=true so the shared gamma head can "
                "produce last_mask_logits."
            )
        mask = torch.sigmoid(logits.float()).to(device=mu.device, dtype=mu.dtype)
        if mask.ndim != mu.ndim:
            raise ValueError(
                f"Predicted cloud mask has shape {tuple(mask.shape)}, "
                f"expected BCHW for cloudy input {tuple(mu.shape)}"
            )
        if mask.shape[0] == 2 * mu.shape[0]:
            mask = mask[-mu.shape[0] :]
        elif mask.shape[0] != mu.shape[0]:
            raise ValueError(
                "Predicted cloud mask batch size does not match cloudy input: "
                f"{mask.shape[0]} vs {mu.shape[0]}"
            )
        if mask.shape[1] != 1:
            mask = mask.max(dim=1, keepdim=True).values
        if mask.shape[-2:] != mu.shape[-2:]:
            mask = F.interpolate(
                mask,
                size=mu.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return mask.clamp(0.0, 1.0)

    def _noise_scale_from_mask(self, mask):
        floor = self.noise_sigma_floor
        return self.noise_sigma * (floor + (1.0 - floor) * mask)

    def _adaptive_skip_enabled(self):
        if self._network_ref is None:
            return False
        net = getattr(self._network_ref, "diffusion_model", self._network_ref)
        return bool(getattr(net, "adaptive_skip_fusion", False))

    def _predict_gamma_prob(self, denoiser, mu, sigma, cond, st, uc):
        # Deliberately omit skip_gamma: the predictor must use the original
        # scalar skip fusion, otherwise gamma would condition its own estimate.
        _ = self._denoise(mu, denoiser, sigma, cond, st, uc)
        return self._get_last_mask_prob(mu).detach()

    def _predict_noise_scale(self, denoiser, mu, sigma, cond, st, uc):
        if not self.spatial_noise:
            return None
        gamma = self._predict_gamma_prob(denoiser, mu, sigma, cond, st, uc)
        return self._noise_scale_from_mask(gamma)

    def _prepare_loop(self, x_randn, mu, cond, uc, num_steps):
        if self.spatial_noise or self._adaptive_skip_enabled():
            raise NotImplementedError(
                "Noise-bridge spatial noise and adaptive skip fusion currently "
                "support the one-step sampler path. Set sampler.num_steps=1."
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
        sigma_max = mu.new_tensor(
            float(getattr(self.discretization, "sigma_max", 1.0))
        )
        uc = default(uc, cond)
        s_in = mu.new_ones([mu.shape[0]])
        sigma = s_in * sigma_max
        st = self.sigma2st(sigma)
        adaptive_skip_enabled = self._adaptive_skip_enabled()
        gamma = None
        if self.spatial_noise or adaptive_skip_enabled:
            gamma = self._predict_gamma_prob(
                denoiser,
                mu,
                sigma,
                cond,
                st,
                uc,
            )
        noise_scale = (
            self._noise_scale_from_mask(gamma)
            if self.spatial_noise
            else None
        )
        x_init = self._noise_start(x, mu, noise_scale)

        if adaptive_skip_enabled:
            velocity = self._denoise(
                x_init,
                denoiser,
                sigma,
                cond,
                st,
                uc,
                skip_gamma=gamma,
            )
        else:
            velocity = self._denoise(x_init, denoiser, sigma, cond, st, uc)
        x_clean = x_init + append_dims(sigma, x_init.ndim) * velocity
        x_clean = self._finalize(x_clean, mu)

        others = {}
        if return_intermediate:
            others["intermediates"] = [tools_scale(x_init.clone().detach())]
        if return_denoised:
            others["denoiseds"] = [tools_scale(x_clean.clone().detach())]
        return x_clean, others
