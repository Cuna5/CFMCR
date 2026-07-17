"""
Consistency Flow Matching sampler.

The default path is true 1-step inference:

    x_init = mu
    v = v_theta(mu, sigma_max, cond)
    x_clean = mu + sigma_max * v

When ``num_steps > 1`` the same trained CFM velocity field is integrated with
Euler steps in sigma space, which provides a quality/speed tradeoff without
requiring a separate sampler module.

Test-Time Augmentation (TTA)
----------------------------
When ``tta=True``, four geometric variants are evaluated and averaged:

    1. original
    2. horizontal flip
    3. vertical flip
    4. horizontal + vertical flip (180° rotation)

Each variant is inverse-transformed before averaging, yielding more robust
predictions at the cost of 4× inference time.  Enable via YAML:

    sampler_config.params.tta: True
"""

from typing import Dict, List, Union

import torch
from omegaconf import ListConfig

from ...util import append_dims, default, instantiate_from_config, tools_scale


# ---------------------------------------------------------------------------
# TTA geometric transforms (operate on BCHW tensors)
# ---------------------------------------------------------------------------

def _hflip(x: torch.Tensor) -> torch.Tensor:
    """Horizontal flip (left-right)."""
    return x.flip(-1)


def _vflip(x: torch.Tensor) -> torch.Tensor:
    """Vertical flip (top-bottom)."""
    return x.flip(-2)


def _hvflip(x: torch.Tensor) -> torch.Tensor:
    """Horizontal + vertical flip (equivalent to 180° rotation)."""
    return x.flip(-1).flip(-2)


# Each entry: (forward_transform, inverse_transform)
_TTA_TRANSFORMS: List = [
    (lambda x: x,      lambda x: x),       # identity
    (_hflip,            _hflip),            # hflip is self-inverse
    (_vflip,            _vflip),            # vflip is self-inverse
    (_hvflip,           _hvflip),           # hvflip is self-inverse
]


class ConsistencyFlowMatchingSampler:
    """Sampler for CFM cloud removal models.

    Args:
        discretization_config: Discretization schedule config.
        num_steps: Number of sampling steps (1 = true one-step inference).
        guider_config: Optional classifier-free guidance config.
        tta: Enable Test-Time Augmentation (4 geometric variants averaged).
        verbose: Print progress bar during multi-step sampling.
        device: Device string.
    """

    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, None] = None,
        tta: bool = False,
        verbose: bool = False,
        device: str = "cuda",
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        clamp_output: bool = False,
        mask_composite: bool = False,
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)

        default_guider = {
            "target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"
        }
        self.guider = instantiate_from_config(default(guider_config, default_guider))
        self.tta = tta
        self.verbose = verbose
        self.device = device
        self.sigma2st = None
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        # clamp_output: clamp the final prediction to [-1, 1] so a few outlier
        # pixels cannot drag down RMSE / PSNR.
        self.clamp_output = clamp_output
        # mask_composite: blend the prediction with the cloudy input using the
        # network's predicted cloud probability (requires
        # network_config.params.predict_cloud_mask=true):
        #     x_final = (1 - m) * mu + m * x_pred
        # Clear pixels are restored verbatim from the input. Not compatible
        # with CFG guiders that duplicate the batch.
        self.mask_composite = mask_composite
        self._network_ref = None

    def set_sigma2st(self, sigma2st):
        self.sigma2st = sigma2st

    def set_network_ref(self, model):
        """Reference to the (wrapped) network so the sampler can read the
        cloud-probability logits exposed by the last forward pass."""
        self._network_ref = model

    def _get_mask_prob(self):
        if not self.mask_composite or self._network_ref is None:
            return None
        net = getattr(self._network_ref, "diffusion_model", self._network_ref)
        logits = getattr(net, "last_mask_logits", None)
        if logits is None:
            return None
        return torch.sigmoid(logits.float())

    def _finalize(self, x_clean, mu):
        """Apply optional output clamping and cloud-mask composition."""
        if self.clamp_output:
            x_clean = x_clean.clamp(-1.0, 1.0)
        m = self._get_mask_prob()
        if m is not None:
            m = m.to(dtype=x_clean.dtype)
            x_clean = (1.0 - m) * mu + m * x_clean
        return x_clean

    def _get_sigma_gen(self, num_sigmas):
        from tqdm import tqdm

        gen = range(num_sigmas - 1)
        if self.verbose:
            gen = tqdm(gen, total=num_sigmas, desc=f"CFM Sampling ({num_sigmas - 1} steps)")
        return gen

    def _denoise(self, x, denoiser, sigma, cond, st, uc, **extra):
        prepared = self.guider.prepare_inputs(x, sigma, cond, uc)
        model_batch = prepared[0].shape[0]
        extra = dict(extra)
        for key, value in extra.items():
            if (
                torch.is_tensor(value)
                and value.ndim > 0
                and value.shape[0] != model_batch
            ):
                if model_batch % value.shape[0] != 0:
                    raise ValueError(
                        f"{key} batch size {value.shape[0]} cannot be aligned "
                        f"with guided model batch size {model_batch}."
                    )
                extra[key] = torch.cat(
                    [value] * (model_batch // value.shape[0]),
                    dim=0,
                )
        velocity = denoiser(*prepared, st=st, **extra)
        return self.guider(velocity, sigma)

    def _prepare_loop(self, x_randn, mu, cond, uc, num_steps):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps,
            device=self.device,
        )
        uc = default(uc, cond)
        x_init = mu.clone()
        s_in = x_init.new_ones([x_init.shape[0]])
        return x_init, s_in, sigmas, len(sigmas), cond, uc

    def _sampler_step(self, sigma, next_sigma, denoiser, x, mu, cond, uc):
        st = self.sigma2st(sigma)
        v_pred = self._denoise(x, denoiser, sigma, cond, st, uc)
        sigma_bc = append_dims(sigma, x.ndim)
        endpoint = x + sigma_bc * v_pred
        dt = append_dims(next_sigma - sigma, x.ndim)
        x_next = x - v_pred * dt
        return x_next, endpoint

    # ------------------------------------------------------------------
    # TTA helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_transform_to_cond(cond, transform_fn):
        """Apply a spatial transform to all 4-D tensors in a cond dict.

        Conditioner outputs that carry spatial information (e.g. the
        concatenated cloudy image) must be transformed in sync with mu.
        Scalar / 1-D / 2-D embeddings are left untouched.
        """
        transformed = {}
        for k, v in cond.items():
            if isinstance(v, torch.Tensor) and v.ndim == 4:
                transformed[k] = transform_fn(v)
            else:
                transformed[k] = v
        return transformed

    def _sample_single(self, denoiser, x, mu, cond, uc, num_steps):
        """Run core sampling (1-step or multi-step) without TTA, returning
        only x_clean (no intermediates/denoiseds — those are not meaningful
        when averaged across transforms)."""
        n = self.num_steps if num_steps is None else num_steps
        if n == 1:
            x_clean, _ = self._one_step(denoiser, x, mu, cond, uc)
        else:
            x_clean, _ = self._multi_step(denoiser, x, mu, cond, uc, num_steps)
        return x_clean

    def _sample_with_tta(self, denoiser, x, mu, cond, uc, num_steps):
        """Run inference with 4 geometric augmentations and average."""
        accumulator = torch.zeros_like(mu)

        for fwd_fn, inv_fn in _TTA_TRANSFORMS:
            mu_t = fwd_fn(mu)
            cond_t = self._apply_transform_to_cond(cond, fwd_fn)
            uc_t = self._apply_transform_to_cond(uc, fwd_fn) if uc is not None else None

            x_clean_t = self._sample_single(denoiser, x, mu_t, cond_t, uc_t, num_steps)
            accumulator += inv_fn(x_clean_t)

        x_clean_avg = accumulator / len(_TTA_TRANSFORMS)
        return x_clean_avg

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        denoiser,
        x,
        mu,
        cond,
        uc=None,
        num_steps=None,
        return_intermediate: bool = False,
        return_denoised: bool = False,
    ):
        # TTA path — averages across geometric transforms
        if self.tta:
            x_clean = self._sample_with_tta(denoiser, x, mu, cond, uc, num_steps)
            # TTA does not produce meaningful intermediates/denoiseds
            others = {}
            return x_clean, others

        # Non-TTA path — preserves intermediates/denoiseds
        n = self.num_steps if num_steps is None else num_steps
        if n == 1:
            return self._one_step(
                denoiser, x, mu, cond, uc, return_intermediate, return_denoised
            )
        return self._multi_step(
            denoiser, x, mu, cond, uc, num_steps, return_intermediate, return_denoised
        )

    def _multi_step(
        self,
        denoiser,
        x,
        mu,
        cond,
        uc=None,
        num_steps=None,
        return_intermediate: bool = False,
        return_denoised: bool = False,
    ):
        x, s_in, sigmas, num_sigmas, cond, uc = self._prepare_loop(
            x, mu, cond, uc, num_steps
        )

        intermediates = []
        denoiseds = []

        for i in self._get_sigma_gen(num_sigmas):
            if return_intermediate:
                intermediates.append(tools_scale(x.clone().detach()))

            sigma_i = s_in * sigmas[i]
            # s_churn stochastic perturbation (mirrors EMRDM ResidualEDMSampler)
            if self.s_churn > 0.0 and self.s_tmin <= float(sigmas[i]) <= self.s_tmax:
                gamma = min(self.s_churn / (num_sigmas - 1), 2 ** 0.5 - 1)
                sigma_max = s_in * sigmas[0]
                sigma_hat = torch.minimum(sigma_i * (1.0 + gamma), sigma_max)
                noise = torch.randn_like(x) * self.s_noise
                noise_scale = torch.sqrt((sigma_hat ** 2 - sigma_i ** 2).clamp_min(0.0))
                x = x + append_dims(noise_scale, x.ndim) * noise
                sigma_i = sigma_hat

            x, endpoint = self._sampler_step(
                sigma_i,
                s_in * sigmas[i + 1],
                denoiser,
                x,
                mu,
                cond,
                uc,
            )

            if return_denoised:
                denoiseds.append(tools_scale(endpoint.clone().detach()))

        x = self._finalize(x, mu)

        others = {}
        if return_intermediate:
            others["intermediates"] = intermediates
        if return_denoised:
            others["denoiseds"] = denoiseds

        return x, others

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
        sigma_max_scalar = float(getattr(self.discretization, "sigma_max", 1.0))
        sigma_max = mu.new_tensor(sigma_max_scalar)

        uc = default(uc, cond)
        s_in = mu.new_ones([mu.shape[0]])
        sigma = s_in * sigma_max
        x_init = mu.clone()
        st = self.sigma2st(sigma)

        v_pred = self._denoise(x_init, denoiser, sigma, cond, st, uc)
        sigma_bc = append_dims(sigma, x_init.ndim)
        x_clean = x_init + sigma_bc * v_pred
        x_clean = self._finalize(x_clean, mu)

        others = {}
        if return_intermediate:
            others["intermediates"] = [tools_scale(x_init.clone().detach())]
        if return_denoised:
            others["denoiseds"] = [tools_scale(x_clean.clone().detach())]

        return x_clean, others


class MeanFlowSampler(ConsistencyFlowMatchingSampler):
    """Sampler for MeanFlow average-velocity models.

    The network predicts the average velocity u(x_s, s, T) over the jump
    [s, T] (CFM time, s = 1 - sigma), conditioned on the jump target through
    `timesteps_r`. Sampling is exact per segment by construction:

        x_T = x_s + (T - s) * u(x_s, s, T)

    1-step inference jumps the full path:  x_pred = mu + u(mu, 0, 1).
    Multi-step splits [0, 1] into segments along the sigma schedule.

    Stochastic churn is not supported (average velocities are only defined on
    the deterministic OT path); s_churn is forced to 0.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s_churn = 0.0

    def _denoise_jump(self, x, denoiser, sigma, sigma_next, cond, st, uc):
        from .loss_meanflow import meanflow_c_noise

        T = self.sigma2st(sigma_next)
        u_pred = denoiser(
            *self.guider.prepare_inputs(x, sigma, cond, uc),
            st=st,
            timesteps_r=meanflow_c_noise(T),
        )
        return self.guider(u_pred, sigma)

    def _sampler_step(self, sigma, next_sigma, denoiser, x, mu, cond, uc):
        st = self.sigma2st(sigma)           # current time s
        st_next = self.sigma2st(next_sigma)  # jump target T
        u_pred = self._denoise_jump(x, denoiser, sigma, next_sigma, cond, st, uc)
        gap = append_dims(st_next - st, x.ndim)
        x_next = x + gap * u_pred
        # Endpoint estimate for logging: extrapolate the same average
        # velocity to the clean end (exact when the jump target is T = 1).
        endpoint = x + append_dims(1.0 - st, x.ndim) * u_pred
        return x_next, endpoint

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
        sigma_zero = torch.zeros_like(sigma)  # jump target T = 1 exactly
        x_init = mu.clone()
        st = self.sigma2st(sigma)

        u_pred = self._denoise_jump(x_init, denoiser, sigma, sigma_zero, cond, st, uc)
        gap = append_dims(self.sigma2st(sigma_zero) - st, x_init.ndim)
        x_clean = x_init + gap * u_pred
        x_clean = self._finalize(x_clean, mu)

        others = {}
        if return_intermediate:
            others["intermediates"] = [tools_scale(x_init.clone().detach())]
        if return_denoised:
            others["denoiseds"] = [tools_scale(x_clean.clone().detach())]

        return x_clean, others
