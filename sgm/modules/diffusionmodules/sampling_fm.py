"""
Flow Matching samplers.

Re-exports all original sampler classes and adds:
  - FlowMatchingResidualSampler: Euler ODE sampler for FM cloud removal.

ODE derivation
--------------
FM path:     x_t = (1-t)·μ + t·x_clean,   dx/dt = x_clean - μ = v*
σ = 1 - t  ⟹  dσ = -dt

dx/dσ = dx/dt · dt/dσ = v · (-1) = -v

Euler step (σ decreasing from σ_max→0, i.e. t increasing 0→1):
    x_next = x + (-v_pred) · (σ_next - σ)
           = x + v_pred · (σ - σ_next)       (σ_next < σ, so Δσ < 0)

Initialisation
--------------
At σ = σ_max ≈ 1 (t ≈ 0): x_init = μ  (start from the cloudy image).
The `x` (randn) argument from ResidualDiffusionEngine.sample() is ignored.

Return values
-------------
The sampler returns (x_clean_estimate, others) where `others` may contain:
  - "intermediates": list of intermediate x states (scaled to [0,1])
  - "denoiseds":     list of x_clean estimates at each step  (μ + v_pred)

Both are compatible with the log_images() visualisation pipeline.
"""

from .sampling import *   # re-export all existing samplers / base classes

from typing import Dict, Union
import torch
from omegaconf import ListConfig, OmegaConf

from ...util import append_dims, default, tools_scale


class FlowMatchingResidualSampler:
    """
    Euler ODE sampler for Optimal Transport Flow Matching cloud removal.

    The network (configured with FlowMatchingScaling) predicts the velocity
    field  v_θ(x_t, σ, μ)  ≈  x_clean - μ.

    Calling convention is identical to ResidualEulerEDMSampler:
        sampler(denoiser, x, mu, cond, uc=..., ...)
    so it can be dropped into ResidualDiffusionEngine without any engine
    changes.

    Discretizer recommendation: EDMDiscretization(sigma_min=0.001,
                                                   sigma_max=0.999, rho=1)
    (rho=1 ⟹ uniform σ steps, matching uniform-t FM training).
    """

    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        from ...util import instantiate_from_config

        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)

        DEFAULT_GUIDER = {
            "target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"
        }
        self.guider = instantiate_from_config(
            default(guider_config, DEFAULT_GUIDER)
        )
        self.verbose = verbose
        self.device = device
        self.sigma2st = None      # set by ResidualDiffusionEngine.__init__

    # ------------------------------------------------------------------
    # Required by ResidualDiffusionEngine
    # ------------------------------------------------------------------

    def set_sigma2st(self, sigma2st):
        self.sigma2st = sigma2st

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_sigma_gen(self, num_sigmas):
        from tqdm import tqdm
        gen = range(num_sigmas - 1)
        if self.verbose:
            gen = tqdm(gen, total=num_sigmas,
                       desc=f"FM Sampling ({num_sigmas-1} steps)")
        return gen

    def _denoise(self, x, denoiser, sigma, cond, st, uc):
        """Query the network (via denoiser wrapper) for velocity prediction."""
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), st=st)
        denoised = self.guider(denoised, sigma)
        return denoised

    # ------------------------------------------------------------------
    # Sampling loop
    # ------------------------------------------------------------------

    def _prepare_loop(self, x_randn, mu, cond, uc, num_steps):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps,
            device=self.device,
        )
        uc = default(uc, cond)

        # FM initialisation: start from the cloudy image (t≈0, σ≈1)
        x_init = mu.clone()

        s_in       = x_init.new_ones([x_init.shape[0]])
        num_sigmas = len(sigmas)
        return x_init, s_in, sigmas, num_sigmas, cond, uc

    def _sampler_step(self, sigma, next_sigma, denoiser, x, mu, cond, uc):
        """One Euler step of the FM ODE (in σ convention)."""
        st = self.sigma2st(sigma)           # st = 1-σ = t

        # Network predicts velocity v ≈ x_clean - μ
        v_pred = self._denoise(x, denoiser, sigma, cond, st, uc)

        # dx/dσ = -v  ⟹  x_next = x + (-v) · (σ_next - σ)
        d  = -v_pred
        dt = append_dims(next_sigma - sigma, x.ndim)   # Δσ < 0
        x_next = x + d * dt                            # x += v · |Δσ|

        return x_next, v_pred

    def __call__(
        self,
        denoiser,
        x,           # randn from engine.sample() — ignored, we init from μ
        mu,          # cloudy image  [B, C, H, W]
        cond,
        uc=None,
        num_steps=None,
        return_intermediate: bool = False,
        return_denoised:     bool = False,
    ):
        x, s_in, sigmas, num_sigmas, cond, uc = self._prepare_loop(
            x, mu, cond, uc, num_steps
        )

        intermediates = []
        denoiseds     = []

        for i in self._get_sigma_gen(num_sigmas):
            if return_intermediate:
                intermediates.append(tools_scale(x.clone().detach()))

            x, v_pred = self._sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser, x, mu, cond, uc,
            )

            if return_denoised:
                # x_clean estimate = μ + v_pred
                x_clean_est = mu + v_pred
                denoiseds.append(tools_scale(x_clean_est.clone().detach()))

        others = {}
        if return_intermediate:
            others["intermediates"] = intermediates
        if return_denoised:
            others["denoiseds"] = denoiseds

        return x, others
