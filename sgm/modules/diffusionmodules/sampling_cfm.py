"""
Consistency Flow Matching samplers.

Re-exports everything from sampling_fm and adds:
  - ConsistencyFlowMatchingSampler: supports both 1-step and multi-step
    inference for a trained CFM consistency model.

1-step inference
----------------
The CFM consistency model satisfies the self-consistency property over
velocity fields.  A single call from x = μ (t ≈ 0, σ ≈ 1) suffices:

    v = v_θ(μ, σ_max, cond)            # network predicts x_clean − μ
    x_clean = μ + v · Δt               # Δt = 1 − t_init = σ_max ≈ 1

This is a single Euler step spanning the full time range [t_init, 1].

Multi-step inference
--------------------
When num_steps > 1 the sampler falls back to the standard FM Euler ODE
loop inherited from FlowMatchingResidualSampler (identical behaviour to
the plain FM Direction 1 sampler, but using CFM-trained weights).
"""

from .sampling_fm import *      # re-export FlowMatchingResidualSampler, …
from .sampling_fm import FlowMatchingResidualSampler

import torch
from ...util import append_dims, default, tools_scale


class ConsistencyFlowMatchingSampler(FlowMatchingResidualSampler):
    """
    Sampler for a Consistency Flow Matching cloud-removal model.

    Supports two inference modes (controlled by num_steps in config or at
    call time):

    * **1-step** (num_steps = 1):
        x_init = μ  (start from the cloudy image)
        v = v_θ(μ, σ_max, cond)
        x_clean = μ + v · σ_max           (single Euler step over full range)

    * **multi-step** (num_steps > 1):
        Identical to FlowMatchingResidualSampler — Euler ODE from σ_max
        down to σ_min in num_steps uniform steps.

    Calling convention is identical to FlowMatchingResidualSampler so the
    sampler can be swapped without engine changes.
    """

    def __call__(
        self,
        denoiser,
        x,              # randn from engine.sample() — ignored, we init from μ
        mu,             # cloudy image  [B, C, H, W]
        cond,
        uc=None,
        num_steps=None,
        return_intermediate: bool = False,
        return_denoised: bool = False,
    ):
        n = self.num_steps if num_steps is None else num_steps
        if n == 1:
            return self._one_step(
                denoiser, x, mu, cond, uc,
                return_intermediate, return_denoised,
            )
        # Multi-step: delegate to FM Euler loop
        return super().__call__(
            denoiser, x, mu, cond, uc, num_steps,
            return_intermediate, return_denoised,
        )

    # ------------------------------------------------------------------
    # True 1-step inference
    # ------------------------------------------------------------------

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
        """
        Single-step CFM inference.

        Physics:
            Euler step from t_init = 1 − σ_max  to  t = 1  (full range):
                x_clean = x_init + v · (1 − t_init)
                        = μ + v · σ_max
        """
        # Use at least 2 discretization points to safely extract σ_max
        n_disc = max(self.num_steps, 2) if self.num_steps is not None else 4
        sigmas    = self.discretization(n_disc, device=mu.device)
        sigma_max = sigmas[0]

        uc    = default(uc, cond)
        s_in  = mu.new_ones([mu.shape[0]])
        sigma = s_in * sigma_max

        # FM initialisation: start from the cloudy image (t ≈ 0)
        x_init = mu.clone()

        # s(σ_max) = 1 − σ_max = t_init  (very small, ≈ 0.001)
        st = self.sigma2st(sigma)

        # Velocity prediction: v_θ(μ, σ_max, cond) ≈ x_clean − μ
        v_pred = self._denoise(x_init, denoiser, sigma, cond, st, uc)

        # 1-step Euler: x_clean = μ + v · σ_max  (Δt = σ_max ≈ 1)
        sigma_max_bc = append_dims(sigma_max, x_init.ndim)
        x_clean = x_init + v_pred * sigma_max_bc

        others = {}
        if return_intermediate:
            others["intermediates"] = [tools_scale(x_init.clone().detach())]
        if return_denoised:
            # x_clean estimate via μ + v  (full-step interpretation)
            others["denoiseds"] = [
                tools_scale((mu + v_pred).clone().detach())
            ]

        return x_clean, others
