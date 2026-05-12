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
    x_clean = μ + v                    # full t=0 → t=1 consistency jump

This is the single full-range consistency jump used for 1-step inference.

Multi-step inference
--------------------
When num_steps > 1 the sampler falls back to the standard FM Euler ODE
loop inherited from FlowMatchingResidualSampler (identical behaviour to
the plain FM Direction 1 sampler, but using CFM-trained weights).
"""

from .sampling_fm import *      # re-export FlowMatchingResidualSampler, …
from .sampling_fm import FlowMatchingResidualSampler

from ...util import append_dims, default, tools_scale


class ConsistencyFlowMatchingSampler(FlowMatchingResidualSampler):
    """
    Sampler for a Consistency Flow Matching cloud-removal model.

    Supports two inference modes (controlled by num_steps in config or at
    call time):

    * **1-step** (num_steps = 1):
        x_init = μ  (start from the cloudy image)
        v = v_θ(μ, σ_max, cond)
        x_clean = μ + v                  (single full-range consistency step)

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
            Use the same endpoint map the loss was trained on:
                f(σ, x) = x + σ · v(x, σ)   (σ = 1 − t)
            Starting from x_init = μ (t ≈ 0, σ ≈ σ_max) the prediction is:
                x_clean = μ + σ_max · v_θ(μ, σ_max, cond)
            This matches the training-time endpoint and keeps the network's
            time conditioning and the applied jump consistent.
        """
        # Read σ_max directly from the discretizer config to avoid depending
        # on `num_steps >= 2` just to index sigmas[0].
        sigma_max_scalar = float(getattr(self.discretization, "sigma_max", 1.0))
        sigma_max = mu.new_tensor(sigma_max_scalar)

        uc = default(uc, cond)
        s_in = mu.new_ones([mu.shape[0]])
        sigma = s_in * sigma_max

        # FM initialisation: start from the cloudy image (t ≈ 0)
        x_init = mu.clone()

        # s(σ_max) = 1 − σ_max = t_init  (very small, ≈ 0.001)
        st = self.sigma2st(sigma)

        # Velocity prediction.  With the endpoint map  f = x + σ·v  a true
        # 1-step prediction at σ = σ_max becomes  x_clean = μ + σ_max · v.
        v_pred = self._denoise(x_init, denoiser, sigma, cond, st, uc)

        sigma_bc = append_dims(sigma, x_init.ndim)
        x_clean = x_init + sigma_bc * v_pred

        others = {}
        if return_intermediate:
            others["intermediates"] = [tools_scale(x_init.clone().detach())]
        if return_denoised:
            others["denoiseds"] = [tools_scale(x_clean.clone().detach())]

        return x_clean, others
