"""
Denoiser scaling for Consistency Flow Matching.

The CFM network predicts the OT velocity directly:

    v_theta(x_t, t, mu) ~= x_clean - mu

so the denoiser wrapper should return the raw network output as the velocity.
"""

from typing import Tuple

import torch

from .denoiser_scaling import DenoiserScaling


class ConsistencyFlowMatchingScaling(DenoiserScaling):
    """Velocity-prediction preconditioning for CFM."""

    def __call__(
        self,
        sigma: torch.Tensor,
        st: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = torch.zeros_like(sigma)
        c_out = torch.ones_like(sigma)
        c_in = torch.ones_like(sigma)

        # t = 1 - sigma. Clamp away from zero for stable log-time embedding.
        t = (1.0 - sigma).clamp(min=1e-4)
        c_noise = 0.25 * torch.log(t)

        return c_skip, c_out, c_in, c_noise
