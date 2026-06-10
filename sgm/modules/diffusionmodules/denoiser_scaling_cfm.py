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


class MeanFlowScaling(DenoiserScaling):
    """Average-velocity preconditioning for MeanFlow.

    Same raw-velocity output as ConsistencyFlowMatchingScaling, but the time
    embedding uses the *smooth* form log(t + eps) instead of log(clamp(t)).
    The MeanFlow identity differentiates the network w.r.t. the current time
    via JVP; clamp() would zero that derivative near t = 0, exactly where the
    one-step pair (s, T) = (0, 1) is trained. For t >> eps both forms agree,
    and at t = 0 they coincide, so CFM checkpoints still warm-start cleanly.
    """

    def __init__(self, time_eps: float = 1e-4):
        self.time_eps = float(time_eps)

    def __call__(
        self,
        sigma: torch.Tensor,
        st: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = torch.zeros_like(sigma)
        c_out = torch.ones_like(sigma)
        c_in = torch.ones_like(sigma)

        t = 1.0 - sigma
        c_noise = 0.25 * torch.log(t + self.time_eps)

        return c_skip, c_out, c_in, c_noise
