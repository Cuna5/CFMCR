"""
Flow Matching denoiser scaling.

Re-exports all original denoiser_scaling classes and adds:
  - FlowMatchingScaling: velocity-prediction preconditioning for FM.

In Flow Matching the network directly predicts the velocity field
    v_θ(x_t, t, μ)  ≈  x_clean - μ
rather than the clean image itself.  Therefore:
  c_skip = 0   (no residual / skip connection)
  c_out  = 1   (raw network output is the velocity)
  c_in   = 1   (no input rescaling)
  c_noise = 0.25·log(t) = 0.25·log(1 - σ)   (log-scale time embedding,
                                               same convention as EDM's log σ)
"""

from .denoiser_scaling import *  # re-export all existing scaling classes

from typing import Tuple
import torch
from .denoiser_scaling import DenoiserScaling


class FlowMatchingScaling(DenoiserScaling):
    """
    Velocity-prediction preconditioning for Optimal Transport Flow Matching.

    Compatible with ResidualDenoiser (same calling convention as
    ResidualEDMScaling):
        c_skip, c_out, c_in, c_noise = scaling(sigma_bc, st_bc)

    Here sigma_bc = σ (shape B×1×1×1, σ = 1-t ∈ [0,1]) and st_bc is ignored
    (kept for interface compatibility).

    c_noise uses log(t) = log(1 - σ) for smooth time conditioning
    (analogous to EDM's 0.25·log(σ)).
    """

    def __call__(
        self,
        sigma: torch.Tensor,          # (B, 1, 1, 1),  σ = 1-t ∈ [0,1]
        st: torch.Tensor = None,      # ignored, kept for interface compat.
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip  = torch.zeros_like(sigma)                          # 0
        c_out   = torch.ones_like(sigma)                           # 1
        c_in    = torch.ones_like(sigma)                           # 1

        # t = 1 - σ; log-scale conditioning clipped away from zero
        t = (1.0 - sigma).clamp(min=1e-4)
        c_noise = 0.25 * torch.log(t)                             # (B,1,1,1)

        return c_skip, c_out, c_in, c_noise
