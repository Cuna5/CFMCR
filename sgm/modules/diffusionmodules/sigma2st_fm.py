"""
Flow Matching sigma2st variants.

Re-exports all original sigma2st classes (drop-in compatible) and adds:
  - FlowMatchingSigma2St: linear s(σ) = 1 - σ for Optimal Transport FM.

Convention used throughout FM modules
--------------------------------------
  σ ∈ [0, 1]  plays the role of (1 - t), where t ∈ [0, 1] is FM time.
  σ = 1  →  t = 0  (cloudy image, start of generation)
  σ = 0  →  t = 1  (clean image, end of generation)

  s(σ) = 1 - σ = t

The sampler runs σ from σ_max ≈ 1 DOWN to σ_min ≈ 0, which corresponds
to integrating the ODE from t ≈ 0 (cloudy) to t ≈ 1 (clean).
"""

from .sigma2st import *  # re-export EDMSigma2St, PSigma2St, NaiveSigma2St, ...

import torch
from .sigma2st import Sigma2St


class FlowMatchingSigma2St(Sigma2St):
    """
    Linear sigma-to-s(t) mapping for Optimal Transport Flow Matching.

    s(σ) = 1 - σ        (so s ≡ t, where t = 1 - σ)
    ds/dσ = -1

    Forward process used in FlowMatchingResidualDiffusionLoss:
        x_t = (1 - t)·μ + t·x_clean
            = σ·μ + (1 - σ)·x_clean        (pure OT interpolation, no extra noise)

    In EMRDM loss notation the noised input is constructed directly via the
    FM interpolation formula — NOT via the (1-s)/s·μ shift — so the only role
    of s(σ) here is to provide the time embedding fed to the network.

    Recommended discretizer: EDMDiscretization(sigma_min=0.001,
                                                sigma_max=0.999, rho=1)
    which gives a LINEAR σ schedule matching the uniform-t FM training.
    """

    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1.0 - sigma          # s = t

    def get_derivative_st(self):
        return lambda sigma: -torch.ones_like(sigma)   # ds/dσ = -1
