"""
Sigma-to-time mapping for Consistency Flow Matching.

CFM uses the straight OT path:

    x_t = (1 - t) * mu + t * x_clean

The EMRDM sampling interface is sigma-based, so CFM sets sigma = 1 - t:

    sigma = 1 -> t = 0  (cloudy input)
    sigma = 0 -> t = 1  (clean target)
    s(sigma) = t = 1 - sigma
"""

import torch

from .sigma2st import Sigma2St


class ConsistencyFlowMatchingSigma2St(Sigma2St):
    """Linear CFM mapping s(sigma) = 1 - sigma."""

    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1.0 - sigma

    def get_derivative_st(self):
        return lambda sigma: -torch.ones_like(sigma)
