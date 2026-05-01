"""
Consistency Flow Matching sigma2st variants.

Re-exports everything from sigma2st_fm (which in turn re-exports from
sigma2st) and adds a named alias so that YAML configs can reference a
CFM-specific class name, making the active training direction unambiguous.

Convention (shared with FM)
----------------------------
  σ = 1 - t,   σ ∈ [0, 1]
  σ_max ≈ 1  →  t ≈ 0  (cloudy image, generation start)
  σ_min ≈ 0  →  t ≈ 1  (clean image, generation end)
  s(σ) = 1 - σ = t
"""

from .sigma2st_fm import *          # re-export FlowMatchingSigma2St, EDMSigma2St, …
from .sigma2st_fm import FlowMatchingSigma2St


class ConsistencyFlowMatchingSigma2St(FlowMatchingSigma2St):
    """
    sigma2st for Consistency Flow Matching (Direction 3).

    Identical to FlowMatchingSigma2St:
        s(σ) = 1 − σ = t,   ds/dσ = −1

    Provided as a distinct class so that CFM YAML configs can declare
        target: sgm.modules.diffusionmodules.sigma2st_cfm.ConsistencyFlowMatchingSigma2St
    making it immediately clear which training direction is active.
    """
    pass
