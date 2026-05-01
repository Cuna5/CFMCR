"""
Consistency Flow Matching denoiser scaling.

Re-exports from denoiser_scaling_fm and adds a named alias for CFM so that
YAML configs can reference a CFM-specific class name.

The velocity-prediction preconditioning is identical to FlowMatchingScaling:
  c_skip = 0     (no residual skip — raw network output IS the velocity)
  c_out  = 1
  c_in   = 1
  c_noise = 0.25 · log(t),   t = 1 − σ
"""

from .denoiser_scaling_fm import *      # re-export FlowMatchingScaling, …
from .denoiser_scaling_fm import FlowMatchingScaling


class ConsistencyFlowMatchingScaling(FlowMatchingScaling):
    """
    Velocity-prediction preconditioning for CFM (Direction 3).

    Identical to FlowMatchingScaling (c_skip=0, c_out=1, c_in=1).
    Provided under a distinct name so that CFM configs can declare
        target: sgm.modules.diffusionmodules.denoiser_scaling_cfm.ConsistencyFlowMatchingScaling
    """
    pass
