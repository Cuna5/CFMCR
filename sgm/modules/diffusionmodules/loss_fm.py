"""
Flow Matching loss functions.

Re-exports all original loss classes and adds:
  - FlowMatchingResidualDiffusionLoss: Conditional Flow Matching loss for
    cloud removal, based on the Optimal Transport (OT) linear path.

Mathematical background
-----------------------
Given clean image x₀ and cloudy mean μ, define the conditional OT path:

    x_t = (1 - t)·μ + t·x₀,   t ∈ [0, 1]

The conditional vector field (velocity) along this path is constant:

    u_t(x_t | x₀) = x₀ - μ

Training objective (Conditional Flow Matching):

    L_CFM = E_{t,x₀,μ} [ ‖v_θ(x_t, t, μ) - (x₀ - μ)‖² ]

where t ~ Uniform[t_min, t_max].

In the σ-based EMRDM framework we set σ = 1 - t, so:
  - σ_max ≈ 1 (t ≈ 0, cloudy)  →  σ_min ≈ 0 (t ≈ 1, clean)
  - x_σ = σ·μ + (1 - σ)·x₀     (pure linear interpolation, no extra noise)
  - velocity target: v* = x₀ - μ (independent of σ)

The network uses FlowMatchingScaling (c_skip=0, c_out=1, c_in=1) so its raw
output IS the predicted velocity; no further postprocessing is needed.
"""

from .loss import *  # re-export StandardDiffusionLoss, ResidualDiffusionLoss, ...

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from ...util import append_dims, instantiate_from_config
from .denoiser import Denoiser
from ...modules.encoders.modules import GeneralConditioner
from .sigma2st import Sigma2St


class FlowMatchingResidualDiffusionLoss(nn.Module):
    """
    Conditional Flow Matching loss for mean-reverting (cloud removal) diffusion.

    Key differences from ResidualDiffusionLoss
    -------------------------------------------
    * Uses a DIRECT linear OT interpolation x_t = (1-t)·μ + t·x_clean
      instead of the EDM-style noised input.
    * The network predicts **velocity** v = x_clean - μ (not x_clean itself).
    * Requires the denoiser to use FlowMatchingScaling (c_skip=0, c_out=1).
    * No sigma sampler needed; time t is sampled uniformly from [t_min, t_max].

    Compatible with ResidualDiffusionEngine (same forward() signature as
    ResidualDiffusionLoss).
    """

    def __init__(
        self,
        t_min: float = 0.001,
        t_max: float = 0.999,
        loss_type: str = "l2",
        batch2model_keys: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__()
        assert loss_type in ["l2", "l1"], f"Unsupported loss_type: {loss_type}"
        self.t_min = t_min
        self.t_max = t_max
        self.loss_type = loss_type

        if not batch2model_keys:
            batch2model_keys = []
        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]
        self.batch2model_keys = set(batch2model_keys)

    # ------------------------------------------------------------------
    # Public interface (matches ResidualDiffusionLoss.forward signature)
    # ------------------------------------------------------------------

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        sigma2st: Sigma2St,        # FlowMatchingSigma2St expected (s = 1-σ = t)
        input: torch.Tensor,       # x_clean  [B, C, H, W]
        mu: torch.Tensor,          # cloudy   [B, C, H, W]
        batch: Dict,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, sigma2st, input, mu, batch)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
    ) -> torch.Tensor:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        B      = input.shape[0]
        device = input.device

        # ── 1. Sample t ~ Uniform[t_min, t_max] ────────────────────────────
        t = (
            torch.rand(B, device=device) * (self.t_max - self.t_min) + self.t_min
        )  # t ∈ [t_min, t_max]

        # ── 2. Map to σ = 1-t (EMRDM sigma convention) ─────────────────────
        sigma = 1.0 - t            # σ ∈ [1-t_max, 1-t_min]
        st    = sigma2st(sigma)    # FlowMatchingSigma2St: st = 1 - σ = t  ✓

        sigma_bc = append_dims(sigma, input.ndim)
        t_bc     = append_dims(t,     input.ndim)

        # ── 3. OT interpolation (no extra noise — straight-line path) ───────
        #   x_t = (1-t)·μ + t·x_clean
        x_t = (1.0 - t_bc) * mu + t_bc * input

        # ── 4. Velocity target: v* = x_clean - μ ───────────────────────────
        velocity_target = input - mu

        # ── 5. Network prediction (FlowMatchingScaling → raw velocity out) ─
        #   denoiser output = network(x_t * c_in) * c_out + x_t * c_skip
        #                   = network(x_t)   (c_in=1, c_out=1, c_skip=0)
        v_pred = denoiser(
            network, x_t, sigma, cond, st, **additional_model_inputs
        )

        # ── 6. CFM loss ─────────────────────────────────────────────────────
        return self._get_loss(v_pred, velocity_target)

    def _get_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.loss_type == "l2":
            return torch.mean(
                ((pred - target) ** 2).reshape(pred.shape[0], -1), dim=1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (pred - target).abs().reshape(pred.shape[0], -1), dim=1
            )
        else:
            raise NotImplementedError(f"Unknown loss_type: {self.loss_type}")
