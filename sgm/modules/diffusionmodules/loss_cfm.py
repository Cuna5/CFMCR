"""
Consistency Flow Matching loss.

Re-exports all FM loss classes and adds:
  - ConsistencyFlowMatchingLoss: endpoint and velocity consistency loss on
    OT straight paths.

Mathematical background
-----------------------
For two consecutive FM time steps  t_n < t_{n+1}  (σ_n > σ_{n+1}):

  x_{t_n}    = (1 − t_n)·μ  +  t_n·x_clean       ← exact OT path, no noise
  x_{t_{n+1}} = (1 − t_{n+1})·μ + t_{n+1}·x_clean  ← exact OT path, no noise

Because the OT path is deterministic and x_clean is known during training,
both interpolation points can be computed directly — NO ODE simulation is
needed, and the teacher only requires ONE forward pass.

Consistency-FM uses an endpoint map:

    f_θ(t, x_t, μ) = x_t + (1 − t) · v_θ(x_t, t, μ)

and constrains both endpoint predictions and velocity fields:

    L_end = ‖ f_θ(t_n, x_{t_n}, μ) − f_{θ⁻}(t_{n+1}, x_{t_{n+1}}, μ) ‖²
    L_vel = ‖ v_θ(x_{t_n}, t_n, μ) − v_{θ⁻}(x_{t_{n+1}}, t_{n+1}, μ) ‖²

The supervised FM anchor keeps that self-consistent velocity tied to the true
OT direction:

    L_FM = ‖ v_θ(x_{t_n}, t_n, μ) − (x_clean − μ) ‖²

where:
  • v_θ       is the student (receives gradients)
  • v_{θ⁻}    is the EMA teacher (stop-gradient, provided via teacher_fn)

At inference a **single** network call from x = μ (t ≈ 0) yields
x_clean ≈ μ + v_θ(μ, t≈0, μ) · 1.

Differences vs. ConsistencyResidualDiffusionLoss
------------------------------------------------
* Path: straight OT line  (not EMRDM mean-reverting noisy path)
* Target: velocity v (not x_clean)
* x_{t_{n+1}}: computed from OT formula  (not from an Euler ODE step)
* Teacher: one call at x_{t_{n+1}}  (not two calls around the Euler step)

Compatible with ConsistencyFlowMatchingEngine (forward() receives teacher_fn
as second positional argument, same pattern as ConsistencyResidualDiffusionLoss).
"""

from .loss_fm import *  # re-export FlowMatchingResidualDiffusionLoss, …

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn

from ...util import append_dims, instantiate_from_config
from .denoiser import Denoiser
from ...modules.encoders.modules import GeneralConditioner
from .sigma2st import Sigma2St


class ConsistencyFlowMatchingLoss(nn.Module):
    """
    Endpoint and velocity consistency loss for Consistency Flow Matching.

    Both training points (x_{t_n} and x_{t_{n+1}}) are constructed from the
    closed-form OT interpolation — no stochastic Euler integration is needed.
    The teacher is called once at the *next* time step; the student is called
    once at the *current* time step.

    Args:
        discretization_config: Config for EDMDiscretization.  Use
            sigma_min=0.001, sigma_max=0.999, rho=1 for a linear schedule
            that matches uniform-t FM training.
        loss_type: "l2" (default) or "l1".
        num_steps:  Number of discretisation steps used to build the σ
            schedule from which consecutive pairs (σ_n, σ_{n+1}) are sampled.
        endpoint_loss_weight: Weight for the endpoint consistency term
            ||f_θ(t, x_t) - f_{θ⁻}(t+Δt, x_{t+Δt})||².
        fm_loss_weight: Weight for the supervised FM anchor term. Keep this
            greater than 0 when training from scratch; consistency alone can
            learn a self-consistent but wrong velocity field.
        consistency_loss_weight: Weight for the teacher-student velocity
            consistency term.
        batch2model_keys: Keys forwarded from batch to the network.
    """

    def __init__(
        self,
        discretization_config: dict,
        loss_type: str = "l2",
        num_steps: int = 18,
        endpoint_loss_weight: float = 1.0,
        fm_loss_weight: float = 1.0,
        consistency_loss_weight: float = 1.0,
        consistency_warmup_steps: int = 0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__()
        assert loss_type in ["l2", "l1"], f"Unsupported loss_type: {loss_type}"

        self.discretization = instantiate_from_config(discretization_config)
        self.loss_type = loss_type
        self.num_steps = num_steps
        self.endpoint_loss_weight = endpoint_loss_weight
        self.fm_loss_weight = fm_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        # While training from scratch the teacher is random for the first
        # few thousand steps. Linearly ramp the endpoint/velocity consistency
        # terms up from 0 so those steps are guided only by the supervised
        # FM anchor (v* = x_clean − μ). See Yang et al. 2024, Sec. 4.2 —
        # "two-stage training".
        self.consistency_warmup_steps = max(int(consistency_warmup_steps), 0)

        if not batch2model_keys:
            batch2model_keys = []
        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]
        self.batch2model_keys = set(batch2model_keys)

    def _consistency_ramp(self, batch: Dict) -> float:
        """Linear warmup multiplier for the consistency / endpoint losses."""
        if self.consistency_warmup_steps == 0:
            return 1.0
        step = int(batch.get("global_step", 0))
        return min(1.0, step / float(self.consistency_warmup_steps))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        network: nn.Module,
        teacher_fn,                    # callable: (x, σ, st, cond, **kw) → v_target
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        sigma2st: Sigma2St,            # FlowMatchingSigma2St expected
        input: torch.Tensor,           # x_clean  [B, C, H, W]
        mu: torch.Tensor,              # cloudy   [B, C, H, W]
        batch: Dict,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        return self._forward(
            network, teacher_fn, denoiser, cond, sigma2st, input, mu, batch
        )

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _forward(
        self,
        network: nn.Module,
        teacher_fn,
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

        # ── 1. Sample a consecutive pair (σ_n, σ_{n+1}) ───────────────────
        # discretization returns [σ_max, …, σ_min]  (num_steps values, descending)
        sigmas = self.discretization(self.num_steps, device=device)
        valid_len = len(sigmas) - 1          # exclude last entry (σ_min itself)
        indices   = torch.randint(0, valid_len, (B,), device=device)
        sigma_n   = sigmas[indices]           # σ_n  (larger = more cloudy, t_n small)
        sigma_n1  = sigmas[indices + 1]       # σ_{n+1} (smaller = more clean, t_{n+1} large)

        # t = 1 − σ  (FM time)
        t_n   = 1.0 - sigma_n
        t_n1  = 1.0 - sigma_n1

        # Time-embedding values via sigma2st  (= t for FlowMatchingSigma2St)
        st_n  = sigma2st(sigma_n)
        st_n1 = sigma2st(sigma_n1)

        t_n_bc  = append_dims(t_n,  input.ndim)
        t_n1_bc = append_dims(t_n1, input.ndim)
        sigma_n_bc = append_dims(sigma_n, input.ndim)
        sigma_n1_bc = append_dims(sigma_n1, input.ndim)

        # ── 2. Construct training points on the OT path (no extra noise) ──
        # x_{t_n}    = (1 − t_n)·μ  + t_n·x_clean   = σ_n·μ + (1−σ_n)·x_clean
        # x_{t_{n+1}} = (1 − t_{n+1})·μ + t_{n+1}·x_clean = σ_{n+1}·μ + (1−σ_{n+1})·x_clean
        x_tn  = (1.0 - t_n_bc)  * mu + t_n_bc  * input
        x_tn1 = (1.0 - t_n1_bc) * mu + t_n1_bc * input

        # ── 3. Teacher velocity at x_{t_{n+1}}  (one call, no grad) ───────
        # teacher_fn is provided by ConsistencyFlowMatchingEngine and wraps
        # the EMA teacher denoiser under torch.no_grad().
        v_target = teacher_fn(
            x_tn1, sigma_n1, st_n1, cond, **additional_model_inputs
        )

        # ── 4. Student velocity at x_{t_n}  (receives gradients) ──────────
        v_student = denoiser(
            network, x_tn, sigma_n, cond, st_n, **additional_model_inputs
        )

        # ── 5. Consistency-FM endpoint map f(t, x) = x + (1 - t) v(t, x) ──
        # Because σ = 1 - t, the remaining time to the endpoint is σ.
        f_student = x_tn + sigma_n_bc * v_student
        f_target = x_tn1 + sigma_n1_bc * v_target

        # ── 6. Supervised FM anchor: true OT velocity ──────────────────────
        # Consistency alone only makes the velocity constant along a
        # trajectory; this term pins that constant to the correct direction.
        velocity_target = input - mu

        # ── 7. Combined CFM loss ───────────────────────────────────────────
        endpoint_loss = self._get_loss(f_student, f_target.detach())
        velocity_consistency_loss = self._get_loss(v_student, v_target)
        fm_anchor_loss = self._get_loss(v_student, velocity_target)

        # Warmup: disable the consistency / endpoint terms during the first
        # `consistency_warmup_steps` gradient updates so the student+teacher
        # first learn a correct velocity field from the supervised anchor.
        ramp = self._consistency_ramp(batch)

        return (
            ramp * self.endpoint_loss_weight * endpoint_loss
            + ramp * self.consistency_loss_weight * velocity_consistency_loss
            + self.fm_loss_weight * fm_anchor_loss
        )

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
