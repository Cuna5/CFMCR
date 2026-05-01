"""
Consistency Flow Matching training engine.

Implements Direction 3: Consistency Flow Matching (CFM).

Key idea
--------
CFM combines:
  • The straight-line OT paths of Flow Matching  (x_t = (1-t)·μ + t·x_clean)
  • The velocity self-consistency constraint of Consistency Models

Training from scratch (no two-stage pre-training required):
  1. Build x_{t_n} and x_{t_{n+1}} from the closed-form OT formula.
  2. Teacher (EMA) predicts velocity at x_{t_{n+1}} → v_{target}.
  3. Student predicts velocity at x_{t_n} → v_{student}.
  4. Loss: ‖v_{student} − v_{target}‖²

At inference a single network call from x = μ gives:
    x_clean ≈ μ + v_θ(μ, σ_max, cond) · σ_max

Differences vs. ConsistencyResidualDiffusionEngine
---------------------------------------------------
* Teacher step:  FM path construction (exact OT formula, no ODE simulation).
* Teacher fn:    simpler signature — wraps a single teacher denoiser call.
* No noise is added when building training points.
* Network predicts velocity, not x_clean.

Requires in YAML config:
    model.target: sgm.models.diffusion_cfm.ConsistencyFlowMatchingEngine
    loss_fn_config.target: …loss_cfm.ConsistencyFlowMatchingLoss
    sampler_config.target: …sampling_cfm.ConsistencyFlowMatchingSampler
    denoiser_scaling_config.target: …denoiser_scaling_cfm.ConsistencyFlowMatchingScaling
    sigma_st_config.target: …sigma2st_cfm.ConsistencyFlowMatchingSigma2St
"""

import copy

import torch

from .diffusion import ResidualDiffusionEngine
from ..util import append_dims


class ConsistencyFlowMatchingEngine(ResidualDiffusionEngine):
    """
    Training engine for Consistency Flow Matching cloud removal.

    Inherits all logging, validation, and sampling logic from
    ResidualDiffusionEngine.  Only forward() and the teacher maintenance
    methods are overridden.

    Args:
        teacher_ema_decay: EMA decay rate for the teacher model (default
            0.9999).  Higher values make the teacher update more slowly
            (more stable but less responsive to student improvements).
        *args, **kwargs: Forwarded to ResidualDiffusionEngine.__init__.
    """

    def __init__(
        self,
        teacher_ema_decay: float = 0.9999,
        *args,
        **kwargs,
    ):
        # Force EMA on — model_ema is used for validation inference.
        kwargs["use_ema"] = True
        super().__init__(*args, **kwargs)

        self.teacher_ema_decay = teacher_ema_decay

        # Separate frozen EMA teacher — never receives gradients.
        # Kept in sync with the student via _update_teacher().
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Teacher maintenance
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_teacher(self):
        """EMA update of the teacher from the student weights."""
        d = self.teacher_ema_decay
        for p_t, p_s in zip(
            self.teacher_model.parameters(), self.model.parameters()
        ):
            p_t.data.mul_(d).add_(p_s.data, alpha=1.0 - d)

    def on_train_batch_end(self, *args, **kwargs):
        # Update model_ema (validation) then the CFM teacher.
        super().on_train_batch_end(*args, **kwargs)
        self._update_teacher()

    # ------------------------------------------------------------------
    # Teacher callable
    # ------------------------------------------------------------------

    def _make_teacher_fn(self):
        """
        Returns a lightweight closure that wraps the EMA teacher denoiser.

        The closure queries the *teacher* network at a given (x, σ, st) and
        returns the velocity prediction detached from the student graph.

        Signature consumed by ConsistencyFlowMatchingLoss:
            teacher_fn(x, sigma, st, cond, **extra) -> v_target  (detached)

        Note: x_{t_{n+1}} is constructed directly from the OT formula inside
        ConsistencyFlowMatchingLoss._forward() — the teacher only needs ONE
        forward pass per training step, unlike CD which requires two.
        """
        denoiser = self.denoiser
        teacher  = self.teacher_model

        @torch.no_grad()
        def teacher_fn(x, sigma, st, cond, **extra):
            v = denoiser(teacher, x, sigma, cond, st, **extra)
            return v.detach()

        return teacher_fn

    # ------------------------------------------------------------------
    # Forward / training step
    # ------------------------------------------------------------------

    def forward(self, x, mu, batch):
        """
        Override ResidualDiffusionEngine.forward() to inject teacher_fn as
        the second positional argument expected by ConsistencyFlowMatchingLoss.
        """
        teacher_fn = self._make_teacher_fn()
        loss = self.loss_fn(
            self.model,
            teacher_fn,
            self.denoiser,
            self.conditioner,
            self.sigma2st,
            x,
            mu,
            batch,
        )
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict
