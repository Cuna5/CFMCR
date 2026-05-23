"""
Consistency Flow Matching loss.

Adds:
  - ConsistencyFlowMatchingLoss: endpoint, velocity, and clean-endpoint
    consistency loss on OT straight paths.

Mathematical background
-----------------------
For two consecutive CFM time steps  t_n < t_{n+1}  (σ_n > σ_{n+1}):

  x_{t_n}    = (1 − t_n)·μ  +  t_n·x_clean       ← exact OT path, no noise
  x_{t_{n+1}} = (1 − t_{n+1})·μ + t_{n+1}·x_clean  ← exact OT path, no noise

Because the OT path is deterministic and x_clean is known during training,
both interpolation points can be computed directly — NO ODE simulation is
needed, and the teacher only requires ONE forward pass.

CFM uses an endpoint map:

    f_θ(t, x_t, μ) = x_t + (1 − t) · v_θ(x_t, t, μ)

and constrains both endpoint predictions and velocity fields:

    L_end = ‖ f_θ(t_n, x_{t_n}, μ) − f_{θ⁻}(t_{n+1}, x_{t_{n+1}}, μ) ‖²
    L_vel = ‖ v_θ(x_{t_n}, t_n, μ) − v_{θ⁻}(x_{t_{n+1}}, t_{n+1}, μ) ‖²

The supervised velocity anchor keeps that self-consistent velocity tied to the
true OT direction:

    L_vel_anchor = ‖ v_θ(x_{t_n}, t_n, μ) − (x_clean − μ) ‖²

For better 1-step image metrics, the student endpoint is also supervised
directly against the clean image:

    L_clean_end = ‖ f_θ(t_n, x_{t_n}, μ) − x_clean ‖²

where:
  • v_θ       is the student (receives gradients)
  • v_{θ⁻}    is the EMA teacher (stop-gradient, provided via teacher_fn)

At inference a **single** network call from x = μ (t ≈ 0) yields
x_clean ≈ μ + v_θ(μ, t≈0, μ) · 1.

Compatible with ConsistencyFlowMatchingEngine (forward() receives teacher_fn
as the second positional argument).
"""

from typing import Dict, List, Optional, Union
import torch
import torch.nn.functional as F
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
            sigma_min=0.001, sigma_max=1.0, rho=1 for a linear schedule
            that matches uniform-t CFM training.
        loss_type: "l2" (default), "l1", or "charbonnier".
        num_steps:  Number of discretisation steps used to build the σ
            schedule from which consecutive pairs (σ_n, σ_{n+1}) are sampled.
        endpoint_loss_weight: Weight for the endpoint consistency term
            ||f_θ(t, x_t) - f_{θ⁻}(t+Δt, x_{t+Δt})||².
        velocity_anchor_loss_weight: Weight for the supervised velocity anchor.
            Keep this greater than 0 when training from scratch; consistency
            alone can learn a self-consistent but wrong velocity field.
        consistency_loss_weight: Weight for the teacher-student velocity
            consistency term.
        clean_endpoint_loss_weight: Weight for direct supervision of
            f_θ(t, x_t, μ) against x_clean. This biases training toward the
            exact image metric target used in 1-step inference.
        start_pair_prob: Probability of replacing the sampled pair with the
            first pair at σ_max. This gives more updates to the state used by
            true 1-step inference (x=μ, σ≈1).
        charbonnier_eps: Epsilon for Charbonnier loss
            sqrt((pred-target)^2 + eps^2). Used only when
            loss_type="charbonnier".
        cloud_mask_key: Batch key for the cloud mask used by cloud-weighted
            supervised losses. Set cloud_loss_weight<=1 to disable.
        cloud_loss_weight: Pixel weight applied on cloud regions. The weight
            map is normalized per sample, so the average loss scale stays
            close to the unweighted setting.
        cloud_weight_velocity_anchor: Also apply cloud weighting to the
            supervised velocity anchor term.
        non_cloud_identity_loss_weight: Weight for the non-cloud identity loss
            MSE((1-M)*f_student, (1-M)*x_cloudy). Penalises changes to already
            clear pixels, which directly improves whole-image PSNR.
        batch2model_keys: Keys forwarded from batch to the network.
    """

    def __init__(
        self,
        discretization_config: dict,
        loss_type: str = "l2",
        num_steps: int = 18,
        endpoint_loss_weight: float = 1.0,
        velocity_anchor_loss_weight: float = 1.0,
        consistency_loss_weight: float = 1.0,
        clean_endpoint_loss_weight: float = 1.0,
        start_pair_prob: float = 0.25,
        charbonnier_eps: float = 1e-3,
        cloud_mask_key: str = "M",
        cloud_loss_weight: float = 1.0,
        cloud_weight_velocity_anchor: bool = False,
        consistency_warmup_steps: int = 0,
        ssim_endpoint_loss_weight: float = 0.0,
        non_cloud_identity_loss_weight: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__()
        assert loss_type in ["l2", "l1", "charbonnier"], f"Unsupported loss_type: {loss_type}"
        assert num_steps >= 2, "ConsistencyFlowMatchingLoss requires num_steps >= 2"

        self.discretization = instantiate_from_config(discretization_config)
        self.loss_type = loss_type
        self.num_steps = num_steps
        self.endpoint_loss_weight = endpoint_loss_weight
        self.velocity_anchor_loss_weight = velocity_anchor_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.clean_endpoint_loss_weight = clean_endpoint_loss_weight
        self.start_pair_prob = min(max(float(start_pair_prob), 0.0), 1.0)
        self.charbonnier_eps = max(float(charbonnier_eps), 1e-12)
        self.cloud_mask_key = cloud_mask_key
        self.cloud_loss_weight = max(float(cloud_loss_weight), 1.0)
        self.cloud_weight_velocity_anchor = bool(cloud_weight_velocity_anchor)
        self.ssim_endpoint_loss_weight = float(ssim_endpoint_loss_weight)
        self.non_cloud_identity_loss_weight = float(non_cloud_identity_loss_weight)
        # While training from scratch the teacher is random for the first
        # few thousand steps. Linearly ramp the endpoint/velocity consistency
        # terms up from 0 so those steps are guided only by the supervised
        # velocity anchor (v* = x_clean − μ).
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

    def _get_cloud_mask(
        self, batch: Dict, input: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Return cloud mask [B,1,H,W] in [0,1], or None if unavailable."""
        if self.cloud_mask_key not in batch:
            return None
        mask = batch[self.cloud_mask_key]
        if torch.is_tensor(mask):
            mask = mask.to(device=input.device, dtype=input.dtype)
        else:
            mask = torch.as_tensor(mask, device=input.device, dtype=input.dtype)
        if mask.ndim == input.ndim - 1:
            mask = mask.unsqueeze(1)
        elif mask.ndim == input.ndim:
            if mask.shape[1] != 1:
                mask = mask.max(dim=1, keepdim=True).values
        else:
            raise ValueError(
                f"Cloud mask '{self.cloud_mask_key}' has shape {tuple(mask.shape)}, "
                f"expected [B,H,W] or [B,1,H,W] for input {tuple(input.shape)}"
            )
        if mask.shape[0] != input.shape[0]:
            raise ValueError(
                f"Cloud mask batch size {mask.shape[0]} does not match input "
                f"batch size {input.shape[0]}"
            )
        if mask.shape[-2:] != input.shape[-2:]:
            mask = F.interpolate(mask, size=input.shape[-2:], mode="nearest")
        return mask.clamp(0.0, 1.0)

    def _get_cloud_weight(
        self, batch: Dict, input: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self.cloud_loss_weight <= 1.0 or self.cloud_mask_key not in batch:
            return None
        mask = self._get_cloud_mask(batch, input)
        if mask is None:
            return None
        weight = 1.0 + (self.cloud_loss_weight - 1.0) * mask
        norm = weight.flatten(1).mean(dim=1).view(
            -1, *([1] * (input.ndim - 1))
        )
        return weight / norm.clamp_min(1e-6)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        network: nn.Module,
        teacher_fn,                    # callable: (x, σ, st, cond, **kw) → v_target
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        sigma2st: Sigma2St,            # ConsistencyFlowMatchingSigma2St expected
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
        if self.start_pair_prob > 0.0:
            start_mask = torch.rand(B, device=device) < self.start_pair_prob
            indices = torch.where(start_mask, torch.zeros_like(indices), indices)
        sigma_n   = sigmas[indices]           # σ_n  (larger = more cloudy, t_n small)
        sigma_n1  = sigmas[indices + 1]       # σ_{n+1} (smaller = more clean, t_{n+1} large)

        # t = 1 − σ  (CFM time)
        t_n   = 1.0 - sigma_n
        t_n1  = 1.0 - sigma_n1

        # Time-embedding values via sigma2st  (= t for CFM sigma2st)
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

        # ── 5. CFM endpoint map f(t, x) = x + (1 - t) v(t, x) ─────────────
        # Because σ = 1 - t, the remaining time to the endpoint is σ.
        f_student = x_tn + sigma_n_bc * v_student
        f_target = x_tn1 + sigma_n1_bc * v_target

        # ── 6. Supervised velocity anchor: true OT velocity ────────────────
        # Consistency alone only makes the velocity constant along a
        # trajectory; this term pins that constant to the correct direction.
        velocity_target = input - mu

        # ── 7. Combined CFM loss ───────────────────────────────────────────
        cloud_weight = self._get_cloud_weight(batch, input)
        velocity_anchor_weight = (
            cloud_weight if self.cloud_weight_velocity_anchor else None
        )
        endpoint_loss = self._get_loss(f_student, f_target.detach())
        velocity_consistency_loss = self._get_loss(v_student, v_target)
        velocity_anchor_loss = self._get_loss(
            v_student, velocity_target, velocity_anchor_weight
        )
        clean_endpoint_loss = self._get_loss(f_student, input, cloud_weight)

        ssim_loss = (
            self._ms_ssim_loss(f_student, input)
            if self.ssim_endpoint_loss_weight > 0.0
            else f_student.new_zeros(B)
        )

        # Non-cloud identity loss: penalise changes to already-clear pixels.
        # L_id = loss((1-M)*f_student, (1-M)*x_cloudy)
        if self.non_cloud_identity_loss_weight > 0.0:
            cloud_mask = self._get_cloud_mask(batch, input)
            if cloud_mask is not None:
                non_cloud = 1.0 - cloud_mask
                identity_loss = self._get_loss(non_cloud * f_student, non_cloud * mu)
            else:
                identity_loss = f_student.new_zeros(B)
        else:
            identity_loss = f_student.new_zeros(B)

        # Warmup: disable the consistency / endpoint terms during the first
        # `consistency_warmup_steps` gradient updates so the student+teacher
        # first learn a correct velocity field from the supervised anchors.
        ramp = self._consistency_ramp(batch)

        return (
            ramp * self.endpoint_loss_weight * endpoint_loss
            + ramp * self.consistency_loss_weight * velocity_consistency_loss
            + self.velocity_anchor_loss_weight * velocity_anchor_loss
            + self.clean_endpoint_loss_weight * clean_endpoint_loss
            + self.ssim_endpoint_loss_weight * ssim_loss
            + self.non_cloud_identity_loss_weight * identity_loss
        )

    def _get_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pixel_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.loss_type == "l2":
            loss = (pred - target) ** 2
        elif self.loss_type == "l1":
            loss = (pred - target).abs()
        elif self.loss_type == "charbonnier":
            diff = (pred - target).float()
            eps_sq = diff.new_tensor(self.charbonnier_eps ** 2)
            loss = torch.sqrt(diff ** 2 + eps_sq)
        else:
            raise NotImplementedError(f"Unknown loss_type: {self.loss_type}")

        if pixel_weight is not None:
            loss = loss * pixel_weight
        return torch.mean(loss.reshape(pred.shape[0], -1), dim=1)

    def _ms_ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MS-SSIM loss averaged over batch. Computed per-channel then averaged.
        Returns shape [B] to match _get_loss convention."""
        weights = pred.new_tensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=torch.float32
        )
        scales = len(weights)
        B, C = pred.shape[:2]
        p = (pred.float().clamp(-1.0, 1.0) + 1.0) * 0.5
        t = (target.float().clamp(-1.0, 1.0) + 1.0) * 0.5
        eps = 1e-6

        ms_vals = []
        for s in range(scales):
            if s > 0:
                p = F.avg_pool2d(p, 2)
                t = F.avg_pool2d(t, 2)
            # gaussian window per channel via depthwise conv
            win_size = min(11, p.shape[-1] - (p.shape[-1] % 2 == 0))
            if win_size < 3:
                break
            sigma_g = 1.5
            coords = torch.arange(win_size, device=p.device, dtype=p.dtype) - win_size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma_g ** 2))
            g = g / g.sum()
            kernel = g[:, None] * g[None, :]  # [win, win]
            kernel = kernel.expand(C, 1, win_size, win_size)
            pad = win_size // 2
            mu1 = F.conv2d(p, kernel, padding=pad, groups=C)
            mu2 = F.conv2d(t, kernel, padding=pad, groups=C)
            mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
            sig1_sq = F.conv2d(p * p, kernel, padding=pad, groups=C) - mu1_sq
            sig2_sq = F.conv2d(t * t, kernel, padding=pad, groups=C) - mu2_sq
            sig12   = F.conv2d(p * t, kernel, padding=pad, groups=C) - mu12
            C1, C2 = 0.01 ** 2, 0.03 ** 2
            cs = (2 * sig12 + C2) / (sig1_sq + sig2_sq + C2)
            if s < scales - 1:
                ms_vals.append(cs.mean(dim=[1, 2, 3]).clamp(eps, 1.0))  # [B]
            else:
                lum = (2 * mu12 + C1) / (mu1_sq + mu2_sq + C1)
                ms_vals.append((lum * cs).mean(dim=[1, 2, 3]).clamp(eps, 1.0))

        ssim_val = torch.ones(B, device=pred.device, dtype=torch.float32)
        for i, v in enumerate(ms_vals):
            ssim_val = ssim_val * (v.clamp(eps, 1.0) ** weights[i])
        return (1.0 - ssim_val.clamp(max=1.0)).to(dtype=pred.dtype)
