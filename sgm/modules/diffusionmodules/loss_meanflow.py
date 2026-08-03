"""
MeanFlow loss for one-step restoration flows (cloud removal).

Reference: Geng et al., "Mean Flows for One-step Generative Modeling"
(NeurIPS 2025), adapted from noise->data generation to the deterministic
cloudy->clean restoration path used by this repository.

Mathematical background
-----------------------
The restoration path is the straight OT interpolation (s in [0, 1]):

    x_s = (1 - s) * mu + s * x_clean,        v = dx/ds = x_clean - mu

Instead of the instantaneous velocity v_theta of CFM, the network models the
*average* velocity over a jump [s, T] (s <= T), evaluated at the current
state x_s:

    u(x_s, s, T) = 1/(T - s) * \\int_s^T v(x_tau, tau) dtau

so that the displacement is exact by construction:

    x_T = x_s + (T - s) * u(x_s, s, T)

Differentiating the integral w.r.t. the *lower* limit s along the trajectory
(dx/ds = v) gives the MeanFlow identity in forward orientation:

    u(x_s, s, T) = v(x_s, s) + (T - s) * d/ds u(x_s, s, T)

with the total derivative d/ds u = v . grad_x u + d_s u, computed in a single
forward pass via a jacobian-vector product (JVP) with tangents (v, 1, 0) for
the arguments (x, s, T). The training target uses the conditional OT velocity
v = x_clean - mu and a stop-gradient on the derivative term:

    u_tgt = (x_clean - mu) + (T - s) * sg[d/ds u_theta]
    L_mf  = d(u_theta(x_s, s, T), sg[u_tgt])

Special cases recover familiar objectives:
  * T = s   : u_tgt = v  ->  supervised velocity anchor (pure flow matching);
  * (s, T) = (0, 1): trains exactly the state used by 1-step inference,
                     x_pred = mu + u_theta(mu, 0, 1).

No EMA teacher, no consistency warmup, and no discrete sigma-pair schedule
are required — the identity itself enforces self-consistency across jumps.

Practical notes
---------------
* `jvp_mode="jvp"` uses torch.func.jvp (exact, single fused forward). Some
  fused attention kernels (natten / flash-attn) may not implement
  forward-mode AD; `jvp_mode="fd"` falls back to a finite-difference
  estimate along the trajectory at the cost of a second (no-grad) forward.
* The clean-endpoint, MS-SSIM, non-cloud-identity and cloud-mask-head terms
  are applied only to batch elements whose jump target is T = 1, because
  only there f = x_s + (1 - s) * u predicts the clean image.
* Cloud weighting is reused from the CFM loss; with
  `cloud_weight_velocity_anchor=True` it also weights the MeanFlow term.
* The denoiser must use MeanFlowScaling (smooth log-time embedding) so the
  JVP time-derivative does not vanish at s = 0, and the network must enable
  `use_dual_time=true` to receive the jump-target embedding `timesteps_r`.

Engine compatibility: forward() keeps the ConsistencyFlowMatchingLoss
signature (teacher_fn is accepted and ignored). The dedicated
``MeanFlowEngine`` passes ``None`` and avoids creating a redundant CFM teacher.
"""

from typing import Dict

import torch
import torch.nn.functional as F
import torch.autograd.forward_ad as fwAD

from ...util import append_dims
from .denoiser import Denoiser
from ...modules.encoders.modules import GeneralConditioner
from .sigma2st import Sigma2St
from .loss_cfm import ConsistencyFlowMatchingLoss


# Smooth time embedding for the jump target T; must match MeanFlowScaling.
MEANFLOW_TIME_EPS = 1e-4


def meanflow_c_noise(t: torch.Tensor, eps: float = MEANFLOW_TIME_EPS) -> torch.Tensor:
    """Transform a raw CFM time in [0, 1] to the network's log-time input."""
    return 0.25 * torch.log(t + eps)


class MeanFlowLoss(ConsistencyFlowMatchingLoss):
    """MeanFlow identity loss on the cloudy->clean OT path.

    Inherits Charbonnier / cloud weighting / MS-SSIM / identity / mask-head
    helpers from ConsistencyFlowMatchingLoss. Parent parameters that only
    concern the EMA teacher (endpoint_loss_weight, consistency_loss_weight,
    consistency_warmup_steps, start_pair_prob, num_steps as pair grid) are
    ignored.

    Args (additional to the parent):
        meanflow_loss_weight: Weight of the MeanFlow identity term.
        full_pair_prob: Probability of forcing (s, T) = (0, 1) — the exact
            1-step inference state (analog of the CFM start_pair_prob).
        t1_pair_prob: Probability of (s ~ U[0,1], T = 1) — random start,
            clean jump target; keeps the endpoint losses active.
        equal_pair_prob: Probability of T = s — degenerates to the supervised
            velocity anchor (the derivative term vanishes).
            The remaining probability mass samples s = min(u1, u2),
            T = max(u1, u2) with u1, u2 ~ U[0, 1] (generic jumps used by
            multi-step sampling).
        jvp_mode: "jvp" (torch.func.jvp, exact) or "fd" (finite difference).
        fd_eps: Step size for the finite-difference fallback.
    """

    def __init__(
        self,
        *args,
        meanflow_loss_weight: float = 1.0,
        full_pair_prob: float = 0.35,
        t1_pair_prob: float = 0.25,
        equal_pair_prob: float = 0.15,
        jvp_mode: str = "jvp",
        fd_eps: float = 1e-2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert jvp_mode in ("jvp", "fd"), f"unsupported jvp_mode: {jvp_mode}"
        if fd_eps <= 0.0:
            raise ValueError("fd_eps must be positive")
        p_sum = full_pair_prob + t1_pair_prob + equal_pair_prob
        assert 0.0 <= p_sum <= 1.0, "pair probabilities must sum to <= 1"
        self.meanflow_loss_weight = float(meanflow_loss_weight)
        self.full_pair_prob = float(full_pair_prob)
        self.t1_pair_prob = float(t1_pair_prob)
        self.equal_pair_prob = float(equal_pair_prob)
        self.jvp_mode = jvp_mode
        self.fd_eps = float(fd_eps)

    # ------------------------------------------------------------------
    # (s, T) pair sampling
    # ------------------------------------------------------------------

    def _sample_pairs(self, B: int, device) -> tuple:
        u1 = torch.rand(B, device=device)
        u2 = torch.rand(B, device=device)
        s = torch.minimum(u1, u2)
        T = torch.maximum(u1, u2)

        cat = torch.rand(B, device=device)
        p_full = self.full_pair_prob
        p_t1 = p_full + self.t1_pair_prob
        p_eq = p_t1 + self.equal_pair_prob
        full = cat < p_full
        t1 = (cat >= p_full) & (cat < p_t1)
        eq = (cat >= p_t1) & (cat < p_eq)

        s = torch.where(full, torch.zeros_like(s), s)
        T = torch.where(full | t1, torch.ones_like(T), T)
        T = torch.where(eq, s, T)
        return s, T

    def _finite_difference_du_ds(
        self,
        u_fn,
        u,
        x_s,
        s,
        T,
        velocity,
        cpu_rng_state,
        cuda_rng_state,
    ):
        """Estimate the trajectory derivative without crossing the jump end.

        Replaying the primary forward's RNG state keeps dropout masks identical
        between ``u`` and the finite-difference probe. The per-sample step is
        clipped to ``T - s``; equal-time pairs therefore have an exact zero
        derivative contribution and no out-of-domain ``s > T`` evaluation.
        """
        gap = (T - s).clamp_min(0.0)
        step = torch.minimum(gap, torch.full_like(gap, self.fd_eps))
        active = step > 0.0
        step_bc = append_dims(step, x_s.ndim)

        devices = [x_s.device.index] if x_s.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, device=x_s.device)
            with torch.no_grad():
                u_shift = u_fn(x_s + step_bc * velocity, s + step, T)

        denom = append_dims(
            step.clamp_min(torch.finfo(step.dtype).eps),
            x_s.ndim,
        )
        du_ds = (u_shift - u.detach()) / denom
        active_bc = append_dims(active, x_s.ndim)
        return torch.where(active_bc, du_ds, torch.zeros_like(du_ds))

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _forward(
        self,
        network,
        teacher_fn,            # accepted for engine compatibility; unused
        denoiser: Denoiser,
        cond: Dict,
        sigma2st: Sigma2St,
        input: torch.Tensor,   # x_clean  [B, C, H, W]
        mu: torch.Tensor,      # cloudy   [B, C, H, W]
        batch: Dict,
    ) -> torch.Tensor:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        B = input.shape[0]
        device = input.device

        # ── 1. Sample jump pairs and build the OT state ────────────────────
        s, T = self._sample_pairs(B, device)
        s_bc = append_dims(s, input.ndim)
        x_s = (1.0 - s_bc) * mu + s_bc * input
        v = input - mu                     # conditional OT velocity (constant)

        # ── 2. Average-velocity field as a function of (x, s, T) ───────────
        # sigma = 1 - s feeds the denoiser exactly like CFM; the jump target
        # enters through the smooth log-time embedding `timesteps_r`.
        diffusion_model = getattr(network, "diffusion_model", network)
        mask_head_enabled = bool(
            getattr(diffusion_model, "predict_cloud_mask", False)
        )
        if mask_head_enabled and self.cloud_mask_pred_loss_weight <= 0.0:
            raise ValueError(
                "predict_cloud_mask=true requires "
                "cloud_mask_pred_loss_weight > 0; otherwise the mask head "
                "is unused under DDP."
            )
        if self.cloud_mask_pred_loss_weight > 0.0 and not mask_head_enabled:
            raise ValueError(
                "cloud_mask_pred_loss_weight > 0 requires "
                "network_config.params.predict_cloud_mask=true."
            )

        def u_fn(x_in, s_in, T_in):
            sigma_in = 1.0 - s_in
            return denoiser(
                network,
                x_in,
                sigma_in,
                cond,
                s_in,                       # st = sigma2st(sigma) = s
                timesteps_r=meanflow_c_noise(T_in),
                **additional_model_inputs,
            )

        # ── 3. MeanFlow identity target via JVP along the trajectory ───────
        # d/ds u = grad_x u . v + d_s u   (T held fixed)
        if self.jvp_mode == "jvp":
            u, du_ds = torch.func.jvp(
                u_fn,
                (x_s, s, T),
                (v, torch.ones_like(s), torch.zeros_like(T)),
            )
            mask_logits = getattr(diffusion_model, "last_mask_logits", None)
        else:
            # Save the state before the primary forward so the FD probe can
            # reuse the exact same dropout masks without rewinding the caller's
            # RNG stream after it completes.
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_state = (
                torch.cuda.get_rng_state(input.device)
                if input.is_cuda
                else None
            )
            u = u_fn(x_s, s, T)
            # Preserve logits from the grad-enabled primary forward. The
            # finite-difference probe below runs under no_grad and overwrites
            # diffusion_model.last_mask_logits.
            mask_logits = getattr(diffusion_model, "last_mask_logits", None)
            du_ds = self._finite_difference_du_ds(
                u_fn,
                u,
                x_s,
                s,
                T,
                v,
                cpu_rng_state,
                cuda_rng_state,
            )

        gap_bc = append_dims(T - s, input.ndim)
        u_tgt = (v + gap_bc * du_ds).detach()

        # ── 4. MeanFlow identity loss (cloud-weighted like the anchor) ─────
        cloud_weight = self._get_cloud_weight(batch, input)
        mf_weight = cloud_weight if self.cloud_weight_velocity_anchor else None
        meanflow_loss = self._get_loss(u, u_tgt, mf_weight)

        # ── 5. Endpoint losses on elements that jump to the clean end ──────
        # f = x_s + (T - s) * u equals the clean prediction only when T = 1.
        f = x_s + gap_bc * u
        end_mask = (T >= 1.0 - 1e-6).to(dtype=input.dtype)

        clean_endpoint_loss = self._get_loss(f, input, cloud_weight) * end_mask

        if self.ssim_endpoint_loss_weight > 0.0:
            ssim_loss = self._ms_ssim_loss(f, input) * end_mask
        else:
            ssim_loss = f.new_zeros(B)

        if self.non_cloud_identity_loss_weight > 0.0:
            cloud_mask = self._get_cloud_mask(batch, input)
            if cloud_mask is not None:
                non_cloud = 1.0 - cloud_mask
                identity_loss = self._get_loss(non_cloud * f, non_cloud * mu) * end_mask
            else:
                identity_loss = f.new_zeros(B)
        else:
            identity_loss = f.new_zeros(B)

        # ── 6. Cloud-probability head supervision ──────────────────────────
        # Logits come from the u_fn forward; under torch.func.jvp they may be
        # dual tensors, so unpack to the primal before BCE.
        if self.cloud_mask_pred_loss_weight > 0.0:
            cloud_mask = self._get_cloud_mask(batch, input)
            if mask_logits is None:
                raise RuntimeError(
                    "Cloud-mask head is enabled but produced no logits."
                )
            if cloud_mask is None:
                raise KeyError(
                    f"cloud_mask_pred_loss_weight > 0 requires batch key "
                    f"{self.cloud_mask_key!r}."
                )
            unpacked = fwAD.unpack_dual(mask_logits)
            mask_logits = (
                unpacked.primal
                if unpacked.primal is not None
                else mask_logits
            )
            mask_pred_loss = F.binary_cross_entropy_with_logits(
                mask_logits.float(), cloud_mask.float(), reduction="none"
            ).reshape(B, -1).mean(dim=1).to(dtype=input.dtype)
        else:
            mask_pred_loss = f.new_zeros(B)

        return (
            self.meanflow_loss_weight * meanflow_loss
            + self.clean_endpoint_loss_weight * clean_endpoint_loss
            + self.ssim_endpoint_loss_weight * ssim_loss
            + self.non_cloud_identity_loss_weight * identity_loss
            + self.cloud_mask_pred_loss_weight * mask_pred_loss
        )

    def forward(
        self,
        network,
        teacher_fn,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        return self._forward(
            network, teacher_fn, denoiser, cond, sigma2st, input, mu, batch
        )
