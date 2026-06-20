"""Noise-bridge Consistency Flow Matching loss for cloud removal.

The student and EMA teacher are evaluated on two points from the same random
straight path:

    y       = x_cloudy + noise_scale * eps
    x_t     = (1 - t) * y + t * x_clean
    v_true  = x_clean - y

The original cloudy image remains the clean network condition. Sharing the
same ``eps`` between both path points is essential: otherwise endpoint and
velocity consistency would compare different random trajectories.
"""

from typing import Dict

import torch
import torch.nn.functional as F

from ...modules.encoders.modules import GeneralConditioner
from ...util import append_dims
from .denoiser import Denoiser
from .loss_cfm import ConsistencyFlowMatchingLoss
from .sigma2st import Sigma2St


class NoiseBridgeConsistencyFlowMatchingLoss(ConsistencyFlowMatchingLoss):
    """CFM loss on a noisy-cloudy to clean straight path."""

    def __init__(
        self,
        *args,
        noise_sigma: float = 0.1,
        noise_ramp_steps: int = 0,
        spatial_noise: bool = False,
        noise_sigma_floor: float = 0.08,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if noise_sigma < 0.0:
            raise ValueError("noise_sigma must be non-negative")
        if noise_ramp_steps < 0:
            raise ValueError("noise_ramp_steps must be non-negative")
        if not 0.0 <= noise_sigma_floor <= 1.0:
            raise ValueError("noise_sigma_floor must be in [0, 1]")
        self.noise_sigma = float(noise_sigma)
        self.noise_ramp_steps = int(noise_ramp_steps)
        self.spatial_noise = bool(spatial_noise)
        self.noise_sigma_floor = float(noise_sigma_floor)

    def noise_sigma_at(self, global_step) -> float:
        if self.noise_ramp_steps == 0:
            return self.noise_sigma
        if torch.is_tensor(global_step):
            global_step = global_step.detach().item()
        step = max(float(global_step), 0.0)
        return self.noise_sigma * min(
            step / float(self.noise_ramp_steps),
            1.0,
        )

    def _build_bridge_points(
        self,
        x_clean: torch.Tensor,
        x_cloudy: torch.Tensor,
        t_current: torch.Tensor,
        t_next: torch.Tensor,
        noise: torch.Tensor,
        noise_scale,
    ):
        noisy_start = x_cloudy + noise_scale * noise
        t_current_bc = append_dims(t_current, x_clean.ndim)
        t_next_bc = append_dims(t_next, x_clean.ndim)
        x_current = (
            (1.0 - t_current_bc) * noisy_start
            + t_current_bc * x_clean
        )
        x_next = (
            (1.0 - t_next_bc) * noisy_start
            + t_next_bc * x_clean
        )
        velocity = x_clean - noisy_start
        return noisy_start, x_current, x_next, velocity

    def _get_noise_scale(
        self,
        batch: Dict,
        input: torch.Tensor,
        noise_sigma: float,
    ):
        if not self.spatial_noise:
            return noise_sigma

        cloud_mask = self._get_cloud_mask(batch, input)
        if cloud_mask is None:
            raise RuntimeError(
                "spatial_noise=true requires the configured cloud mask "
                f"batch key {self.cloud_mask_key!r}."
            )
        floor = self.noise_sigma_floor
        return noise_sigma * (floor + (1.0 - floor) * cloud_mask)

    def _forward(
        self,
        network,
        teacher_fn,
        denoiser: Denoiser,
        cond: Dict,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        batch_size = input.shape[0]
        device = input.device
        diffusion_model = getattr(network, "diffusion_model", network)
        mask_head_enabled = bool(
            getattr(diffusion_model, "predict_cloud_mask", False)
        )
        if mask_head_enabled and self.cloud_mask_pred_loss_weight <= 0.0:
            raise ValueError(
                "predict_cloud_mask=true requires "
                "cloud_mask_pred_loss_weight > 0."
            )
        if self.cloud_mask_pred_loss_weight > 0.0 and not mask_head_enabled:
            raise ValueError(
                "cloud_mask_pred_loss_weight > 0 requires "
                "predict_cloud_mask=true."
            )

        sigmas = self.discretization(self.num_steps, device=device)
        valid_len = len(sigmas) - 1
        indices = torch.randint(0, valid_len, (batch_size,), device=device)
        if self.start_pair_prob > 0.0:
            start_mask = torch.rand(batch_size, device=device) < self.start_pair_prob
            indices = torch.where(start_mask, torch.zeros_like(indices), indices)

        sigma_current = sigmas[indices]
        sigma_next = sigmas[indices + 1]
        t_current = 1.0 - sigma_current
        t_next = 1.0 - sigma_next
        st_current = sigma2st(sigma_current)
        st_next = sigma2st(sigma_next)

        current_noise_sigma = self.noise_sigma_at(batch.get("global_step", 0))
        if noise is None:
            noise = (
                torch.zeros_like(input)
                if current_noise_sigma == 0.0
                else torch.randn_like(input)
            )
        elif noise.shape != input.shape:
            raise ValueError(
                f"noise shape {tuple(noise.shape)} does not match input "
                f"{tuple(input.shape)}"
            )

        noise_scale = self._get_noise_scale(batch, input, current_noise_sigma)
        _, x_current, x_next, velocity_target = self._build_bridge_points(
            input,
            mu,
            t_current,
            t_next,
            noise,
            noise_scale,
        )

        velocity_teacher = teacher_fn(
            x_next,
            sigma_next,
            st_next,
            cond,
            **additional_model_inputs,
        )
        velocity_student = denoiser(
            network,
            x_current,
            sigma_current,
            cond,
            st_current,
            **additional_model_inputs,
        )

        sigma_current_bc = append_dims(sigma_current, input.ndim)
        sigma_next_bc = append_dims(sigma_next, input.ndim)
        endpoint_student = x_current + sigma_current_bc * velocity_student
        endpoint_teacher = x_next + sigma_next_bc * velocity_teacher

        cloud_weight = self._get_cloud_weight(batch, input)
        velocity_anchor_weight = (
            cloud_weight if self.cloud_weight_velocity_anchor else None
        )
        endpoint_loss = self._get_loss(
            endpoint_student,
            endpoint_teacher.detach(),
        )
        velocity_consistency_loss = self._get_loss(
            velocity_student,
            velocity_teacher,
        )
        velocity_anchor_loss = self._get_loss(
            velocity_student,
            velocity_target,
            velocity_anchor_weight,
        )
        clean_endpoint_loss = self._get_loss(
            endpoint_student,
            input,
            cloud_weight,
        )

        if self.ssim_endpoint_loss_weight > 0.0:
            ssim_loss = self._ms_ssim_loss(endpoint_student, input)
        else:
            ssim_loss = endpoint_student.new_zeros(batch_size)

        if self.non_cloud_identity_loss_weight > 0.0:
            cloud_mask = self._get_cloud_mask(batch, input)
            if cloud_mask is None:
                identity_loss = endpoint_student.new_zeros(batch_size)
            else:
                non_cloud = 1.0 - cloud_mask
                identity_loss = self._get_loss(
                    non_cloud * endpoint_student,
                    non_cloud * mu,
                )
        else:
            identity_loss = endpoint_student.new_zeros(batch_size)

        if self.cloud_mask_pred_loss_weight > 0.0:
            mask_logits = getattr(
                diffusion_model,
                "last_mask_logits",
                None,
            )
            cloud_mask = self._get_cloud_mask(batch, input)
            if mask_logits is None or cloud_mask is None:
                raise RuntimeError(
                    "cloud_mask_pred_loss_weight requires a cloud-mask head "
                    "and the configured cloud mask batch key."
                )
            mask_pred_loss = F.binary_cross_entropy_with_logits(
                mask_logits.float(),
                cloud_mask.float(),
                reduction="none",
            ).reshape(batch_size, -1).mean(dim=1).to(dtype=input.dtype)
        else:
            mask_pred_loss = endpoint_student.new_zeros(batch_size)

        consistency_ramp = self._consistency_ramp(batch)
        return (
            consistency_ramp * self.endpoint_loss_weight * endpoint_loss
            + consistency_ramp
            * self.consistency_loss_weight
            * velocity_consistency_loss
            + self.velocity_anchor_loss_weight * velocity_anchor_loss
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
            network,
            teacher_fn,
            denoiser,
            cond,
            sigma2st,
            input,
            mu,
            batch,
        )
