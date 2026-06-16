"""Noise-bridge MeanFlow loss for one-step cloud removal.

The bridge starts from a noisy cloudy observation while keeping the original
cloudy image as a clean condition:

    y   = x_cloudy + noise_sigma * eps
    x_s = (1 - s) * y + s * x_clean
    v   = x_clean - y

The network predicts the average velocity over [s, T]. This module is the P1
implementation: it keeps the existing MeanFlow pair sampler, Charbonnier loss,
dual-time conditioning, and clean-endpoint supervision while leaving mask
heads, perceptual terms, and statistical preconditioning disabled by config.
"""

from typing import Dict

import torch

from ...modules.encoders.modules import GeneralConditioner
from ...util import append_dims
from .denoiser import Denoiser
from .loss_meanflow import MeanFlowLoss, meanflow_c_noise
from .sigma2st import Sigma2St


class NoiseBridgeMeanFlowLoss(MeanFlowLoss):
    """MeanFlow identity loss on a noisy-cloudy to clean straight path.

    Args:
        noise_sigma: Target standard deviation of additive start-point noise.
        noise_ramp_steps: Number of optimizer steps used to linearly ramp the
            noise from zero to ``noise_sigma``. Set to zero to disable ramping.

    The public ``forward`` signature matches ``ResidualDiffusionEngine`` and
    intentionally has no EMA-teacher argument.
    """

    def __init__(
        self,
        *args,
        noise_sigma: float = 0.1,
        noise_ramp_steps: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if noise_sigma < 0.0:
            raise ValueError("noise_sigma must be non-negative")
        if noise_ramp_steps < 0:
            raise ValueError("noise_ramp_steps must be non-negative")

        self.noise_sigma = float(noise_sigma)
        self.noise_ramp_steps = int(noise_ramp_steps)

        if self.cloud_mask_pred_loss_weight > 0.0:
            raise ValueError(
                "P1 NoiseBridgeMeanFlowLoss does not train a cloud-mask head; "
                "set cloud_mask_pred_loss_weight=0."
            )

    def noise_sigma_at(self, global_step) -> float:
        """Return the linearly ramped noise level for a training step."""
        if self.noise_ramp_steps == 0:
            return self.noise_sigma

        if torch.is_tensor(global_step):
            global_step = global_step.detach().item()
        step = max(float(global_step), 0.0)
        ramp = min(step / float(self.noise_ramp_steps), 1.0)
        return self.noise_sigma * ramp

    def _build_bridge(
        self,
        x_clean: torch.Tensor,
        x_cloudy: torch.Tensor,
        s: torch.Tensor,
        noise: torch.Tensor,
        noise_sigma: float,
    ):
        """Construct the noisy start, current state, and conditional velocity."""
        noisy_start = x_cloudy + noise_sigma * noise
        s_bc = append_dims(s, x_clean.ndim)
        x_s = (1.0 - s_bc) * noisy_start + s_bc * x_clean
        velocity = x_clean - noisy_start
        return noisy_start, x_s, velocity

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
        """One-sided, domain-valid derivative along the bridge trajectory.

        The step is shortened per sample so that ``s + h <= T``. Equal-time
        pairs need no derivative because their MeanFlow correction is exactly
        multiplied by zero.
        """
        gap = (T - s).clamp_min(0.0)
        step = torch.minimum(gap, torch.full_like(gap, self.fd_eps))
        active = step > 0.0
        step_bc = append_dims(step, x_s.ndim)

        devices = [x_s.device.index] if x_s.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            # Reuse the primary forward's dropout masks for the FD probe.
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, device=x_s.device)
            with torch.no_grad():
                u_shift = u_fn(x_s + step_bc * velocity, s + step, T)

        denom = append_dims(step.clamp_min(torch.finfo(step.dtype).eps), x_s.ndim)
        du_ds = (u_shift - u.detach()) / denom
        active_bc = append_dims(active, x_s.ndim)
        return torch.where(active_bc, du_ds, torch.zeros_like(du_ds))

    def _forward(
        self,
        network,
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
        diffusion_model = getattr(network, "diffusion_model", network)
        if getattr(diffusion_model, "predict_cloud_mask", False):
            raise ValueError(
                "P1 noise-bridge training requires predict_cloud_mask=false."
            )

        s, T = self._sample_pairs(batch_size, input.device)
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

        _, x_s, velocity = self._build_bridge(
            input, mu, s, noise, current_noise_sigma
        )

        def u_fn(x_in, s_in, T_in):
            sigma_in = 1.0 - s_in
            return denoiser(
                network,
                x_in,
                sigma_in,
                cond,
                s_in,
                timesteps_r=meanflow_c_noise(T_in),
                **additional_model_inputs,
            )

        if self.jvp_mode == "jvp":
            u, du_ds = torch.func.jvp(
                u_fn,
                (x_s, s, T),
                (
                    velocity,
                    torch.ones_like(s),
                    torch.zeros_like(T),
                ),
            )
        else:
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_state = (
                torch.cuda.get_rng_state(input.device)
                if input.is_cuda
                else None
            )
            u = u_fn(x_s, s, T)
            du_ds = self._finite_difference_du_ds(
                u_fn,
                u,
                x_s,
                s,
                T,
                velocity,
                cpu_rng_state,
                cuda_rng_state,
            )

        gap_bc = append_dims(T - s, input.ndim)
        u_target = (velocity + gap_bc * du_ds).detach()

        cloud_weight = self._get_cloud_weight(batch, input)
        mf_weight = cloud_weight if self.cloud_weight_velocity_anchor else None
        meanflow_loss = self._get_loss(u, u_target, mf_weight)

        endpoint = x_s + gap_bc * u
        end_mask = (T >= 1.0 - 1e-6).to(dtype=input.dtype)
        clean_endpoint_loss = (
            self._get_loss(endpoint, input, cloud_weight) * end_mask
        )

        if self.ssim_endpoint_loss_weight > 0.0:
            ssim_loss = self._ms_ssim_loss(endpoint, input) * end_mask
        else:
            ssim_loss = endpoint.new_zeros(batch_size)

        if self.non_cloud_identity_loss_weight > 0.0:
            cloud_mask = self._get_cloud_mask(batch, input)
            if cloud_mask is None:
                identity_loss = endpoint.new_zeros(batch_size)
            else:
                non_cloud = 1.0 - cloud_mask
                identity_loss = self._get_loss(
                    non_cloud * endpoint,
                    non_cloud * mu,
                ) * end_mask
        else:
            identity_loss = endpoint.new_zeros(batch_size)

        return (
            self.meanflow_loss_weight * meanflow_loss
            + self.clean_endpoint_loss_weight * clean_endpoint_loss
            + self.ssim_endpoint_loss_weight * ssim_loss
            + self.non_cloud_identity_loss_weight * identity_loss
        )

    def forward(
        self,
        network,
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
            denoiser,
            cond,
            sigma2st,
            input,
            mu,
            batch,
        )
