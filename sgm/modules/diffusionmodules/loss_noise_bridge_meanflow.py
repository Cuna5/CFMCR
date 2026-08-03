"""Noise-bridge MeanFlow loss for one-step cloud removal.

The bridge starts from a noisy cloudy observation while keeping the original
cloudy image as a clean condition:

    gamma_train = clamp(degradation(x_cloudy, x_clean) / tau, 0, 1)
    noise_scale = noise_sigma * (floor + (1 - floor) * gamma_train)
    y   = x_cloudy + noise_scale * eps
    x_s = (1 - s) * y + s * x_clean
    v   = x_clean - y

The network predicts the average velocity over [s, T]. A dedicated prepass at
the exact one-step state (s=0, T=1) trains the shared one-channel head against
the paired-degradation gamma target. Inference uses that same state and head to
predict the spatial noise scale without access to the clean target.
"""

from typing import Dict

import torch

from ...modules.encoders.modules import GeneralConditioner
from ...util import append_dims
from .denoiser import Denoiser
from .loss_meanflow import MeanFlowLoss, meanflow_c_noise
from .loss_noise_bridge_cfm import NoiseBridgeConsistencyFlowMatchingLoss
from .sigma2st import Sigma2St


class NoiseBridgeMeanFlowLoss(MeanFlowLoss):
    """MeanFlow identity loss on a noisy-cloudy to clean straight path.

    Args:
        noise_sigma: Target standard deviation of additive start-point noise.
        noise_ramp_steps: Number of optimizer steps used to linearly ramp the
            noise from zero to ``noise_sigma``. Set to zero to disable ramping.
        spatial_noise: Modulate the bridge noise per pixel with paired
            degradation gamma.
        noise_sigma_floor: Fraction of ``noise_sigma`` retained where gamma is
            zero.
        gamma_head_loss_weight: Weight for teaching the shared one-channel
            head to reproduce the training noise scale at inference.

    The public ``forward`` signature matches ``ResidualDiffusionEngine`` and
    intentionally has no EMA-teacher argument.
    """

    def __init__(
        self,
        *args,
        noise_sigma: float = 0.1,
        noise_ramp_steps: int = 0,
        spatial_noise: bool = False,
        noise_sigma_floor: float = 0.08,
        spatial_noise_source: str = "degradation",
        gamma_delta_tau: float = 0.5,
        gamma_delta_reduction: str = "rms",
        gamma_smooth_kernel: int = 0,
        gamma_head_loss_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if noise_sigma < 0.0:
            raise ValueError("noise_sigma must be non-negative")
        if noise_ramp_steps < 0:
            raise ValueError("noise_ramp_steps must be non-negative")
        if not 0.0 <= noise_sigma_floor <= 1.0:
            raise ValueError("noise_sigma_floor must be in [0, 1]")
        if spatial_noise_source != "degradation":
            raise ValueError(
                "NoiseBridgeMeanFlowLoss supports only "
                "spatial_noise_source='degradation'."
            )
        if gamma_delta_tau <= 0.0:
            raise ValueError("gamma_delta_tau must be positive")
        if gamma_delta_reduction not in ("rms", "mean_abs", "l2"):
            raise ValueError(
                "gamma_delta_reduction must be 'rms', 'mean_abs', or 'l2'"
            )
        if gamma_smooth_kernel < 0:
            raise ValueError("gamma_smooth_kernel must be non-negative")
        if gamma_head_loss_weight < 0.0:
            raise ValueError("gamma_head_loss_weight must be non-negative")

        self.noise_sigma = float(noise_sigma)
        self.noise_ramp_steps = int(noise_ramp_steps)
        self.spatial_noise = bool(spatial_noise)
        self.noise_sigma_floor = float(noise_sigma_floor)
        self.spatial_noise_source = spatial_noise_source
        self.gamma_delta_tau = float(gamma_delta_tau)
        self.gamma_delta_reduction = gamma_delta_reduction
        self.gamma_smooth_kernel = int(gamma_smooth_kernel)
        self.gamma_head_loss_weight = float(gamma_head_loss_weight)

        if self.cloud_mask_pred_loss_weight > 0.0:
            raise ValueError(
                "NoiseBridgeMeanFlowLoss reserves the shared one-channel head "
                "for paired-degradation gamma; set "
                "cloud_mask_pred_loss_weight=0."
            )

    # Keep the gamma target, spatial scaling, resizing, and head-loss semantics
    # identical to the already-tested CFM bridge. These methods depend only on
    # attributes initialized above and do not pull in teacher/Stage-C behavior.
    _noise_scale_from_prob = (
        NoiseBridgeConsistencyFlowMatchingLoss._noise_scale_from_prob
    )
    _get_degradation_prob = (
        NoiseBridgeConsistencyFlowMatchingLoss._get_degradation_prob
    )
    _get_noise_scale = NoiseBridgeConsistencyFlowMatchingLoss._get_noise_scale
    _get_head_logits = NoiseBridgeConsistencyFlowMatchingLoss._get_head_logits
    _gamma_prob_from_logits = (
        NoiseBridgeConsistencyFlowMatchingLoss._gamma_prob_from_logits
    )
    _noise_scale_from_logits = (
        NoiseBridgeConsistencyFlowMatchingLoss._noise_scale_from_logits
    )
    _gamma_head_loss = NoiseBridgeConsistencyFlowMatchingLoss._gamma_head_loss

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
        noise_scale,
    ):
        """Construct the noisy start, current state, and conditional velocity."""
        noisy_start = x_cloudy + noise_scale * noise
        s_bc = append_dims(s, x_clean.ndim)
        x_s = (1.0 - s_bc) * noisy_start + s_bc * x_clean
        velocity = x_clean - noisy_start
        return noisy_start, x_s, velocity

    def _gamma_prepass_logits(
        self,
        network,
        denoiser: Denoiser,
        diffusion_model,
        cond: Dict,
        sigma2st: Sigma2St,
        mu: torch.Tensor,
        additional_model_inputs: Dict,
    ):
        """Predict gamma at the exact inference state and preserve its logits.

        MeanFlow uses two time inputs. The gamma prepass must therefore match
        one-step inference exactly: current ``s=0`` (``sigma=1``) and jump
        target ``T=1``. Returning the tensor itself is important because the
        subsequent primary/FD forwards overwrite ``last_mask_logits``.
        """
        sigma_max = float(getattr(self.discretization, "sigma_max", 1.0))
        sigma = mu.new_full((mu.shape[0],), sigma_max)
        st = sigma2st(sigma)
        prepass_inputs = dict(additional_model_inputs)
        prepass_inputs.pop("skip_gamma", None)
        prepass_inputs.pop("timesteps_r", None)
        _ = denoiser(
            network,
            mu,
            sigma,
            cond,
            st,
            timesteps_r=meanflow_c_noise(torch.ones_like(sigma)),
            **prepass_inputs,
        )
        logits = self._get_head_logits(diffusion_model)
        if logits is None:
            raise RuntimeError(
                "gamma-head prepass produced no logits; enable "
                "network_config.params.predict_cloud_mask."
            )
        return logits

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
        if getattr(diffusion_model, "adaptive_skip_fusion", False):
            raise ValueError(
                "NoiseBridgeMeanFlowLoss does not implement adaptive skip "
                "fusion; keep network_config.params.adaptive_skip_fusion=false."
            )
        gamma_head_enabled = bool(
            getattr(diffusion_model, "predict_cloud_mask", False)
        )
        if self.spatial_noise and not gamma_head_enabled:
            raise ValueError(
                "spatial_noise=true requires "
                "network_config.params.predict_cloud_mask=true."
            )
        if self.spatial_noise and self.gamma_head_loss_weight <= 0.0:
            raise ValueError(
                "spatial_noise=true requires gamma_head_loss_weight > 0 so "
                "inference can predict the paired-degradation gamma."
            )
        if self.gamma_head_loss_weight > 0.0 and not gamma_head_enabled:
            raise ValueError(
                "gamma_head_loss_weight > 0 requires "
                "network_config.params.predict_cloud_mask=true."
            )
        if gamma_head_enabled and self.gamma_head_loss_weight <= 0.0:
            raise ValueError(
                "predict_cloud_mask=true requires gamma_head_loss_weight > 0 "
                "for NoiseBridgeMeanFlowLoss."
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

        train_noise_scale = self._get_noise_scale(
            batch,
            input,
            current_noise_sigma,
            mu,
        )
        gamma_head_logits = None
        if self.gamma_head_loss_weight > 0.0:
            gamma_head_logits = self._gamma_prepass_logits(
                network,
                denoiser,
                diffusion_model,
                cond,
                sigma2st,
                mu,
                additional_model_inputs,
            )

        _, x_s, velocity = self._build_bridge(
            input, mu, s, noise, train_noise_scale
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

        if self.gamma_head_loss_weight > 0.0:
            pred_noise_scale = self._noise_scale_from_logits(
                gamma_head_logits,
                input,
                current_noise_sigma,
            )
            gamma_head_loss = self._gamma_head_loss(
                pred_noise_scale,
                train_noise_scale,
                batch_size,
            )
        else:
            gamma_head_loss = endpoint.new_zeros(batch_size)

        return (
            self.meanflow_loss_weight * meanflow_loss
            + self.clean_endpoint_loss_weight * clean_endpoint_loss
            + self.ssim_endpoint_loss_weight * ssim_loss
            + self.non_cloud_identity_loss_weight * identity_loss
            + self.gamma_head_loss_weight * gamma_head_loss
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
