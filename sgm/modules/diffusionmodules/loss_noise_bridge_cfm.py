"""Noise-bridge Consistency Flow Matching loss for cloud removal.

The student and EMA teacher are evaluated on two points from the same random
straight path:

    y       = x_cloudy + noise_scale * eps
    x_t     = (1 - t) * y + t * x_clean
    v_true  = x_clean - y

The original cloudy image remains the clean network condition. Sharing the
same ``eps`` between both path points is essential: otherwise endpoint and
velocity consistency would compare different random trajectories.

When ``spatial_noise_source="degradation"``, the training scale is computed
from the paired degradation magnitude ``|x_cloudy - x_clean|`` and the shared
one-channel head is supervised to predict that gamma scale for inference. This
avoids using label-derived cloud masks at sampling time.

Optional residual-domain supervision uses the same paired degradation gamma:
the Haar LL band preserves coarse structure, gamma-weighted Haar high bands
focus detail learning on degraded regions, and a small gamma-windowed complex
FFT loss aligns the residual spectrum. These terms are training-only and do
not change the sampler or inference cost.
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
    """CFM loss on a noisy-cloudy to clean straight path.

    ``residual_wavelet_ll_loss_weight`` supervises the low-pass Haar band of
    ``endpoint - cloudy``. ``residual_wavelet_hf_loss_weight`` supervises the
    three high-pass bands with the paired-degradation gamma as a spatial
    weight. ``residual_fft_loss_weight`` compares the complex spectra of
    gamma-windowed residuals. All three default to zero for backward
    compatibility; the detail terms can be delayed and linearly warmed up.
    """

    def __init__(
        self,
        *args,
        noise_sigma: float = 0.1,
        noise_ramp_steps: int = 0,
        spatial_noise: bool = False,
        noise_sigma_floor: float = 0.08,
        spatial_noise_source: str = "mask",
        gamma_delta_tau: float = 0.5,
        gamma_delta_reduction: str = "rms",
        gamma_smooth_kernel: int = 0,
        gamma_head_loss_weight: float = 0.0,
        gamma_mix_start_step: int = 0,
        gamma_mix_end_step: int = 0,
        gamma_mix_max_prob: float = 0.0,
        residual_wavelet_ll_loss_weight: float = 0.0,
        residual_wavelet_hf_loss_weight: float = 0.0,
        residual_fft_loss_weight: float = 0.0,
        residual_detail_start_step: int = 0,
        residual_detail_warmup_steps: int = 0,
        residual_fft_channels: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if noise_sigma < 0.0:
            raise ValueError("noise_sigma must be non-negative")
        if noise_ramp_steps < 0:
            raise ValueError("noise_ramp_steps must be non-negative")
        if not 0.0 <= noise_sigma_floor <= 1.0:
            raise ValueError("noise_sigma_floor must be in [0, 1]")
        if spatial_noise_source not in ("mask", "degradation"):
            raise ValueError(
                "spatial_noise_source must be 'mask' or 'degradation'"
            )
        if gamma_delta_tau <= 0.0:
            raise ValueError("gamma_delta_tau must be positive")
        if gamma_delta_reduction not in ("rms", "mean_abs", "l2"):
            raise ValueError(
                "gamma_delta_reduction must be 'rms', 'mean_abs', or 'l2'"
            )
        frequency_weights = (
            residual_wavelet_ll_loss_weight,
            residual_wavelet_hf_loss_weight,
            residual_fft_loss_weight,
        )
        if any(weight < 0.0 for weight in frequency_weights):
            raise ValueError("residual wavelet/FFT loss weights must be non-negative")
        if residual_detail_start_step < 0:
            raise ValueError("residual_detail_start_step must be non-negative")
        if residual_detail_warmup_steps < 0:
            raise ValueError("residual_detail_warmup_steps must be non-negative")
        if residual_fft_channels < 0:
            raise ValueError("residual_fft_channels must be non-negative")
        self.noise_sigma = float(noise_sigma)
        self.noise_ramp_steps = int(noise_ramp_steps)
        self.spatial_noise = bool(spatial_noise)
        self.noise_sigma_floor = float(noise_sigma_floor)
        self.spatial_noise_source = spatial_noise_source
        self.gamma_delta_tau = float(gamma_delta_tau)
        self.gamma_delta_reduction = gamma_delta_reduction
        self.gamma_smooth_kernel = int(gamma_smooth_kernel)
        self.gamma_head_loss_weight = float(gamma_head_loss_weight)
        self.gamma_mix_start_step = max(int(gamma_mix_start_step), 0)
        self.gamma_mix_end_step = max(int(gamma_mix_end_step), 0)
        self.gamma_mix_max_prob = min(max(float(gamma_mix_max_prob), 0.0), 1.0)
        self.residual_wavelet_ll_loss_weight = float(
            residual_wavelet_ll_loss_weight
        )
        self.residual_wavelet_hf_loss_weight = float(
            residual_wavelet_hf_loss_weight
        )
        self.residual_fft_loss_weight = float(residual_fft_loss_weight)
        self.residual_detail_start_step = int(residual_detail_start_step)
        self.residual_detail_warmup_steps = int(residual_detail_warmup_steps)
        self.residual_fft_channels = int(residual_fft_channels)

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

    def _noise_scale_from_prob(self, prob: torch.Tensor, noise_sigma: float):
        floor = self.noise_sigma_floor
        return noise_sigma * (floor + (1.0 - floor) * prob)

    def _get_degradation_prob(
        self,
        x_clean: torch.Tensor,
        x_cloudy: torch.Tensor,
    ) -> torch.Tensor:
        """Return soft degradation probability from paired cloudy/clean data.

        This is the training-time gamma target proposed in P0调整.md. It does
        not depend on an external detector, so cloud-covered pixels cannot be
        missed by a bad inference-time mask while the main bridge is trained.
        """
        diff = (x_cloudy - x_clean).float()
        if self.gamma_delta_reduction == "rms":
            delta = torch.sqrt((diff ** 2).mean(dim=1, keepdim=True) + 1e-12)
        elif self.gamma_delta_reduction == "mean_abs":
            delta = diff.abs().mean(dim=1, keepdim=True)
        else:
            delta = torch.sqrt((diff ** 2).sum(dim=1, keepdim=True) + 1e-12)

        prob = (delta / self.gamma_delta_tau).clamp(0.0, 1.0)
        if self.gamma_smooth_kernel > 1:
            k_size = self.gamma_smooth_kernel
            if k_size % 2 == 0:
                k_size += 1
            prob = F.avg_pool2d(
                prob,
                kernel_size=k_size,
                stride=1,
                padding=k_size // 2,
            )
        return prob.to(dtype=x_clean.dtype)

    def _residual_detail_ramp(self, batch: Dict) -> float:
        """Return the delayed linear ramp for Haar-HF and FFT terms."""
        step = int(batch.get("global_step", 0))
        if step < self.residual_detail_start_step:
            return 0.0
        if self.residual_detail_warmup_steps == 0:
            return 1.0
        progress = (
            step - self.residual_detail_start_step
        ) / float(self.residual_detail_warmup_steps)
        return min(max(progress, 0.0), 1.0)

    @staticmethod
    def _haar2d(x: torch.Tensor):
        """One-level orthonormal 2-D Haar transform for BCHW tensors.

        Odd spatial dimensions are padded by replicating the last row/column,
        keeping the loss usable for arbitrary training crops.
        """
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]
        ll = 0.5 * (x00 + x01 + x10 + x11)
        lh = 0.5 * (-x00 - x01 + x10 + x11)
        hl = 0.5 * (-x00 + x01 - x10 + x11)
        hh = 0.5 * (x00 - x01 - x10 + x11)
        return ll, lh, hl, hh

    def _residual_multidomain_losses(
        self,
        endpoint: torch.Tensor,
        x_clean: torch.Tensor,
        x_cloudy: torch.Tensor,
        gamma_train=None,
        include_detail: bool = True,
    ):
        """Return per-sample Haar-LL, gamma-HF and gamma-FFT losses.

        The supervised signal is the restoration residual rather than the
        complete image, so already-clear content remains governed by the
        existing endpoint and identity terms.
        """
        batch_size = endpoint.shape[0]
        zero = endpoint.new_zeros(batch_size)
        residual_pred = endpoint - x_cloudy
        residual_target = x_clean - x_cloudy

        wavelet_ll_loss = zero
        wavelet_hf_loss = zero
        if (
            self.residual_wavelet_ll_loss_weight > 0.0
            or self.residual_wavelet_hf_loss_weight > 0.0
        ):
            pred_bands = self._haar2d(residual_pred)
            target_bands = self._haar2d(residual_target)
            if self.residual_wavelet_ll_loss_weight > 0.0:
                wavelet_ll_loss = self._get_loss(
                    pred_bands[0], target_bands[0]
                )
            if (
                include_detail
                and self.residual_wavelet_hf_loss_weight > 0.0
            ):
                if gamma_train is None:
                    raise ValueError("gamma_train is required for Haar-HF loss")
                pad_h = gamma_train.shape[-2] % 2
                pad_w = gamma_train.shape[-1] % 2
                gamma_for_haar = gamma_train
                if pad_h or pad_w:
                    gamma_for_haar = F.pad(
                        gamma_for_haar,
                        (0, pad_w, 0, pad_h),
                        mode="replicate",
                    )
                gamma_half = F.avg_pool2d(
                    gamma_for_haar.float(), kernel_size=2, stride=2
                ).to(dtype=endpoint.dtype)
                pred_high = torch.cat(pred_bands[1:], dim=1)
                target_high = torch.cat(target_bands[1:], dim=1)
                wavelet_hf_loss = self._get_loss(
                    pred_high,
                    target_high,
                    pixel_weight=gamma_half,
                )

        fft_loss = zero
        if include_detail and self.residual_fft_loss_weight > 0.0:
            if gamma_train is None:
                raise ValueError("gamma_train is required for FFT loss")
            channels = residual_pred.shape[1]
            if self.residual_fft_channels > 0:
                channels = min(self.residual_fft_channels, channels)
            gamma_float = gamma_train.float()
            weighted_error = (
                residual_pred[:, :channels].float()
                - residual_target[:, :channels].float()
            ) * gamma_float
            spectral_error = torch.fft.rfft2(
                weighted_error,
                norm="ortho",
            )
            spectral_error = spectral_error.abs()
            if self.loss_type == "l2":
                spectral_loss = spectral_error.square()
            elif self.loss_type == "l1":
                spectral_loss = spectral_error
            elif self.loss_type == "charbonnier":
                eps = spectral_error.new_tensor(self.charbonnier_eps)
                # Subtracting eps makes an exact spectral match contribute 0
                # instead of a constant that cannot affect the gradients.
                spectral_loss = (
                    torch.sqrt(spectral_error.square() + eps.square()) - eps
                )
            else:
                raise NotImplementedError(f"Unknown loss_type: {self.loss_type}")
            fft_loss = spectral_loss.reshape(batch_size, -1).mean(dim=1)
            fft_loss = fft_loss.to(dtype=endpoint.dtype)

        return wavelet_ll_loss, wavelet_hf_loss, fft_loss

    def _get_noise_scale(
        self,
        batch: Dict,
        input: torch.Tensor,
        noise_sigma: float,
        mu: torch.Tensor = None,
    ):
        if not self.spatial_noise:
            return noise_sigma

        if self.spatial_noise_source == "degradation":
            if mu is None:
                raise RuntimeError(
                    "spatial_noise_source='degradation' requires the cloudy "
                    "input tensor so gamma_train can be computed from "
                    "|x_cloudy - x_clean|."
                )
            prob = self._get_degradation_prob(input, mu)
            return self._noise_scale_from_prob(prob, noise_sigma)

        cloud_mask = self._get_cloud_mask(batch, input)
        if cloud_mask is None:
            raise RuntimeError(
                "spatial_noise=true requires the configured cloud mask "
                f"batch key {self.cloud_mask_key!r}."
            )
        return self._noise_scale_from_prob(cloud_mask, noise_sigma)

    def _get_head_logits(self, diffusion_model):
        return getattr(diffusion_model, "last_mask_logits", None)

    def _noise_scale_from_logits(
        self,
        logits: torch.Tensor,
        reference: torch.Tensor,
        noise_sigma: float,
    ) -> torch.Tensor:
        prob = self._gamma_prob_from_logits(logits, reference)
        return self._noise_scale_from_prob(prob, noise_sigma)

    def _gamma_prob_from_logits(
        self,
        logits: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if logits is None:
            raise RuntimeError(
                "gamma-head supervision, gamma mixing, or adaptive skip "
                "fusion requires "
                "network_config.params.predict_cloud_mask=true so the "
                "shared gamma head exposes last_mask_logits."
            )
        prob = torch.sigmoid(logits.float()).to(
            device=reference.device,
            dtype=reference.dtype,
        )
        if prob.ndim != reference.ndim:
            raise ValueError(
                f"gamma head logits have shape {tuple(prob.shape)}, expected "
                f"BCHW for reference {tuple(reference.shape)}"
            )
        if prob.shape[0] != reference.shape[0]:
            raise ValueError(
                "gamma head batch size does not match input: "
                f"{prob.shape[0]} vs {reference.shape[0]}"
            )
        if prob.shape[1] != 1:
            prob = prob.max(dim=1, keepdim=True).values
        if prob.shape[-2:] != reference.shape[-2:]:
            prob = F.interpolate(
                prob,
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return prob.clamp(0.0, 1.0)

    def _gamma_mix_probability(self, batch: Dict) -> float:
        if self.gamma_mix_max_prob <= 0.0:
            return 0.0
        if self.gamma_mix_end_step <= self.gamma_mix_start_step:
            return self.gamma_mix_max_prob
        step = int(batch.get("global_step", 0))
        frac = (step - self.gamma_mix_start_step) / float(
            self.gamma_mix_end_step - self.gamma_mix_start_step
        )
        frac = min(max(frac, 0.0), 1.0)
        return self.gamma_mix_max_prob * frac

    def _predict_noise_scale_for_bridge(
        self,
        network,
        denoiser: Denoiser,
        diffusion_model,
        cond: Dict,
        sigma2st: Sigma2St,
        mu: torch.Tensor,
        noise_sigma: float,
        additional_model_inputs: Dict,
    ) -> torch.Tensor:
        sigma_max = float(getattr(self.discretization, "sigma_max", 1.0))
        sigma = mu.new_full((mu.shape[0],), sigma_max)
        st = sigma2st(sigma)
        with torch.no_grad():
            _ = denoiser(
                network,
                mu,
                sigma,
                cond,
                st,
                **additional_model_inputs,
            )
        logits = self._get_head_logits(diffusion_model)
        return self._noise_scale_from_logits(logits, mu, noise_sigma).detach()

    def _mix_with_predicted_noise_scale(
        self,
        train_scale,
        pred_scale: torch.Tensor,
        batch: Dict,
        reference: torch.Tensor,
    ):
        mix_prob = self._gamma_mix_probability(batch)
        if mix_prob <= 0.0:
            return train_scale
        if not torch.is_tensor(train_scale):
            train_scale = reference.new_full(
                (reference.shape[0], 1, 1, 1),
                float(train_scale),
            )
        mask = torch.rand(
            reference.shape[0],
            1,
            1,
            1,
            device=reference.device,
        ) < mix_prob
        return torch.where(mask, pred_scale, train_scale)

    def _gamma_head_loss(
        self,
        pred_scale: torch.Tensor,
        target_scale,
        batch_size: int,
    ) -> torch.Tensor:
        if not torch.is_tensor(target_scale):
            target_scale = pred_scale.new_full(pred_scale.shape, float(target_scale))
        elif target_scale.shape != pred_scale.shape:
            target_scale = target_scale.expand_as(pred_scale)
        return (pred_scale - target_scale.detach()).abs().reshape(
            batch_size, -1
        ).mean(dim=1)

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
        adaptive_skip_enabled = bool(
            getattr(diffusion_model, "adaptive_skip_fusion", False)
        )
        head_loss_weight = (
            self.cloud_mask_pred_loss_weight + self.gamma_head_loss_weight
        )
        if mask_head_enabled and head_loss_weight <= 0.0:
            raise ValueError(
                "predict_cloud_mask=true requires "
                "cloud_mask_pred_loss_weight > 0 or "
                "gamma_head_loss_weight > 0."
            )
        if head_loss_weight > 0.0 and not mask_head_enabled:
            raise ValueError(
                "cloud_mask_pred_loss_weight or gamma_head_loss_weight requires "
                "predict_cloud_mask=true."
            )
        if adaptive_skip_enabled and not mask_head_enabled:
            raise ValueError(
                "adaptive_skip_fusion=true requires predict_cloud_mask=true."
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

        train_noise_scale = self._get_noise_scale(
            batch,
            input,
            current_noise_sigma,
            mu,
        )
        noise_scale = train_noise_scale
        mix_prob = self._gamma_mix_probability(batch)
        gamma_head_logits = None
        pred_gamma = None
        pred_noise_scale = None
        if (
            self.gamma_head_loss_weight > 0.0
            or adaptive_skip_enabled
            or (
                self.spatial_noise
                and self.spatial_noise_source == "degradation"
                and mix_prob > 0.0
            )
        ):
            if self.gamma_head_loss_weight > 0.0 or adaptive_skip_enabled:
                sigma_max = float(getattr(self.discretization, "sigma_max", 1.0))
                gamma_sigma = mu.new_full((mu.shape[0],), sigma_max)
                gamma_st = sigma2st(gamma_sigma)
                prepass_inputs = dict(additional_model_inputs)
                prepass_inputs.pop("skip_gamma", None)
                with torch.set_grad_enabled(self.gamma_head_loss_weight > 0.0):
                    _ = denoiser(
                        network,
                        mu,
                        gamma_sigma,
                        cond,
                        gamma_st,
                        **prepass_inputs,
                    )
                gamma_head_logits = self._get_head_logits(diffusion_model)
                pred_gamma = self._gamma_prob_from_logits(
                    gamma_head_logits,
                    input,
                )
                pred_noise_scale = self._noise_scale_from_prob(
                    pred_gamma,
                    current_noise_sigma,
                )
            else:
                pred_noise_scale = self._predict_noise_scale_for_bridge(
                    network,
                    denoiser,
                    diffusion_model,
                    cond,
                    sigma2st,
                    mu,
                    current_noise_sigma,
                    additional_model_inputs,
                )
            if mix_prob > 0.0 and not adaptive_skip_enabled:
                noise_scale = self._mix_with_predicted_noise_scale(
                    train_noise_scale,
                    pred_noise_scale.detach(),
                    batch,
                    input,
                )

        main_model_inputs = dict(additional_model_inputs)
        if adaptive_skip_enabled:
            gamma_train_for_skip = self._get_degradation_prob(input, mu).detach()
            skip_gamma = gamma_train_for_skip
            if mix_prob > 0.0:
                if pred_gamma is None or pred_noise_scale is None:
                    raise RuntimeError(
                        "adaptive skip Stage-C mixing requires a gamma-head "
                        "prepass."
                    )
                use_predicted = torch.rand(
                    batch_size,
                    1,
                    1,
                    1,
                    device=device,
                ) < mix_prob
                skip_gamma = torch.where(
                    use_predicted,
                    pred_gamma.detach(),
                    gamma_train_for_skip,
                )
                if (
                    self.spatial_noise
                    and self.spatial_noise_source == "degradation"
                ):
                    if not torch.is_tensor(train_noise_scale):
                        train_noise_scale = input.new_full(
                            (batch_size, 1, 1, 1),
                            float(train_noise_scale),
                        )
                    noise_scale = torch.where(
                        use_predicted,
                        pred_noise_scale.detach(),
                        train_noise_scale,
                    )
            main_model_inputs["skip_gamma"] = skip_gamma.detach()
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
            **main_model_inputs,
        )
        velocity_student = denoiser(
            network,
            x_current,
            sigma_current,
            cond,
            st_current,
            **main_model_inputs,
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

        residual_detail_ramp = self._residual_detail_ramp(batch)
        residual_ll_enabled = self.residual_wavelet_ll_loss_weight > 0.0
        residual_detail_enabled = residual_detail_ramp > 0.0 and (
            self.residual_wavelet_hf_loss_weight > 0.0
            or self.residual_fft_loss_weight > 0.0
        )
        if residual_ll_enabled or residual_detail_enabled:
            # Use the paired-degradation gamma directly. Do not recover it
            # from train_noise_scale: that value also contains noise_sigma,
            # the ramp and the non-zero noise floor.
            gamma_train = (
                self._get_degradation_prob(input, mu).detach()
                if residual_detail_enabled
                else None
            )
            (
                residual_wavelet_ll_loss,
                residual_wavelet_hf_loss,
                residual_fft_loss,
            ) = self._residual_multidomain_losses(
                endpoint_student,
                input,
                mu,
                gamma_train,
                include_detail=residual_detail_enabled,
            )
        else:
            residual_wavelet_ll_loss = endpoint_student.new_zeros(batch_size)
            residual_wavelet_hf_loss = endpoint_student.new_zeros(batch_size)
            residual_fft_loss = endpoint_student.new_zeros(batch_size)

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

        if self.gamma_head_loss_weight > 0.0:
            mask_logits = (
                gamma_head_logits
                if gamma_head_logits is not None
                else self._get_head_logits(diffusion_model)
            )
            pred_noise_scale = self._noise_scale_from_logits(
                mask_logits,
                input,
                current_noise_sigma,
            )
            gamma_head_loss = self._gamma_head_loss(
                pred_noise_scale,
                train_noise_scale,
                batch_size,
            )
        else:
            gamma_head_loss = endpoint_student.new_zeros(batch_size)

        consistency_ramp = self._consistency_ramp(batch)
        return (
            consistency_ramp * self.endpoint_loss_weight * endpoint_loss
            + consistency_ramp
            * self.consistency_loss_weight
            * velocity_consistency_loss
            + self.velocity_anchor_loss_weight * velocity_anchor_loss
            + self.clean_endpoint_loss_weight * clean_endpoint_loss
            + self.ssim_endpoint_loss_weight * ssim_loss
            + self.residual_wavelet_ll_loss_weight * residual_wavelet_ll_loss
            + residual_detail_ramp
            * self.residual_wavelet_hf_loss_weight
            * residual_wavelet_hf_loss
            + residual_detail_ramp
            * self.residual_fft_loss_weight
            * residual_fft_loss
            + self.non_cloud_identity_loss_weight * identity_loss
            + self.cloud_mask_pred_loss_weight * mask_pred_loss
            + self.gamma_head_loss_weight * gamma_head_loss
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
