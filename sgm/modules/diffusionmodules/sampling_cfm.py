"""
Consistency Flow Matching sampler.

The default path is true 1-step inference:

    x_init = mu
    v = v_theta(mu, sigma_max, cond)
    x_clean = mu + sigma_max * v

When ``num_steps > 1`` the same trained CFM velocity field is integrated with
Euler steps in sigma space, which provides a quality/speed tradeoff without
requiring a separate sampler module.
"""

from typing import Dict, Union

from omegaconf import ListConfig

from ...util import append_dims, default, instantiate_from_config, tools_scale


class ConsistencyFlowMatchingSampler:
    """Sampler for CFM cloud removal models."""

    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)

        default_guider = {
            "target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"
        }
        self.guider = instantiate_from_config(default(guider_config, default_guider))
        self.verbose = verbose
        self.device = device
        self.sigma2st = None

    def set_sigma2st(self, sigma2st):
        self.sigma2st = sigma2st

    def _get_sigma_gen(self, num_sigmas):
        from tqdm import tqdm

        gen = range(num_sigmas - 1)
        if self.verbose:
            gen = tqdm(gen, total=num_sigmas, desc=f"CFM Sampling ({num_sigmas - 1} steps)")
        return gen

    def _denoise(self, x, denoiser, sigma, cond, st, uc):
        velocity = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), st=st)
        return self.guider(velocity, sigma)

    def _prepare_loop(self, x_randn, mu, cond, uc, num_steps):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps,
            device=self.device,
        )
        uc = default(uc, cond)
        x_init = mu.clone()
        s_in = x_init.new_ones([x_init.shape[0]])
        return x_init, s_in, sigmas, len(sigmas), cond, uc

    def _sampler_step(self, sigma, next_sigma, denoiser, x, mu, cond, uc):
        st = self.sigma2st(sigma)
        v_pred = self._denoise(x, denoiser, sigma, cond, st, uc)
        sigma_bc = append_dims(sigma, x.ndim)
        endpoint = x + sigma_bc * v_pred
        dt = append_dims(next_sigma - sigma, x.ndim)
        x_next = x - v_pred * dt
        return x_next, endpoint

    def __call__(
        self,
        denoiser,
        x,
        mu,
        cond,
        uc=None,
        num_steps=None,
        return_intermediate: bool = False,
        return_denoised: bool = False,
    ):
        n = self.num_steps if num_steps is None else num_steps
        if n == 1:
            return self._one_step(
                denoiser, x, mu, cond, uc, return_intermediate, return_denoised
            )
        return self._multi_step(
            denoiser, x, mu, cond, uc, num_steps, return_intermediate, return_denoised
        )

    def _multi_step(
        self,
        denoiser,
        x,
        mu,
        cond,
        uc=None,
        num_steps=None,
        return_intermediate: bool = False,
        return_denoised: bool = False,
    ):
        x, s_in, sigmas, num_sigmas, cond, uc = self._prepare_loop(
            x, mu, cond, uc, num_steps
        )

        intermediates = []
        denoiseds = []

        for i in self._get_sigma_gen(num_sigmas):
            if return_intermediate:
                intermediates.append(tools_scale(x.clone().detach()))

            x, endpoint = self._sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                mu,
                cond,
                uc,
            )

            if return_denoised:
                denoiseds.append(tools_scale(endpoint.clone().detach()))

        others = {}
        if return_intermediate:
            others["intermediates"] = intermediates
        if return_denoised:
            others["denoiseds"] = denoiseds

        return x, others

    def _one_step(
        self,
        denoiser,
        x,
        mu,
        cond,
        uc=None,
        return_intermediate: bool = False,
        return_denoised: bool = False,
    ):
        sigma_max_scalar = float(getattr(self.discretization, "sigma_max", 1.0))
        sigma_max = mu.new_tensor(sigma_max_scalar)

        uc = default(uc, cond)
        s_in = mu.new_ones([mu.shape[0]])
        sigma = s_in * sigma_max
        x_init = mu.clone()
        st = self.sigma2st(sigma)

        v_pred = self._denoise(x_init, denoiser, sigma, cond, st, uc)
        sigma_bc = append_dims(sigma, x_init.ndim)
        x_clean = x_init + sigma_bc * v_pred

        others = {}
        if return_intermediate:
            others["intermediates"] = [tools_scale(x_init.clone().detach())]
        if return_denoised:
            others["denoiseds"] = [tools_scale(x_clean.clone().detach())]

        return x_clean, others
