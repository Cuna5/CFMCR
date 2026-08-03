from pathlib import Path

import yaml


CONFIG_ROOT = Path(__file__).parent / "configs" / "example_training"
SINGLE_IMAGE_NETWORK = (
    "sgm.modules.diffusionmodules.k_diffusion.image_transformer."
    "ImageTransformerDenoiserModelInterface"
)
V41_KEYS = (
    "adaptive_skip_fusion",
    "adaptive_skip_hidden_channels",
    "adaptive_skip_max_delta",
)
MEANFLOW_ENGINE = "sgm.models.diffusion_meanflow.MeanFlowEngine"
RESIDUAL_ENGINE = "sgm.models.diffusion.ResidualDiffusionEngine"
V42_GAMMA_KEYS = (
    "spatial_noise",
    "noise_sigma_floor",
    "spatial_noise_source",
    "gamma_delta_tau",
    "gamma_delta_reduction",
    "gamma_smooth_kernel",
    "gamma_head_loss_weight",
)


def _load_configs():
    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        with path.open(encoding="utf-8") as handle:
            yield path, yaml.safe_load(handle)


def test_all_example_configs_expose_disabled_fid_switch():
    for path, config in _load_configs():
        assert config["model"]["params"]["enable_fid"] is False, path


def test_single_image_configs_expose_v41_skip_fusion_controls():
    compatible_configs = []
    for path, config in _load_configs():
        network = config["model"]["params"].get("network_config")
        if not network or network.get("target") != SINGLE_IMAGE_NETWORK:
            continue

        compatible_configs.append(path)
        params = network["params"]
        assert all(key in params for key in V41_KEYS), path
        assert params["adaptive_skip_hidden_channels"] == 16, path
        assert params["adaptive_skip_max_delta"] == 0.25, path

        is_noise_bridge_cfm = path.name.endswith("_noise_bridge_cfm.yaml")
        assert params["adaptive_skip_fusion"] is is_noise_bridge_cfm, path
        if is_noise_bridge_cfm:
            assert params["predict_cloud_mask"] is True, path

    assert compatible_configs


def test_v42_meanflow_engine_and_spatial_gamma_scope():
    ordinary = []
    noise_bridge = []
    for path, config in _load_configs():
        if not path.name.endswith("_meanflow.yaml"):
            continue

        model = config["model"]
        params = model["params"]
        if path.name.endswith("_noise_bridge_meanflow.yaml"):
            noise_bridge.append(path)
            network_params = params["network_config"]["params"]
            loss_params = params["loss_fn_config"]["params"]
            sampler_params = params["sampler_config"]["params"]

            assert model["target"] == RESIDUAL_ENGINE, path
            assert network_params["predict_cloud_mask"] is True, path
            assert network_params["adaptive_skip_fusion"] is False, path
            assert all(key in loss_params for key in V42_GAMMA_KEYS), path
            assert loss_params["spatial_noise"] is True, path
            assert loss_params["spatial_noise_source"] == "degradation", path
            assert loss_params["gamma_head_loss_weight"] > 0.0, path
            assert sampler_params["spatial_noise"] is True, path
            assert (
                sampler_params["noise_sigma_floor"]
                == loss_params["noise_sigma_floor"]
            ), path
            assert not any(key.startswith("gamma_mix_") for key in loss_params), path
            assert not any(key.startswith("residual_") for key in loss_params), path
        else:
            ordinary.append(path)
            assert model["target"] == MEANFLOW_ENGINE, path
            assert params["use_ema"] is True, path
            assert "teacher_ema_decay" not in params, path

    assert len(ordinary) == 4
    assert len(noise_bridge) == 4
