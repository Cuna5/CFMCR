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
