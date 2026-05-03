"""Configuration management using OmegaConf."""

from omegaconf import DictConfig, OmegaConf
from typing import cast
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Derived config helpers (exposed via OmegaConf resolvers below)
# ---------------------------------------------------------------------------


# Spatial dim at the CNN bottleneck (after all encoder strided convs).
# Combined with the per-layer strides this fixes the input resolution:
#   image_size = CNN_BOTTLENECK_HW * prod(strides)
CNN_BOTTLENECK_HW = 4


def derive_image_size(strides: tuple[int, ...]) -> int:
  """Image side length implied by an encoder's per-layer strides.

  Each conv strides spatial dims down and we want the post-encoder feature
  map to be ``CNN_BOTTLENECK_HW`` per side, so the input must be
  ``CNN_BOTTLENECK_HW * prod(strides)`` per side. Padding is chosen
  per-layer in the encoder so each layer divides cleanly.
  """
  prod = 1
  for s in strides:
    prod *= int(s)
  return CNN_BOTTLENECK_HW * prod


# Register custom resolvers so YAML interpolation can compute derived fields.
# Idempotent: ``replace=True`` keeps re-imports during testing from raising.
OmegaConf.register_new_resolver(
  "derive_image_size",
  lambda strides: derive_image_size(tuple(int(s) for s in strides)),
  replace=True,
)


def load_config(config_path: Optional[str] = None) -> DictConfig:
    """
    Load configuration from file or return default config.

    Args:
        config_path: Path to configuration file. If None, returns
            default config.

    Returns:
        Configuration object
    """
    if config_path is not None and Path(config_path).exists():
        config = cast(DictConfig, OmegaConf.load(config_path))
    else:
        config = get_default_config()

    return config


def get_default_config() -> DictConfig:
    """Get default configuration from YAML file."""

    # Try to load from default.yaml first
    default_config_path = Path(
        __file__).parent.parent.parent / "configs" / "default.yaml"
    return cast(DictConfig, OmegaConf.load(default_config_path))


def save_config(config: DictConfig, save_path: str) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration object to save
        save_path: Path where to save the configuration
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, save_path)
    logger.info(f"Configuration saved to {save_path}")


def merge_configs(
        base_config: DictConfig,
        override_config: DictConfig) -> DictConfig:
    """
    Merge two configurations with override taking precedence.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    return cast(DictConfig, OmegaConf.merge(base_config, override_config))


def _drop_none(d: dict) -> dict:
    """Recursively remove keys whose value is None from a nested dict."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            cleaned = _drop_none(v)
            if cleaned:
                out[k] = cleaned
        elif v is not None:
            out[k] = v
    return out


def merge_overrides(
        base_config: DictConfig,
        overrides: dict) -> DictConfig:
    """
    Merge a (possibly nested) dict of overrides into a config, skipping None leaves.

    Args:
        base_config: Base configuration (typically the full loaded config)
        overrides: Nested dict of overrides (e.g. {"sim": {"seed": 42}});
                   keys with None leaf values are ignored

    Returns:
        Merged configuration
    """
    clean = _drop_none(overrides)
    return cast(DictConfig, OmegaConf.merge(base_config, OmegaConf.create(clean)))


if __name__ == "__main__":
    config = get_default_config()
    print(OmegaConf.to_yaml(config))
