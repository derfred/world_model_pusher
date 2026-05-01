"""Configuration management using OmegaConf."""

from omegaconf import DictConfig, OmegaConf
from typing import cast
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


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
