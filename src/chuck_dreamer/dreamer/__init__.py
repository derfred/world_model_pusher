"""Model definitions and utilities."""

from .policy import DreamerPolicy

from .mlx_model import DreamerMLXModel


def build_model(config, obs_shape: tuple[int, ...], action_dim: int):
  if config.hardware.device == "mlx":
    return DreamerMLXModel(config, obs_shape=obs_shape, action_dim=action_dim)
  else:
    raise ValueError(f"Unsupported library specified in config: {config.hardware.device}")


__all__ = [
  "build_model",
  "DreamerPolicy"
]
