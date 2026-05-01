from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
  from chuck_dreamer.sim.scene_config import SceneConfig


class Policy(Protocol):
  def reset(self, scene: "SceneConfig") -> None: ...
  def act(self, obs: Any) -> np.ndarray: ...
