import numpy as np


class DreamerPolicy:
  def __init__(self, model):
    self.model = model

  def reset(self, scene):
    pass

  def act(self, obs):
    # Stub: returns a zero EE pose action with identity quat. Wired up in §6+.
    return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
