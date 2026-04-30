import numpy as np

from ..policy import Policy, Action


class DreamerPolicy(Policy):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def reset(self, controller, scene):
    pass

  def act(self, obs):
    return Action.from_qpos(np.zeros(6)), None
