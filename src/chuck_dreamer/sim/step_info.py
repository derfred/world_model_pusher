"""Per-step information needed to (re)compute reward and inspect episodes.

`StepInfo` is the dataclass form used at the env/collector boundary.
Writers and loaders flatten it into named columns; the buffer stores the
columns directly and reconstructs `StepInfo` only as needed for reward
recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_COLUMNS = ("object_xy", "ee_pos", "ee_quat", "step", "time")


@dataclass
class StepInfo:
  object_xy: np.ndarray  # (2,)
  ee_pos:    np.ndarray  # (3,)
  ee_quat:   np.ndarray  # (4,)
  goal_xy:   np.ndarray  # (2,) — held constant within an episode (scene-scalar)
  step:      int
  time:      float

  def to_columns(self) -> dict[str, np.ndarray]:
    """Per-step columns suitable for stacking. ``goal_xy`` lives in episode metadata."""
    return {
      "object_xy": np.asarray(self.object_xy, dtype=np.float32),
      "ee_pos":    np.asarray(self.ee_pos,    dtype=np.float32),
      "ee_quat":   np.asarray(self.ee_quat,   dtype=np.float32),
      "step":      np.asarray(self.step, dtype=np.int32),
      "time":      np.asarray(self.time, dtype=np.float32),
    }


def stack_step_infos(items: list[StepInfo]) -> dict[str, np.ndarray]:
  """Stack a list of per-step StepInfo into a dict of T-stacked arrays."""
  if not items:
    raise ValueError("cannot stack empty step-info list")
  return {
    "object_xy": np.stack([s.object_xy for s in items], axis=0).astype(np.float32),
    "ee_pos":    np.stack([s.ee_pos    for s in items], axis=0).astype(np.float32),
    "ee_quat":   np.stack([s.ee_quat   for s in items], axis=0).astype(np.float32),
    "step":      np.asarray([s.step for s in items], dtype=np.int32),
    "time":      np.asarray([s.time for s in items], dtype=np.float32),
  }
