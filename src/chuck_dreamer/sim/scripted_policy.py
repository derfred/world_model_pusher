"""Heuristic scripted push policy."""

from __future__ import annotations

import logging
from typing import Any, cast

import mujoco  # type: ignore[import-untyped]
import numpy as np

from .scene_config import SceneConfig
from ..policy import Policy, Action

logger = logging.getLogger(__name__)


class ScriptedPolicy(Policy):
  """
      A simple heuristic push policy.

      State 1 - initial: arm in rest position pointing to the middle of the table
      State 2 — approach: move the end-effector toward the object, maintain a standoff distance.
      State 3 — push: once close, move toward the goal.
      State 4 — done: push complete, hold position.

      Returns a (3,) float32 action [dx, dy, dz].
  """

  _PUSH_Z = 0.075          # EE height — midpoint of typical object side
  _MOVE_SPEED = 0.015
  _LOOKAHEAD  = 0.02   # meters ahead of projected position to command
  _STANDOFF = 0.06         # approach to this distance behind the object
  _CLOSE_THRESH = 0.08      # switch from approach to push phase

  state: str = "initial"  # "initial", "ready", "approach", "push", "done"

  def __init__(self) -> None:
    self.controller: Any = None
    self.start_xyz: np.ndarray | None = None

  @property
  def ready_xy(self) -> np.ndarray:
    base  = np.array(self.scene.robot_base_pos[:2], dtype=np.float64)
    zero  = np.array([0.0, 0.0], dtype=np.float64)
    return cast(np.ndarray, base + (zero - base) * 0.3)

  @property
  def goal_xy(self) -> np.ndarray:
    return np.array(self.scene.goal_pos, dtype=np.float64)

  @property
  def goal_xyz(self) -> np.ndarray:
    return cast(np.ndarray, np.append(self.goal_xy, self._PUSH_Z))

  @property
  def object_xy(self) -> np.ndarray:
    return np.array(self.scene.target.pos[:2], dtype=np.float64)

  @property
  def approach_xy(self) -> np.ndarray:
    push_dir  = self.goal_xy - self.object_xy
    push_dist = np.linalg.norm(push_dir)
    if push_dist > 1e-6:
      push_dir /= push_dist
    else:
      push_dir = np.array([1.0, 0.0])
    approach_point = self.object_xy - push_dir * self._STANDOFF
    return cast(np.ndarray, approach_point)

  @property
  def approach_xyz(self) -> np.ndarray:
    return cast(np.ndarray, np.append(self.approach_xy, self._PUSH_Z))

  def _determine_state(self, obs: dict[str, np.ndarray]) -> str | None:
      if self.state == "initial":
          if np.linalg.norm(obs["ee_pos"][:2] - self.ready_xy) < self._CLOSE_THRESH:
              return "ready"
      elif self.state == "approach":
          if np.linalg.norm(obs["ee_pos"] - self.approach_xyz) < self._CLOSE_THRESH:
              return "push"
      elif self.state == "push":
          if np.linalg.norm(obs["ee_pos"] - self.goal_xyz) < self._CLOSE_THRESH:
              return "done"
      return None

  def _step_to(
      self,
      start_xyz: np.ndarray,
      target_xyz: np.ndarray,
      obs: dict[str, np.ndarray],
  ) -> Action:
    if self.start_xyz is None:
      self.start_xyz = np.asarray(start_xyz, dtype=np.float64).copy()

    start = self.start_xyz
    end   = np.asarray(target_xyz, dtype=np.float64)
    ee    = np.asarray(obs["ee_pos"], dtype=np.float64)

    # Direction and total length of the segment.
    segment = end - start
    seg_len = float(np.linalg.norm(segment))
    if seg_len < 1e-6:
        return Action.from_qpos(self.controller.ik_for_ee_pos(end, obs["qpos"]))
    direction = segment / seg_len

    # Project current EE position onto the segment to find actual progress.
    # This is what makes it robust: if the EE lagged behind or got pushed
    # sideways by contact, we re-reference from where we actually are.
    progress = float(np.dot(ee - start, direction))
    progress = float(np.clip(progress, 0.0, seg_len))

    # Command a point a small lookahead past the projection, capped at the end.
    # Lookahead > step size gives the controller something to chase and
    # smooths out jitter from the projection.
    commanded_progress = min(progress + self._LOOKAHEAD, seg_len)
    commanded = start + direction * commanded_progress

    return Action.from_qpos(self.controller.ik_for_ee_pos(commanded, obs["qpos"]))

  def _act_initial(self, obs: dict[str, np.ndarray]) -> Action:
    target = np.append(self.ready_xy, self._PUSH_Z)
    return self._step_to(obs["ee_pos"], target, obs)

  def _act_ready(self, obs: dict[str, np.ndarray]) -> Action:
    return Action.from_qpos(obs["arm_qpos"])

  def _act_approach(self, obs: dict[str, np.ndarray]) -> Action:
    return self._step_to(obs["ee_pos"], self.approach_xyz, obs)

  def _act_push(self, obs: dict[str, np.ndarray]) -> Action:
    return self._step_to(self.approach_xyz, self.goal_xyz, obs)

  def _act_done(self, obs: dict[str, np.ndarray]) -> Action:
    return Action.from_qpos(obs["arm_qpos"])

  def reset(self, controller, scene: SceneConfig) -> None:
    self.controller = controller
    self.scene      = scene
    self.state      = "initial"
    self.start_xyz  = None

  def act(self, obs: dict[str, np.ndarray]) -> tuple[Action, str | None]:
    cur_state  = self.state
    next_state = self._determine_state(obs)
    changed    = next_state is not None and next_state != cur_state
    if changed:
      assert next_state is not None
      self.start_xyz = None
      self.state     = next_state

    action = getattr(self, f"_act_{self.state}")(obs)
    return action, (cur_state if changed else None)

  def is_done(self) -> bool:
    return self.state == "done"

  def insert_hints(self, viewer: mujoco.Viewer, rgba: np.ndarray = np.array([0.0, 1.0, 0.0, 0.8], dtype=np.float32)) -> None:
    """Insert custom geoms into the scene for visualization hints."""
    if self.state != "ready":
      return

    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
      return

    assert self.scene is not None

    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
      g,
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=np.array([0.02, 0.0, 0.0], dtype=np.float64),
      pos=self.goal_xyz,
      mat=np.eye(3).flatten(),
      rgba=rgba,
    )
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    viewer.user_scn.ngeom += 1

    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
      return

    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
      g,
      type=mujoco.mjtGeom.mjGEOM_ARROW,
      size=np.zeros(3),
      pos=np.zeros(3),
      mat=np.eye(3).flatten(),
      rgba=rgba,
    )
    mujoco.mjv_connector(
      g,
      mujoco.mjtGeom.mjGEOM_ARROW,
      0.01,
      self.approach_xyz,
      self.goal_xyz,
    )
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    viewer.user_scn.ngeom += 1

  def advance_from_ready(self) -> None:
    if self.state == "ready":
      self.state = "approach"
