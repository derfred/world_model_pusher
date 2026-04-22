"""Gym-compatible pushing environment wrapping SceneBuilder."""

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import mujoco  # type: ignore[import-untyped]
import numpy as np
from gymnasium import spaces

from .scene_builder import SceneBuilder
from .scene_generator import SceneGenerator
from .scene_config import SceneConfig

Observation = dict[str, Any]


class Controller:
  def reset(self, model, config: SceneConfig) -> None:
    self.model   = model
    self.config  = config
    self.ik_data = mujoco.MjData(model)

    self.ee_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in config.joint_names
    ]
    self.arm_qpos_adr = np.array([model.jnt_qposadr[jid] for jid in joint_ids])
    self.arm_dof_idx  = np.array([model.jnt_dofadr[jid]  for jid in joint_ids])

    self.arm_ctrl_idx = np.array([
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
      for n in config.actuator_names
    ])
    self.arm_ctrl_range = model.actuator_ctrlrange[self.arm_ctrl_idx].copy()  # (n_joints, 2)

  def get_ee_pos(self, data) -> np.ndarray:
    return cast(np.ndarray, data.site_xpos[self.ee_sid].copy().astype(np.float32))

  def get_ee_quat(self, data) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.site_xmat[self.ee_sid])
    return cast(np.ndarray, quat.astype(np.float32))

  def get_arm_qpos(self, data) -> np.ndarray:
    return cast(np.ndarray, data.qpos[self.arm_qpos_adr].copy().astype(np.float32))

  def reset_initial_qpos(self, data, qpos) -> None:
    """Set arm joint positions and matching control setpoints so the
    position actuators hold the initial pose instead of driving back to zero."""
    qpos = np.asarray(qpos, dtype=np.float64)
    data.qpos[self.arm_qpos_adr] = qpos
    data.qvel[self.arm_dof_idx]  = 0.0
    data.ctrl[self.arm_ctrl_idx] = qpos
    mujoco.mj_forward(self.model, data)

  def ik_for_ee_pos(self, target_xyz, qpos) -> np.ndarray:
    d = self.ik_data
    d.qpos[:] = qpos
    q = qpos[self.arm_qpos_adr].copy()

    lam_sq    = 0.05 ** 2
    max_dq    = 0.3  # radians per iteration, cap to prevent runaway
    _jac_pos  = np.zeros((3, self.model.nv))
    _eye3     = np.eye(3)
    converged = False
    err = np.array([np.inf, np.inf, np.inf])

    for _ in range(20):
        mujoco.mj_forward(self.model, d)
        ee = d.site_xpos[self.ee_sid]
        if not np.all(np.isfinite(ee)):
            raise RuntimeError(f"IK diverged (non-finite EE pos), last q={q}")
        err = target_xyz - ee
        if np.linalg.norm(err) < 1e-3:
            converged = True
            break
        mujoco.mj_jacSite(self.model, d, _jac_pos, None, self.ee_sid)
        J = _jac_pos[:, self.arm_dof_idx]
        dq = J.T @ np.linalg.solve(J @ J.T + lam_sq * _eye3, err)

        # Cap step size — the big win against runaway.
        dq_norm = np.linalg.norm(dq)
        if dq_norm > max_dq:
            dq *= max_dq / dq_norm

        q = q + dq
        d.qpos[self.arm_qpos_adr] = q

    if not converged:
      raise RuntimeError(f"IK did not converge: err={err}, ||err||={np.linalg.norm(err):.4f}")

    return cast(np.ndarray, q)

  def update_arm(self, data, action):
    ctrl = np.clip(
        action.qpos,
        self.arm_ctrl_range[:, 0],
        self.arm_ctrl_range[:, 1],
    )
    data.ctrl[self.arm_ctrl_idx] = ctrl


class PushingEnv(gym.Env):
  """
  MuJoCo pushing environment.

  Action space: Box(3,) — end-effector position deltas [dx, dy, dz].
  Observation space: Dict with keys 'image', 'ee_pos', 'object_pos'.
  """

  metadata = {"render_modes": ["rgb_array"]}

  def __init__(self, config) -> None:
    super().__init__()
    self.config    = config
    self.builder   = SceneBuilder()
    self.generator = SceneGenerator(config)

    self.model: mujoco.MjModel | None = None
    self.data: mujoco.MjData | None = None
    self.renderer: mujoco.Renderer | None = None
    self.scene: SceneConfig | None = None
    self.controller: Controller = Controller()
    self.step_count: int = 0

    H, W = self.render_size
    self.observation_space = spaces.Dict({
      "image": spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8),
      "ee_pos": spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32),
      "ee_quat": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
      "object_xy": spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32),
    })
    self.action_space = spaces.Box(
      low=np.array([-0.02, -0.02, -0.01], dtype=np.float32),
      high=np.array([0.02, 0.02, 0.01], dtype=np.float32),
      dtype=np.float32,
    )

  @property
  def render_size(self) -> tuple[int, int]:
    """Parse a 'WxH' string and return (render_h, render_w) as expected by PushingEnv."""
    w, h = self.config.sim.render_size.lower().split("x")
    return int(h), int(w)

  def generate_scene(self) -> SceneConfig:
    """Generate a new random scene config."""
    return self.generator.sample()

  # ------------------------------------------------------------------
  # Gym API
  # ------------------------------------------------------------------

  def reset(  # type: ignore[override]
      self,
      *,
      scene: SceneConfig | None = None,
      seed: int | None = None
  ) -> tuple[Observation, dict[str, Any]]:
    assert scene is not None, "Must provide a SceneConfig to reset()."
    super().reset(seed=seed)

    self.scene = scene
    self.model  = self.builder.build(scene, render_size=self.render_size)
    self.data   = mujoco.MjData(self.model)
    self.controller.reset(self.model, scene)

    if self.renderer is not None:
      self.renderer.close()
    self.renderer = mujoco.Renderer(self.model, self.render_size[0], self.render_size[1])
    mujoco.mj_forward(self.model, self.data)

    initial_qpos = scene.joint_initial_qpos
    if initial_qpos is not None:
      self.controller.reset_initial_qpos(self.data, initial_qpos)

    self.step_count = 0
    return self._get_obs(), {}

  def step(self, action: np.ndarray) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
    assert (
        self.model is not None
        and self.data is not None
        and self.scene is not None
    ), "Call reset() before step()."

    self.controller.update_arm(self.data, action)

    # Step physics
    n_substeps = max(1, int(round(self.scene.control_dt / self.model.opt.timestep)))
    for _ in range(n_substeps):
        mujoco.mj_step(self.model, self.data)

    self.step_count += 1

    obs        = self._get_obs()
    reward     = self._compute_reward()
    terminated = self._check_done()
    truncated  = False
    info: dict[str, Any] = {
        "object_xy": self._get_object_pos(),
        "ee_pos": self.controller.get_ee_pos(self.data),
        "goal_xy": np.array(self.scene.goal_pos, dtype=np.float32),
        "step": self.step_count,
    }
    return obs, reward, terminated, truncated, info

  def render(self) -> np.ndarray | None:  # type: ignore[override]
    if self.renderer is None or self.data is None:
      return None
    self.renderer.update_scene(self.data)
    return np.asarray(self.renderer.render())

  def close(self) -> None:
    if self.renderer is not None:
      self.renderer.close()
      self.renderer = None

  # ------------------------------------------------------------------
  # Internal helpers
  # ------------------------------------------------------------------

  def _get_obs(self) -> Observation:
    assert (
        self.renderer is not None
        and self.data is not None
        and self.scene is not None
    )
    self.renderer.update_scene(self.data)
    image = self.renderer.render()
    return {
        "image": image,
        "ee_pos": self.controller.get_ee_pos(self.data),
        "ee_quat": self.controller.get_ee_quat(self.data),
        "arm_qpos": self.controller.get_arm_qpos(self.data),
        "object_xy": self._get_object_pos(),
        "goal_xy": np.array(self.scene.goal_pos, dtype=np.float32),
        "qpos": self.data.qpos.copy(),
        "step": self.step_count,
        "time": float(self.data.time),
    }

  def _compute_reward(self) -> float:
      assert self.scene is not None
      obj_pos = self._get_object_pos()
      goal_pos = np.array(self.scene.goal_pos, dtype=np.float32)
      distance = float(np.linalg.norm(obj_pos - goal_pos))
      return -distance

  def _check_done(self) -> bool:
      assert self.scene is not None
      obj_pos = self._get_object_pos()
      goal_pos = np.array(self.scene.goal_pos, dtype=np.float32)
      distance = float(np.linalg.norm(obj_pos - goal_pos))
      at_goal = distance < self.scene.goal_tolerance
      timeout = self.step_count >= self.scene.max_steps
      return at_goal or timeout

  def _get_object_pos(self) -> np.ndarray:
    assert self.model is not None and self.data is not None
    body_id = mujoco.mj_name2id(
        self.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "target_object")
    return np.asarray(
        self.data.xpos[body_id][:2].copy(), dtype=np.float32)
