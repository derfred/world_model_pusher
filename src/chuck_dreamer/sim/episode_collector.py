"""Headless episode collection — env-agnostic of policy internals.

Used both by ``generate-scenes`` (offline collection) and by the
trainer's collect phase. The collector maps:

  reset() → SceneConfig
  run()   → (RawEpisode | None, outcome)

It does not poll ``policy.is_done`` or ``policy.state``: termination is
the env's job. ``outcome ∈ {"done", "terminated", "timeout", "crashed"}``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .scene_config import SceneConfig
from .step_info import StepInfo, stack_step_infos

if TYPE_CHECKING:
  from ..policy import Policy
  from .pushing_env import PushingEnv

logger = logging.getLogger(__name__)


RawEpisode = dict[str, Any]


_SHARED_KEYS = (
  "image", "reward", "timestamp",
  "joint_qpos", "ee_pos", "ee_quat", "object_xy",
)


def _stack_steps(steps: list[dict[str, Any]], action_kind: str) -> RawEpisode:
  """Stack per-step records into T-stacked arrays, plus the action under its kind name."""
  out: RawEpisode = {
    k: np.stack([np.asarray(s[k]) for s in steps], axis=0) for k in _SHARED_KEYS
  }
  out[action_kind] = np.stack([np.asarray(s["action"]) for s in steps], axis=0)
  out["step_info"] = stack_step_infos([s["step_info"] for s in steps])
  return out


class EpisodeCollector:
  """Drive any ``Policy`` in any ``PushingEnv``; record one episode."""

  def __init__(self, env: "PushingEnv", policy: "Policy") -> None:
    self.env    = env
    self.policy = policy
    self.scene: SceneConfig | None = None

  def reset(self) -> SceneConfig:
    """Sample a scene, reset env+policy, leave them ready for ``run()``."""
    scene: SceneConfig = self.env.generate_scene()
    self.scene = scene
    self.env.reset(scene=scene)
    self.policy.reset(scene)
    return scene

  def run(self) -> tuple[RawEpisode | None, str]:
    """Roll one episode using ``scene.max_steps`` as the cap.

    Returns ``(episode, outcome)`` where ``episode`` is a ``RawEpisode``
    (dict of T-stacked arrays) or ``None`` if no steps were collected.
    Catches exceptions raised during ``policy.act`` or ``env.step`` (e.g.
    IK non-convergence) and reports ``"crashed"`` with whatever data was
    collected up to that point.
    """
    assert self.scene is not None, "Call reset() before run()."

    act_mode    = self.env.act_mode
    action_kind = "joint_action" if act_mode == "joint" else "ee_action"

    steps: list[dict[str, Any]] = []
    outcome = "timeout"
    try:
      obs, _ = self.env.reset(scene=self.scene)
      for _ in range(self.scene.max_steps):
        # Policies receive the full obs dict. State-mode policies (scripted)
        # read the named keys; modal policies (Dreamer in §6+) project
        # internally via env.policy_obs or by knowing their obs_mode.
        action = self.policy.act(obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        step_info: StepInfo = info["step_info"]
        steps.append({
          "image":      obs["image"],
          "action":     np.asarray(action, dtype=np.float32),
          "reward":     float(reward),
          "timestamp":  float(step_info.time),
          "joint_qpos": np.asarray(next_obs["arm_qpos"], dtype=np.float32),
          "ee_pos":     np.asarray(next_obs["ee_pos"],   dtype=np.float32),
          "ee_quat":    np.asarray(next_obs["ee_quat"],  dtype=np.float32),
          "object_xy":  np.asarray(next_obs["object_xy"], dtype=np.float32),
          "step_info":  step_info,
        })

        obs = next_obs
        if terminated:
          outcome = "done"
          break
        if truncated:
          outcome = "timeout"
          break
    except Exception as e:
      logger.warning("Simulation crashed after %d steps: %s", len(steps), e)
      outcome = "crashed"

    if not steps:
      return None, outcome
    return _stack_steps(steps, action_kind), outcome
