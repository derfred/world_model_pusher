"""Data collection utilities: EpisodeWriter and random_push_policy."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .scene_config import SceneConfig


# ---------------------------------------------------------------------------
# EpisodeWriter
# ---------------------------------------------------------------------------

class EpisodeWriter:
  """
  Writes episodes to HDF5 files.

  Each episode is stored in ``output_dir/episode_NNNNN.hdf5`` with the
  structure::

      images      (T, H, W, 3)  uint8
      actions     (T, 3)        float32
      rewards     (T,)          float32
      metadata/
          config  scalar string (JSON)
          seed    scalar int64
          source  scalar string
  """

  def __init__(self, output_dir: str, format: str = "hdf5") -> None:
    if format != "hdf5":
      raise ValueError(f"Unsupported format '{format}'. Only 'hdf5' is supported.")
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self._ep_count = self._count_existing_episodes()

  def _count_existing_episodes(self) -> int:
    return len(list(self.output_dir.glob("episode_*.hdf5")))

  def write_episode(
    self,
    episode: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
  ) -> Path:
    """
    Persist one episode.

    Parameters
    ----------
    episode:
        List of dicts with keys ``image`` (H,W,3 uint8), ``action`` (3,),
        ``reward`` (float).
    metadata:
        Optional dict. May include ``config`` (SceneConfig or dict),
        ``seed`` (int), ``source`` (str).

    Returns
    -------
    Path to the written file.
    """
    if not episode:
      raise ValueError("episode must not be empty")

    images = np.stack([s["image"] for s in episode], axis=0).astype(np.uint8)
    actions = np.stack([np.asarray(s["action"], dtype=np.float32) for s in episode])
    rewards = np.array([float(s["reward"]) for s in episode], dtype=np.float32)

    ep_path = self.output_dir / f"episode_{self._ep_count:05d}.hdf5"
    with h5py.File(ep_path, "w") as f:
      f.create_dataset("images", data=images, compression="gzip", compression_opts=4)
      f.create_dataset("actions", data=actions)
      f.create_dataset("rewards", data=rewards)

      meta_grp = f.create_group("metadata")
      if metadata is not None:
        cfg = metadata.get("config")
        if cfg is not None:
          if not isinstance(cfg, (str, bytes)):
            cfg = json.dumps(cfg if isinstance(cfg, dict) else asdict(cfg))  # type: ignore[arg-type]
          meta_grp.create_dataset("config", data=cfg)
        seed = metadata.get("seed", -1)
        meta_grp.create_dataset("seed", data=int(seed))
        source = metadata.get("source", "sim")
        meta_grp.create_dataset("source", data=str(source))

    self._ep_count += 1
    return ep_path


# ---------------------------------------------------------------------------
# Random push policy
# ---------------------------------------------------------------------------

def random_push_policy(
  obs: dict[str, np.ndarray],
  config: SceneConfig,
  rng: np.random.Generator | None = None,
) -> np.ndarray:
  """
  A simple heuristic push policy.

  Phase 1 — approach: move the end-effector toward the object.
  Phase 2 — push: once close, move toward the goal.

  Returns a (3,) float32 action [dx, dy, dz].
  """
  if rng is None:
    rng = np.random.default_rng()

  ee_pos = obs["ee_pos"].astype(np.float64)         # (3,)
  obj_pos_xy = obs["object_pos"].astype(np.float64)  # (2,)
  goal_xy = np.array(config.goal_pos, dtype=np.float64)

  ee_xy = ee_pos[:2]

  dist_to_obj = float(np.linalg.norm(ee_xy - obj_pos_xy))
  approach_threshold = 0.06

  if dist_to_obj > approach_threshold:
    # Phase 1: approach the object
    direction = obj_pos_xy - ee_xy
    direction /= (np.linalg.norm(direction) + 1e-8)
    dx, dy = direction * 0.018
  else:
    # Phase 2: push toward goal
    direction = goal_xy - obj_pos_xy
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
      direction /= norm
    dx, dy = direction * 0.018

  # Add small noise
  dx += float(rng.uniform(-0.003, 0.003))
  dy += float(rng.uniform(-0.003, 0.003))

  dx = float(np.clip(dx, -0.02, 0.02))
  dy = float(np.clip(dy, -0.02, 0.02))
  dz = 0.0  # keep on surface

  return np.array([dx, dy, dz], dtype=np.float32)
