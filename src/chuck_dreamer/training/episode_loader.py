"""Load sim-collected episodes into the replay buffer.

Episodes recorded by :class:`HDF5EpisodeWriter`/:class:`RerunEpisodeWriter`
carry one of ``joint_action`` / ``ee_action`` (selected by ``act_mode``
in metadata), the achieved arm qpos and EE pose per step, and the
object/goal state needed to recompute reward at training time.

The buffer's schema is ``(obs, action, reward, done)`` plus a
``step_info`` block that mirrors :class:`StepInfo`. This module bridges
the two: a small processor abstraction extracts the modal observation;
the last step is dropped so the obs count is T+1 for T actions,
preserving the buffer's invariant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator, Protocol, Union

import h5py  # type: ignore[import-untyped]
import numpy as np


ProgressCallback = Callable[[int, int, Path], None]
Progress = Union[bool, ProgressCallback]


RawEpisode = dict[str, Any]
Episode = dict[str, Any]


_STEP_INFO_KEYS = ("object_xy", "ee_pos", "ee_quat")


# ---------------------------------------------------------------------------
# Processor abstraction
# ---------------------------------------------------------------------------


class EpisodeProcessor(Protocol):
  """Converts a raw sim episode into replay-buffer schema.

  A processor must return a dict with keys ``obs`` (T+1, *obs_shape),
  ``action`` (T, action_dim), ``reward`` (T,), ``done`` (T,), and
  ``step_info`` (a dict of T+1-length arrays mirroring StepInfo columns).
  """

  def __call__(self, raw: RawEpisode) -> Episode: ...


def _slice_step_info(raw: RawEpisode, n: int) -> dict[str, np.ndarray]:
  """Extract per-step info columns sliced to length ``n``.

  In the recorded format, ``step_info[t]`` describes the state AFTER
  action_t — i.e. it aligns with the buffer's ``reward[t]`` (reward
  earned by taking ``action_t``). Callers pass ``n = T = N - 1`` so the
  result aligns with the action/reward/done axis.
  """
  out: dict[str, np.ndarray] = {}
  for key in _STEP_INFO_KEYS:
    out[key] = np.asarray(raw[key], dtype=np.float32)[:n]
  if "timestamp" in raw:
    out["time"] = np.asarray(raw["timestamp"], dtype=np.float32)[:n]
  return out


def _resolve_action(raw: RawEpisode) -> np.ndarray:
  if "joint_action" in raw:
    return np.asarray(raw["joint_action"], dtype=np.float32)
  if "ee_action" in raw:
    return np.asarray(raw["ee_action"], dtype=np.float32)
  raise KeyError("raw episode missing an action field (joint_action / ee_action)")


def _drop_last_and_pack(obs: np.ndarray, raw: RawEpisode) -> Episode:
  """Align a per-step obs array with the buffer's (T+1 obs, T action) layout.

  Sim episodes record N aligned steps of (obs_t, action_t, reward_t).
  We treat the N recorded obs as ``obs[0..N-1]`` and drop the last
  action/reward, yielding N obs and N-1 actions — i.e. T = N-1 with
  T+1 = N obs. The final ``done`` is set to True. ``step_info`` is
  sliced to length N (matching obs).
  """
  N = obs.shape[0]
  if N < 2:
    raise ValueError(f"episode too short: {N} steps (need >= 2)")

  action = _resolve_action(raw)[: N - 1]
  reward = np.asarray(raw["reward"], dtype=np.float32)[: N - 1]
  done = np.zeros((N - 1,), dtype=bool)
  done[-1] = True

  goal_xy = None
  if "goal_xy" in raw:
    goal_xy = np.asarray(raw["goal_xy"], dtype=np.float32)

  ep: Episode = {
    "obs": obs.astype(obs.dtype, copy=False),
    "action": action,
    "reward": reward,
    "done": done,
    "step_info": _slice_step_info(raw, N - 1),
  }
  if goal_xy is not None:
    ep["goal_xy"] = goal_xy
  return ep


class StateVectorProcessor:
  """Default processor: concat ee_pos + ee_quat + object_xy + joint_qpos.

  Produces a flat float32 observation vector per step. Order is fixed
  so downstream consumers can unpack it consistently.
  """

  def __call__(self, raw: RawEpisode) -> Episode:
    ee_pos = np.asarray(raw["ee_pos"], dtype=np.float32)
    ee_quat = np.asarray(raw["ee_quat"], dtype=np.float32)
    object_xy = np.asarray(raw["object_xy"], dtype=np.float32)
    joint_qpos = np.asarray(raw["joint_qpos"], dtype=np.float32)
    obs = np.concatenate([ee_pos, ee_quat, object_xy, joint_qpos], axis=1)
    return _drop_last_and_pack(obs, raw)


class ImageProcessor:
  """Processor that uses the raw RGB image as the observation (uint8)."""

  def __call__(self, raw: RawEpisode) -> Episode:
    obs = np.asarray(raw["image"], dtype=np.uint8)
    return _drop_last_and_pack(obs, raw)


class ImageProprioProcessor:
  """Returns (image, proprio) per step.

  proprio = ee_pos (3) + ee_quat (4) + joint_qpos (n_joints).
  No object_xy: proprioception is body-internal.
  """

  def __call__(self, raw: RawEpisode) -> Episode:
    image = np.asarray(raw["image"], dtype=np.uint8)
    ee_pos = np.asarray(raw["ee_pos"], dtype=np.float32)
    ee_quat = np.asarray(raw["ee_quat"], dtype=np.float32)
    joint_qpos = np.asarray(raw["joint_qpos"], dtype=np.float32)
    proprio = np.concatenate([ee_pos, ee_quat, joint_qpos], axis=1)

    N = image.shape[0]
    if N < 2:
      raise ValueError(f"episode too short: {N} steps (need >= 2)")

    action = _resolve_action(raw)[: N - 1]
    reward = np.asarray(raw["reward"], dtype=np.float32)[: N - 1]
    done = np.zeros((N - 1,), dtype=bool)
    done[-1] = True

    ep: Episode = {
      "obs": (image, proprio),
      "action": action,
      "reward": reward,
      "done": done,
      "step_info": _slice_step_info(raw, N - 1),
    }
    if "goal_xy" in raw:
      ep["goal_xy"] = np.asarray(raw["goal_xy"], dtype=np.float32)
    return ep


def processor_for(obs_mode: str) -> EpisodeProcessor:
  """Build the processor matching a given env ``obs_mode``."""
  if obs_mode == "state":
    return StateVectorProcessor()
  if obs_mode == "image":
    return ImageProcessor()
  if obs_mode == "image_proprio":
    return ImageProprioProcessor()
  raise ValueError(f"Unknown obs_mode: {obs_mode!r}")


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


# Maps RawEpisode key -> HDF5 dataset name. Action datasets are resolved
# dynamically since exactly one of joint_action / ee_action is present.
_HDF5_DATASET = {
  "image": "images",
  "reward": "rewards",
  "timestamp": "timestamps",
  "joint_qpos": "joint_qpos",
  "ee_pos": "ee_pos",
  "ee_quat": "ee_quat",
  "object_xy": "object_xy",
}


def load_hdf5_episode(path: str | Path) -> RawEpisode:
  """Read one HDF5 episode written by ``HDF5EpisodeWriter``."""
  raw: RawEpisode = {}
  with h5py.File(path, "r") as f:
    for key, dataset in _HDF5_DATASET.items():
      raw[key] = np.asarray(f[dataset][()])

    if "joint_action" in f:
      raw["joint_action"] = np.asarray(f["joint_action"][()])
    elif "ee_action" in f:
      raw["ee_action"] = np.asarray(f["ee_action"][()])
    else:
      raise KeyError(f"{path}: missing action dataset (joint_action or ee_action)")

    if "metadata" in f:
      meta = f["metadata"]
      if "act_mode" in meta:
        am = meta["act_mode"][()]
        raw["act_mode"] = am.decode("utf-8") if isinstance(am, bytes) else str(am)
      if "goal_xy" in meta:
        raw["goal_xy"] = np.asarray(meta["goal_xy"][()], dtype=np.float32)
  return raw


def _collect_chunks_by_entity(recording) -> dict[str, list[dict]]:
  """Group a recording's chunks by entity path, skipping metadata."""
  by_entity: dict[str, list[dict]] = {}
  static: dict[str, dict] = {}
  for chunk in recording.chunks():
    entity = str(chunk.entity_path)
    if chunk.is_static:
      static[entity] = chunk.to_record_batch().to_pydict()
      continue
    if entity.startswith("/__"):
      continue
    by_entity.setdefault(entity, []).append(chunk.to_record_batch().to_pydict())
  by_entity["__static__"] = [static]  # stash for later inspection
  return by_entity


def _ordered_scalar_column(chunk_dicts: list[dict], step_key: str = "step") -> np.ndarray:
  """Flatten per-chunk scalar rows into a single (N, d) array sorted by step."""
  rows: list[tuple[int, np.ndarray]] = []
  for d in chunk_dicts:
    scalars = d["Scalars:scalars"]
    steps = d[step_key]
    for s, v in zip(steps, scalars):
      rows.append((int(s), np.asarray(v, dtype=np.float32)))
  rows.sort(key=lambda x: x[0])
  return np.stack([v for _, v in rows], axis=0)


def load_rerun_episode(path: str | Path) -> RawEpisode:
  """Read one Rerun ``.rrd`` episode written by ``RerunEpisodeWriter``."""
  from rerun.recording import load_recording

  rec = load_recording(str(path))
  by_entity = _collect_chunks_by_entity(rec)

  def _scalars(entity: str) -> np.ndarray:
    if entity not in by_entity:
      raise KeyError(f"entity {entity!r} not found in {path}")
    return _ordered_scalar_column(by_entity[entity])

  raw: RawEpisode = {}
  if "/joint_action" in by_entity:
    raw["joint_action"] = _scalars("/joint_action")
  elif "/ee_action" in by_entity:
    raw["ee_action"] = _scalars("/ee_action")
  else:
    raise KeyError(f"{path}: missing action entity (/joint_action or /ee_action)")

  raw["reward"] = _scalars("/reward").reshape(-1)
  raw["joint_qpos"] = _scalars("/joint_qpos")
  raw["ee_pos"] = _scalars("/ee_pos")
  raw["ee_quat"] = _scalars("/ee_quat")
  raw["object_xy"] = _scalars("/object_xy")

  image_rows: list[tuple[int, np.ndarray]] = []
  time_rows: list[tuple[int, float]] = []
  for d in by_entity["/camera/image"]:
    for s, buf, fmt, t in zip(
      d["step"], d["Image:buffer"], d["Image:format"], d["time"]
    ):
      f0 = fmt[0] if isinstance(fmt, list) else fmt
      w, h = int(f0["width"]), int(f0["height"])
      image_rows.append((int(s), np.asarray(buf, dtype=np.uint8).reshape(h, w, 3)))
      time_rows.append(
        (int(s), t.total_seconds() if hasattr(t, "total_seconds") else float(t))
      )
  image_rows.sort(key=lambda x: x[0])
  time_rows.sort(key=lambda x: x[0])
  raw["image"] = np.stack([img for _, img in image_rows], axis=0)
  raw["timestamp"] = np.asarray([t for _, t in time_rows], dtype=np.float32)

  return raw


# ---------------------------------------------------------------------------
# Directory iteration
# ---------------------------------------------------------------------------


def _resolve_progress(progress: Progress) -> ProgressCallback | None:
  """Turn a ``Progress`` value into a per-file callback (or None)."""
  if progress is False:
    return None
  if callable(progress):
    return progress

  try:
    from tqdm import tqdm  # type: ignore[import-not-found]
  except ImportError:
    def _print_cb(i: int, total: int, path: Path) -> None:
      print(f"[{i}/{total}] {path.name}")

    return _print_cb

  bar: dict[str, Any] = {"tqdm": None}

  def _tqdm_cb(i: int, total: int, path: Path) -> None:
    if bar["tqdm"] is None:
      bar["tqdm"] = tqdm(total=total, unit="ep", desc="episodes")
    bar["tqdm"].set_postfix_str(path.name, refresh=False)
    bar["tqdm"].update(1)
    if i == total:
      bar["tqdm"].close()

  return _tqdm_cb


def iter_episodes(
  directory: str | Path,
  format: str = "hdf5",
  progress: Progress = False,
) -> Iterator[RawEpisode]:
  """Yield raw episodes from a directory of writer output."""
  directory = Path(directory)
  if format == "hdf5":
    paths = sorted(directory.glob("episode_*.hdf5"))
    loader = load_hdf5_episode
  elif format == "rerun":
    paths = sorted(directory.glob("episode_*.rrd"))
    loader = load_rerun_episode
  else:
    raise ValueError(f"unsupported format {format!r}; use 'hdf5' or 'rerun'")

  callback = _resolve_progress(progress)
  total = len(paths)
  for i, p in enumerate(paths, start=1):
    episode = loader(p)
    if callback is not None:
      callback(i, total, p)
    yield episode
