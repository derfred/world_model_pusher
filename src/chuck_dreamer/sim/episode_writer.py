"""Episode writers — persist sim episodes to HDF5 or Rerun ``.rrd`` files."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ("hdf5", "rerun")


def EpisodeWriter(output_dir: str, format: str = "hdf5"):
    """Factory that returns the concrete writer for the requested ``format``.

    Supported formats: ``hdf5`` (default), ``rerun``.
    """
    if format == "hdf5":
        return HDF5EpisodeWriter(output_dir)
    if format == "rerun":
        return RerunEpisodeWriter(output_dir)
    raise ValueError(
        f"Unsupported format '{format}'. Supported formats: {SUPPORTED_FORMATS}.")


def _serialize_metadata_config(metadata: dict[str, Any] | None) -> str | None:
    if metadata is None:
        return None
    cfg = metadata.get("config")
    if cfg is None:
        return None
    if isinstance(cfg, (str, bytes)):
        return cfg if isinstance(cfg, str) else cfg.decode("utf-8")
    return json.dumps(cfg if isinstance(cfg, dict) else asdict(cfg))  # type: ignore[arg-type]


class HDF5EpisodeWriter:
    """
    Writes episodes to HDF5 files.

    Each episode is stored in ``output_dir/episode_NNNNN.hdf5`` with the
    structure::

        images       (T, H, W, 3)    uint8
        actions      (T, n_joints)   float32
        rewards      (T,)            float32
        timestamps   (T,)            float32   seconds since episode start
        joint_qpos   (T, n_joints)   float32   arm joint positions
        ee_pos       (T, 3)          float32
        ee_quat      (T, 4)          float32
        object_xy    (T, 2)          float32
        metadata/
            config   scalar string (JSON)
            seed     scalar int64
            source   scalar string
            outcome  scalar string   "done" | "terminated" | "timeout" | "crashed"
            goal_xy  (2,) float32
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ep_count = len(list(self.output_dir.glob("episode_*.hdf5")))

    def write_episode(
        self,
        episode: dict[str, np.ndarray],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Persist one episode.

        Parameters
        ----------
        episode:
            Dict of T-stacked arrays with keys ``image`` (T,H,W,3 uint8),
            ``action`` (T,n_joints), ``reward`` (T,), ``timestamp`` (T,),
            ``joint_qpos`` (T,n_joints), ``ee_pos`` (T,3), ``ee_quat`` (T,4),
            ``object_xy`` (T,2).
        metadata:
            Optional dict. May include ``config`` (SceneConfig or dict),
            ``seed`` (int), ``source`` (str).

        Returns
        -------
        Path to the written file.
        """
        if not episode or episode["action"].shape[0] == 0:
            raise ValueError("episode must not be empty")

        actions    = np.asarray(episode["action"],     dtype=np.float32)
        rewards    = np.asarray(episode["reward"],     dtype=np.float32)
        timestamps = np.asarray(episode["timestamp"],  dtype=np.float32)
        joint_qpos = np.asarray(episode["joint_qpos"], dtype=np.float32)
        ee_pos     = np.asarray(episode["ee_pos"],     dtype=np.float32)
        ee_quat    = np.asarray(episode["ee_quat"],    dtype=np.float32)
        object_xy  = np.asarray(episode["object_xy"],  dtype=np.float32)
        images     = np.asarray(episode["image"],      dtype=np.uint8)

        ep_path = self.output_dir / f"episode_{self._ep_count:05d}.hdf5"
        with h5py.File(ep_path, "w") as f:
            f.create_dataset("images",     data=images,  compression="gzip", compression_opts=4)
            f.create_dataset("actions",    data=actions)
            f.create_dataset("rewards",    data=rewards)
            f.create_dataset("timestamps", data=timestamps)
            f.create_dataset("joint_qpos", data=joint_qpos)
            f.create_dataset("ee_pos",     data=ee_pos)
            f.create_dataset("ee_quat",    data=ee_quat)
            f.create_dataset("object_xy",  data=object_xy)

            meta_grp = f.create_group("metadata")
            if metadata is not None:
                cfg = _serialize_metadata_config(metadata)
                if cfg is not None:
                    meta_grp.create_dataset("config", data=cfg)
                seed = metadata.get("seed", -1)
                meta_grp.create_dataset("seed", data=int(seed))
                source = metadata.get("source", "sim")
                meta_grp.create_dataset("source", data=str(source))
                outcome = metadata.get("outcome")
                if outcome is not None:
                    meta_grp.create_dataset("outcome", data=str(outcome))
                goal_xy = metadata.get("goal_xy")
                if goal_xy is not None:
                    meta_grp.create_dataset(
                        "goal_xy",
                        data=np.asarray(goal_xy, dtype=np.float32))

        self._ep_count += 1
        return ep_path


class RerunEpisodeWriter:
    """
    Writes episodes to Rerun ``.rrd`` files (one per episode).

    Each episode is stored in ``output_dir/episode_NNNNN.rrd``. Images are
    logged as ``camera/image``; scalar/vector signals use
    the per-component ``Scalars`` archetype. Metadata is attached on the
    recording properties entity so it is visible in the viewer.
    """

    def __init__(self, output_dir: str) -> None:
        import rerun as rr  # noqa: F401  — fail fast if rerun is missing
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ep_count = len(list(self.output_dir.glob("episode_*.rrd")))

    def write_episode(
        self,
        episode: dict[str, np.ndarray],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        if not episode or episode["action"].shape[0] == 0:
            raise ValueError("episode must not be empty")

        import rerun as rr

        ep_path = self.output_dir / f"episode_{self._ep_count:05d}.rrd"
        rec = rr.RecordingStream(
            application_id="chuck_dreamer",
            recording_id=f"episode_{self._ep_count:05d}",
        )

        if metadata is not None:
            cfg_json = _serialize_metadata_config(metadata)
            props: dict[str, Any] = {}
            if cfg_json is not None:
                props["config"] = cfg_json
            if "seed" in metadata:
                props["seed"] = str(int(metadata["seed"]))
            if "source" in metadata:
                props["source"] = str(metadata["source"])
            if metadata.get("outcome") is not None:
                props["outcome"] = str(metadata["outcome"])
            if metadata.get("goal_xy") is not None:
                goal = np.asarray(metadata["goal_xy"], dtype=np.float32)
                props["goal_xy"] = f"[{float(goal[0])}, {float(goal[1])}]"
            for key, value in props.items():
                rec.log(f"metadata/{key}", rr.TextDocument(value), static=True)

        images     = np.asarray(episode["image"],      dtype=np.uint8)
        actions    = np.asarray(episode["action"],     dtype=np.float32)
        rewards    = np.asarray(episode["reward"],     dtype=np.float32)
        timestamps = np.asarray(episode["timestamp"],  dtype=np.float32)
        joint_qpos = np.asarray(episode["joint_qpos"], dtype=np.float32)
        ee_pos     = np.asarray(episode["ee_pos"],     dtype=np.float32)
        ee_quat    = np.asarray(episode["ee_quat"],    dtype=np.float32)
        object_xy  = np.asarray(episode["object_xy"],  dtype=np.float32)

        T = actions.shape[0]
        for i in range(T):
            rec.set_time("step", sequence=i)
            rec.set_time("time", duration=float(timestamps[i]))

            rec.log("camera/image",  rr.Image(images[i]))
            rec.log("action",        rr.Scalars(actions[i].tolist()))
            rec.log("reward",        rr.Scalars(float(rewards[i])))
            rec.log("joint_qpos",    rr.Scalars(joint_qpos[i].tolist()))
            rec.log("ee_pos",        rr.Scalars(ee_pos[i].tolist()))
            rec.log("ee_quat",       rr.Scalars(ee_quat[i].tolist()))
            rec.log("object_xy",     rr.Scalars(object_xy[i].tolist()))

        rec.save(str(ep_path))
        self._ep_count += 1
        return ep_path
