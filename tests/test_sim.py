"""Tests for the MuJoCo pushing simulation."""

from __future__ import annotations

import json

import h5py
import mujoco
import numpy as np
import pytest
from omegaconf import OmegaConf

from chuck_dreamer.sim import (
    CameraConfig,
    EpisodeWriter,
    LightingConfig,
    ObjectConfig,
    PushingEnv,
    ScriptedPolicy,
    SceneBuilder,
    SceneConfig,
    SceneGenerator,
    ScenePlayer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_TABLE_SIZE = [0.60, 0.5, 0.02]


def _make_simple_config(robot_type: str = "so100") -> SceneConfig:
  """Return a minimal valid SceneConfig. Uses so100 arm by default since
  PushingEnv/Controller require an arm with named actuators and ee_site."""
  return SceneConfig(
      table_size=[0.30, 0.25, 0.02],
      table_friction=0.5,
      table_color=[0.6, 0.5, 0.4, 1.0],
      robot_type=robot_type,
      robot_base_pos=[-0.30, 0.0, 0.04],
      robot_initial_qpos=None,
      target=ObjectConfig(
          shape="box",
          size=[0.03, 0.03, 0.03],
          mass=0.1,
          friction=0.5,
          pos=[0.05, 0.0, 0.07],
          orientation=0.0,
          color=[1.0, 0.0, 0.0, 1.0],
      ),
      goal_pos=[0.10, 0.05],
      goal_tolerance=0.04,
      obstacles=[],
      clutter=[],
      camera=CameraConfig(pos=[0.0, -0.40, 0.55], look_at=[0.0, 0.0, 0.04], fov=60.0),
      lighting=LightingConfig(
          direction=[0.0, -0.5, -1.0], intensity=0.8, ambient=0.3),
      max_steps=50,
      control_dt=0.1,
  )


def _make_env_cfg(
    difficulty: str = "easy",
    render_size: str = "64x64",
    seed: int = 42,
    obs_mode: str = "state",
    act_mode: str = "joint",
):
  """Build a DictConfig that PushingEnv / SceneGenerator accept."""
  return OmegaConf.create({
      "seed": seed,
      "sim": {
          "difficulty": difficulty,
          "render_size": render_size,
          "table_size": list(_DEFAULT_TABLE_SIZE),
          "max_steps": 50,
      },
      "env": {
          "obs_mode": obs_mode,
          "act_mode": act_mode,
      },
  })


@pytest.fixture
def builder():
  return SceneBuilder()


@pytest.fixture
def env():
  e = PushingEnv(_make_env_cfg())
  yield e
  e.close()


@pytest.fixture
def env_ee():
  e = PushingEnv(_make_env_cfg(act_mode="ee"))
  yield e
  e.close()


# ---------------------------------------------------------------------------
# SceneConfig dataclass tests
# ---------------------------------------------------------------------------

class TestSceneConfig:
  def test_construction(self):
    cfg = _make_simple_config()
    assert cfg.table_size == [0.30, 0.25, 0.02]
    assert cfg.target.shape == "box"
    assert cfg.target.mass == 0.1
    assert cfg.goal_pos == [0.10, 0.05]
    assert cfg.max_steps == 50
    assert cfg.obstacles == []

  def test_object_config_fields(self):
    obj = ObjectConfig(
        shape="cylinder",
        size=[0.03, 0.05],
        mass=0.15,
        friction=0.6,
        pos=[0.1, 0.0, 0.09],
        orientation=1.0,
        color=[0.0, 1.0, 0.0, 1.0],
    )
    assert obj.shape == "cylinder"
    assert obj.mesh_path is None

  def test_camera_config(self):
    cam = CameraConfig(pos=[0.0, -0.4, 0.5],
                       look_at=[0.0, 0.0, 0.04], fov=60.0)
    assert cam.fov == 60.0

  def test_lighting_config(self):
    light = LightingConfig(
        direction=[
            0.0, -0.5, -1.0], intensity=0.8, ambient=0.3)
    assert 0.0 <= light.intensity <= 1.0


# ---------------------------------------------------------------------------
# SceneGenerator tests
# ---------------------------------------------------------------------------

class TestSceneGenerator:
  def test_sample_easy(self):
    gen = SceneGenerator(_make_env_cfg(difficulty="easy"))
    cfg = gen.sample()
    assert isinstance(cfg, SceneConfig)
    assert cfg.target.shape in ["box", "cylinder"]
    assert len(cfg.obstacles) == 0

  def test_sample_medium(self):
    gen = SceneGenerator(_make_env_cfg(difficulty="medium"))
    cfg = gen.sample()
    assert isinstance(cfg, SceneConfig)
    assert cfg.target.shape in ["box", "cylinder", "capsule"]

  def test_sample_hard(self):
    gen = SceneGenerator(_make_env_cfg(difficulty="hard"))
    cfg = gen.sample()
    assert isinstance(cfg, SceneConfig)

  def test_invalid_difficulty(self):
    with pytest.raises(ValueError):
      SceneGenerator(_make_env_cfg(difficulty="extreme"))

  def test_validity_checks_reject_bad_target(self):
    gen = SceneGenerator(_make_env_cfg(difficulty="easy"))
    cfg = gen.sample()
    cfg.target.pos = [0.50, 0.50, cfg.target.pos[2]]
    assert not gen._check_reachability(cfg)

  def test_validity_goal_on_table(self):
    gen = SceneGenerator(_make_env_cfg(difficulty="easy"))
    cfg = gen.sample()
    cfg.goal_pos = [5.0, 5.0]
    assert not gen._check_goal_on_table(cfg)

  def test_multiple_samples_are_valid(self):
    gen = SceneGenerator(_make_env_cfg(difficulty="easy", seed=0))
    for _ in range(10):
      cfg = gen.sample()
      assert gen._is_valid(cfg), "Generated config should be valid"


# ---------------------------------------------------------------------------
# SceneBuilder tests
# ---------------------------------------------------------------------------

class TestSceneBuilder:
  def test_build_returns_mjmodel(self, builder):
    cfg = _make_simple_config()
    model = builder.build(cfg)
    assert isinstance(model, mujoco.MjModel)

  def test_target_object_in_model(self, builder):
    cfg = _make_simple_config()
    model = builder.build(cfg)
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    assert body_id >= 0

  def test_obstacles_in_model(self, builder):
    cfg = _make_simple_config()
    cfg.obstacles = [
        ObjectConfig(
            shape="box",
            size=[0.03, 0.03, 0.03],
            mass=0.2,
            friction=0.5,
            pos=[0.08, 0.08, 0.07],
            orientation=0.0,
            color=[0.0, 0.0, 1.0, 1.0],
        )
    ]
    model = builder.build(cfg)
    obs_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_0")
    assert obs_id >= 0

  def test_simple_arm_xml_exists(self):
    from chuck_dreamer.sim.scene_builder import _SIMPLE_ARM_XML
    assert _SIMPLE_ARM_XML.exists()
    content = _SIMPLE_ARM_XML.read_text()
    assert "ee_frame" in content
    assert "joint1" in content


# ---------------------------------------------------------------------------
# PushingEnv tests — joint action mode (default for these fixtures)
# ---------------------------------------------------------------------------

def _zero_joint_action(n_joints: int) -> np.ndarray:
  return np.zeros(n_joints, dtype=np.float32)


def _ee_pose_action(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
  return np.concatenate([
    np.asarray(pos, dtype=np.float32),
    np.asarray(quat, dtype=np.float32),
  ])


class TestPushingEnv:
  def test_reset_returns_correct_shapes(self, env):
    cfg = _make_simple_config()
    obs, info = env.reset(scene=cfg)
    assert obs["image"].shape == (64, 64, 3)
    assert obs["image"].dtype == np.uint8
    assert obs["ee_pos"].shape == (3,)
    assert obs["object_xy"].shape == (2,)

  def test_step_returns_correct_shapes(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    action = _zero_joint_action(len(cfg.joint_names))
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs["image"].shape == (64, 64, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "object_xy" in info
    assert "ee_pos" in info

  def test_action_clipping(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    # Oversized joint action should be clipped without error.
    action = np.full(len(cfg.joint_names), 100.0, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs["image"].shape == (64, 64, 3)

  def test_episode_terminates_on_timeout(self, env):
    cfg = _make_simple_config()
    cfg.max_steps = 3
    env.reset(scene=cfg)
    action = _zero_joint_action(len(cfg.joint_names))
    done = False
    for _ in range(5):
      _, _, term, _, _ = env.step(action)
      if term:
        done = True
        break
    assert done

  def test_reset_requires_config(self, env):
    with pytest.raises((AssertionError, ValueError, TypeError)):
      env.reset()

  def test_observation_and_action_spaces_joint(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    assert env.observation_space is not None
    assert env.action_space is not None
    assert env.action_space.shape == (len(cfg.joint_names),)

  def test_action_space_ee_mode(self, env_ee):
    cfg = _make_simple_config()
    env_ee.reset(scene=cfg)
    assert env_ee.action_space.shape == (7,)

  def test_observation_space_state_mode(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    # state = ee_pos(3) + ee_quat(4) + object_xy(2) + arm_qpos(n_joints)
    assert env.observation_space.shape == (3 + 4 + 2 + len(cfg.joint_names),)

  def test_render_returns_image(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    img = env.render()
    assert img is not None
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8

  def test_render_before_reset_returns_none(self, env):
    assert env.render() is None

  def test_close_is_idempotent(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    env.close()
    env.close()

  def test_reset_twice_releases_previous_renderer(self, env):
    """Second reset() must close the previous renderer."""
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    first_renderer = env.renderer
    env.reset(scene=cfg)
    assert env.renderer is not None
    assert env.renderer is not first_renderer

  def test_step_info_contents(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    action = _zero_joint_action(len(cfg.joint_names))
    _, _, _, _, info = env.step(action)
    assert info["object_xy"].shape == (2,)
    assert info["ee_pos"].shape == (3,)
    assert info["goal_xy"].shape == (2,)
    assert info["step"] == 1

  def test_step_increments_step_count(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    action = _zero_joint_action(len(cfg.joint_names))
    for expected in (1, 2, 3):
      _, _, _, _, info = env.step(action)
      assert info["step"] == expected

  def test_reset_clears_step_count(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    action = _zero_joint_action(len(cfg.joint_names))
    env.step(action)
    env.step(action)
    env.reset(scene=cfg)
    _, _, _, _, info = env.step(action)
    assert info["step"] == 1

  def test_reward_is_negative_distance_to_goal(self, env):
    cfg = _make_simple_config()
    env.reset(scene=cfg)
    action = _zero_joint_action(len(cfg.joint_names))
    _, reward, _, _, info = env.step(action)
    expected = -float(np.linalg.norm(info["object_xy"] - info["goal_xy"]))
    assert reward == pytest.approx(expected, abs=1e-5)

  def test_goal_reached_terminates(self, env):
    cfg = _make_simple_config()
    cfg.goal_pos = [cfg.target.pos[0], cfg.target.pos[1]]
    cfg.goal_tolerance = 0.5
    env.reset(scene=cfg)
    action = _zero_joint_action(len(cfg.joint_names))
    _, _, terminated, _, _ = env.step(action)
    assert terminated

  def test_step_without_reset_asserts(self, env):
    with pytest.raises(AssertionError):
      env.step(_zero_joint_action(6))

  def test_observation_keys(self, env):
    cfg = _make_simple_config()
    obs, _ = env.reset(scene=cfg)
    expected = {"image", "ee_pos", "ee_quat", "arm_qpos",
                "object_xy", "goal_xy", "step", "time"}
    assert expected.issubset(obs.keys())
    # Full data.qpos was previously exposed but is no longer needed.
    assert "qpos" not in obs

  def test_initial_qpos_is_applied(self, env):
    cfg = _make_simple_config()
    cfg.robot_initial_qpos = [0.1, -0.2, 0.3, -0.1, 0.2, -0.3]
    obs, _ = env.reset(scene=cfg)
    assert obs["arm_qpos"] == pytest.approx(cfg.robot_initial_qpos, abs=1e-3)


class TestController:
  """Tests for PushingEnv.Controller — IK with the new ik_for_pose API."""

  def test_ik_position_only_at_current_returns_current_qpos(self, env):
    """Position-only IK at the current EE position returns ≈ the current qpos."""
    cfg = _make_simple_config()
    obs, _ = env.reset(scene=cfg)
    current_ee = obs["ee_pos"].astype(np.float64)
    q = env.controller.ik_for_pose(env.data, current_ee, target_quat=None)
    assert np.all(np.isfinite(q))
    assert q.shape == obs["arm_qpos"].shape
    np.testing.assert_allclose(q, obs["arm_qpos"], atol=1e-3)

  def test_ik_position_only_moves_toward_target(self, env):
    """Position IK for a nearby reachable target should bring the EE closer."""
    cfg = _make_simple_config()
    obs, _ = env.reset(scene=cfg)
    current_ee = obs["ee_pos"].astype(np.float64)
    target = current_ee + np.array([0.02, 0.0, 0.0])
    q_new = env.controller.ik_for_pose(env.data, target, target_quat=None)

    env.controller.reset_initial_qpos(env.data, q_new)
    new_ee = env.controller.get_ee_pos(env.data).astype(np.float64)
    assert np.linalg.norm(new_ee - target) < np.linalg.norm(current_ee - target)

  def test_ik_pose_with_current_quat_converges(self, env):
    """6-DOF IK at the current pose (pos + quat) converges to current qpos."""
    cfg = _make_simple_config()
    obs, _ = env.reset(scene=cfg)
    current_ee   = obs["ee_pos"].astype(np.float64)
    current_quat = obs["ee_quat"].astype(np.float64)
    q = env.controller.ik_for_pose(env.data, current_ee, target_quat=current_quat)
    assert np.all(np.isfinite(q))
    np.testing.assert_allclose(q, obs["arm_qpos"], atol=1e-3)

  def test_ik_raises_when_unreachable(self, env):
    """A wildly-unreachable target should raise RuntimeError."""
    cfg = _make_simple_config()
    obs, _ = env.reset(scene=cfg)
    unreachable = np.array([100.0, 100.0, 100.0])
    with pytest.raises(RuntimeError):
      env.controller.ik_for_pose(env.data, unreachable, target_quat=None)

  def test_ee_step_runs_through_ik(self, env_ee):
    """End-to-end: an EE-mode step uses IK and lands close to the target."""
    cfg = _make_simple_config()
    obs, _ = env_ee.reset(scene=cfg)
    target_pos = obs["ee_pos"] + np.array([0.01, 0.0, 0.0], dtype=np.float32)
    action = _ee_pose_action(target_pos, obs["ee_quat"])
    next_obs, _, _, _, _ = env_ee.step(action)
    # The actuator may not fully track the IK solution in one step; we
    # only require the EE moved toward the target.
    assert np.linalg.norm(next_obs["ee_pos"] - target_pos) <= np.linalg.norm(obs["ee_pos"] - target_pos)


class _FakeViewerWithGeoms:
  """Minimal viewer stand-in for insert_hints."""

  def __init__(self, maxgeom: int):
    self.user_scn = type("UserScn", (), {})()
    self.user_scn.ngeom = 0
    self.user_scn.maxgeom = maxgeom
    self.user_scn.geoms = [mujoco.MjvGeom() for _ in range(maxgeom)]


class TestPolicyInsertHints:
  """insert_hints lives on the policy, not the env."""

  def _make_policy(self):
    cfg = _make_simple_config()
    cfg.target.pos = [0.05, 0.00, 0.03]
    cfg.goal_pos = [0.20, 0.00]
    p = ScriptedPolicy()
    p.reset(cfg)
    p.state = "ready"
    return p

  def test_insert_hints_adds_two_geoms(self):
    policy = self._make_policy()
    viewer = _FakeViewerWithGeoms(maxgeom=10)
    policy.insert_hints(viewer)
    assert viewer.user_scn.ngeom == 2

  def test_insert_hints_noop_when_geom_buffer_full(self):
    policy = self._make_policy()
    viewer = _FakeViewerWithGeoms(maxgeom=2)
    viewer.user_scn.ngeom = 2
    policy.insert_hints(viewer)
    assert viewer.user_scn.ngeom == 2

  def test_insert_hints_noop_when_not_ready(self):
    policy = self._make_policy()
    policy.state = "approach"
    viewer = _FakeViewerWithGeoms(maxgeom=10)
    policy.insert_hints(viewer)
    assert viewer.user_scn.ngeom == 0


# ---------------------------------------------------------------------------
# EpisodeWriter tests
# ---------------------------------------------------------------------------

class TestEpisodeWriter:
  def _make_fake_episode(self, T: int = 5, H: int = 64, W: int = 64, n_joints: int = 6):
    rng = np.random.default_rng(42)
    return {
        "image":      rng.integers(0, 256, (T, H, W, 3), dtype=np.uint8),
        "action":     rng.uniform(-0.02, 0.02, (T, n_joints)).astype(np.float32),
        "reward":     rng.uniform(-1.0, 0.0, (T,)).astype(np.float32),
        "timestamp":  (np.arange(T) * 0.1).astype(np.float32),
        "joint_qpos": rng.uniform(-1.0, 1.0, (T, n_joints)).astype(np.float32),
        "ee_pos":     rng.uniform(-0.5, 0.5, (T, 3)).astype(np.float32),
        "ee_quat":    np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (T, 1)),
        "object_xy":  rng.uniform(-0.3, 0.3, (T, 2)).astype(np.float32),
    }

  def test_write_and_read_back(self, tmp_path):
    writer = EpisodeWriter(str(tmp_path), format="hdf5")
    episode = self._make_fake_episode(T=5, H=64, W=64, n_joints=6)
    cfg = _make_simple_config()
    path = writer.write_episode(
        episode,
        metadata={
            "config":  cfg,
            "seed":    42,
            "source":  "sim",
            "outcome": "done",
            "goal_xy": cfg.goal_pos,
        })

    assert path.exists()
    with h5py.File(path, "r") as f:
      assert f["images"].shape     == (5, 64, 64, 3)
      assert f["images"].dtype     == np.uint8
      assert f["actions"].shape    == (5, 6)
      assert f["rewards"].shape    == (5,)
      assert f["timestamps"].shape == (5,)
      assert f["joint_qpos"].shape == (5, 6)
      assert f["ee_pos"].shape     == (5, 3)
      assert f["ee_quat"].shape    == (5, 4)
      assert f["object_xy"].shape  == (5, 2)
      meta = f["metadata"]
      assert int(meta["seed"][()])        == 42
      assert "sim" in str(meta["source"][()])
      assert "done" in str(meta["outcome"][()])
      assert meta["goal_xy"].shape == (2,)

  def test_config_is_valid_json(self, tmp_path):
    writer = EpisodeWriter(str(tmp_path))
    episode = self._make_fake_episode(T=3)
    cfg = _make_simple_config()
    path = writer.write_episode(
        episode,
        metadata={
            "config": cfg,
            "seed": 0,
            "source": "sim"})
    with h5py.File(path, "r") as f:
      raw = f["metadata/config"][()]
      if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
      parsed = json.loads(raw)
    assert "target" in parsed

  def test_episode_counter_increments(self, tmp_path):
    writer = EpisodeWriter(str(tmp_path))
    ep = self._make_fake_episode(T=2)
    p1 = writer.write_episode(ep)
    p2 = writer.write_episode(ep)
    assert p1 != p2
    assert p1.name == "episode_00000.hdf5"
    assert p2.name == "episode_00001.hdf5"

  def test_unsupported_format_raises(self, tmp_path):
    with pytest.raises(ValueError):
      EpisodeWriter(str(tmp_path), format="csv")

  def test_rerun_writes_rrd_file(self, tmp_path):
    pytest.importorskip("rerun")
    writer = EpisodeWriter(str(tmp_path), format="rerun")
    episode = self._make_fake_episode(T=3, H=32, W=32, n_joints=4)
    cfg = _make_simple_config()
    path = writer.write_episode(
        episode,
        metadata={
            "config":  cfg,
            "seed":    7,
            "source":  "sim",
            "outcome": "done",
            "goal_xy": cfg.goal_pos,
        })
    assert path.exists()
    assert path.suffix == ".rrd"
    assert path.stat().st_size > 0

  def test_rerun_counter_increments(self, tmp_path):
    pytest.importorskip("rerun")
    writer = EpisodeWriter(str(tmp_path), format="rerun")
    ep = self._make_fake_episode(T=2)
    p1 = writer.write_episode(ep)
    p2 = writer.write_episode(ep)
    assert p1.name == "episode_00000.rrd"
    assert p2.name == "episode_00001.rrd"


# ---------------------------------------------------------------------------
# ScriptedPolicy tests
# ---------------------------------------------------------------------------

_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _policy_obs(ee_pos, qpos=None, n_joints: int = 6, ee_quat: np.ndarray = _IDENTITY_QUAT):
  qpos_arr = (np.zeros(n_joints, dtype=np.float32)
              if qpos is None else np.asarray(qpos, dtype=np.float32))
  return {
      "ee_pos":   np.asarray(ee_pos, dtype=np.float32),
      "ee_quat":  np.asarray(ee_quat, dtype=np.float32),
      "arm_qpos": qpos_arr,
  }


def _fresh_policy():
  cfg = _make_simple_config()
  cfg.target.pos = [0.05, 0.00, 0.03]
  cfg.goal_pos = [0.20, 0.00]
  p = ScriptedPolicy()
  p.reset(cfg)
  return p


class TestScriptedPolicy:
  """Exercises the state-machine logic. Policy outputs (7,) EE poses."""

  def test_initial_state(self):
    p = _fresh_policy()
    assert p.state == "initial"

  def test_approach_xy_offsets_behind_object(self):
    p = _fresh_policy()
    assert p.scene is not None
    obj = np.array(p.scene.target.pos[:2])
    assert p.approach_xy[0] < obj[0]
    d = float(np.linalg.norm(p.approach_xy - obj))
    assert d == pytest.approx(ScriptedPolicy._STANDOFF, rel=1e-3)

  def test_approach_xyz_uses_push_z(self):
    p = _fresh_policy()
    assert p.approach_xyz[2] == pytest.approx(ScriptedPolicy._PUSH_Z)

  def test_goal_xyz_uses_push_z(self):
    p = _fresh_policy()
    assert p.goal_xyz[2] == pytest.approx(ScriptedPolicy._PUSH_Z)

  def test_transition_initial_to_ready(self):
    p = _fresh_policy()
    obs = _policy_obs(np.append(p.ready_xy, ScriptedPolicy._PUSH_Z))
    p.act(obs)
    assert p.state == "ready"

  def test_ready_is_sticky_without_external_signal(self):
    """ready → approach is a manual transition; policy itself does not self-advance."""
    p = _fresh_policy()
    p.state = "ready"
    obs = _policy_obs(np.append(p.ready_xy, ScriptedPolicy._PUSH_Z))
    p.act(obs)
    assert p.state == "ready"

  def test_transition_approach_to_push(self):
    p = _fresh_policy()
    p.state = "approach"
    obs = _policy_obs(p.approach_xyz)
    p.act(obs)
    assert p.state == "push"

  def test_transition_push_to_done(self):
    p = _fresh_policy()
    p.state = "push"
    obs = _policy_obs(p.goal_xyz)
    p.act(obs)
    assert p.state == "done"

  def test_done_is_terminal(self):
    p = _fresh_policy()
    p.state = "done"
    obs = _policy_obs(p.goal_xyz)
    p.act(obs)
    assert p.state == "done"

  def test_act_returns_pose_action(self):
    p = _fresh_policy()
    obs = _policy_obs(np.array([0.0, 0.0, 0.1]))
    action = p.act(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == (7,)
    # Quat half should match the captured hold_quat (set on first act).
    np.testing.assert_array_equal(action[3:], _IDENTITY_QUAT)


# ---------------------------------------------------------------------------
# ScenePlayer tests (fake env + fake policy — no MuJoCo)
# ---------------------------------------------------------------------------

def _fake_obs(step: int, n_joints: int = 6):
  return {
      "image":    np.full((4, 4, 3), step % 256, dtype=np.uint8),
      "ee_pos":   np.array([0.0, 0.0, 0.1], dtype=np.float32),
      "ee_quat":  np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
      "arm_qpos": np.zeros(n_joints, dtype=np.float32),
      "object_xy": np.array([0.05, 0.0], dtype=np.float32),
      "goal_xy":  np.array([0.2, 0.0], dtype=np.float32),
      "time":     float(step) * 0.1,
      "step":     step,
  }


class _FakeEnv:
  """Minimal PushingEnv stand-in. Tracks step count, honors control over done flags."""

  def __init__(self, terminate_at: int | None = None, crash_at: int | None = None, n_joints: int = 6):
    self.terminate_at = terminate_at
    self.crash_at     = crash_at
    self.n_joints     = n_joints
    self._step        = 0
    self.reset_calls  = 0

  def generate_scene(self):
    return _make_simple_config()

  def reset(self, *, scene=None, seed=None):
    self.reset_calls += 1
    self._step = 0
    return _fake_obs(0, self.n_joints), {}

  def step(self, action):
    self._step += 1
    if self.crash_at is not None and self._step >= self.crash_at:
      raise RuntimeError("simulated IK failure")
    terminated = self.terminate_at is not None and self._step >= self.terminate_at
    return _fake_obs(self._step, self.n_joints), -0.1, terminated, False, {}

  def _get_obs(self):
    return _fake_obs(self._step, self.n_joints)


class _FakePolicy:
  """Policy stub with scripted state progression. New contract: act -> ndarray."""

  def __init__(self, states: list[str], n_joints: int = 6):
    self._states = list(states)
    self.state = self._states[0] if self._states else "initial"
    self._idx = 0
    self.n_joints = n_joints
    self.act_calls = 0
    self.insert_hints_calls = 0
    self.reset_calls = 0

  def reset(self, scene):
    self.reset_calls += 1
    self.state = self._states[0] if self._states else "initial"
    self._idx = 0

  def act(self, obs):
    if self._idx + 1 < len(self._states):
      next_state = self._states[self._idx + 1]
      self.state = next_state
      self._idx += 1
    self.act_calls += 1
    return np.zeros(7, dtype=np.float32)

  def insert_hints(self, viewer):
    self.insert_hints_calls += 1


class _FakeViewer:
  def __init__(self, max_frames: int):
    self._frames = max_frames
    self.user_scn = type("UserScn", (), {"ngeom": 0, "maxgeom": 100})()
    self.sync_calls = 0

  def is_running(self):
    return self._frames > 0

  def sync(self):
    self.sync_calls += 1
    self._frames -= 1


class TestScenePlayerHeadless:
  def _make_player(self, env, policy):
    cfg = _make_simple_config()
    cfg.max_steps = 20
    player = ScenePlayer(cfg, env, policy)
    player.reset()
    return player

  def test_auto_advances_ready_to_approach(self):
    env    = _FakeEnv(terminate_at=None)
    policy = _FakePolicy(["ready", "ready", "approach", "push", "done"])
    player = self._make_player(env, policy)

    episode, outcome = player.run_headless(max_steps=10)
    assert outcome == "done"
    assert policy.state == "done"
    assert episode is not None
    assert episode["action"].shape[0] >= 1

  def test_episode_has_all_required_fields(self):
    env    = _FakeEnv()
    policy = _FakePolicy(["approach", "done"])
    player = self._make_player(env, policy)

    episode, outcome = player.run_headless(max_steps=5)
    assert outcome == "done"
    assert episode is not None
    expected_keys = {
        "image", "action", "reward",
        "timestamp", "joint_qpos", "ee_pos", "ee_quat", "object_xy",
    }
    assert expected_keys.issubset(episode.keys())
    T = episode["action"].shape[0]
    assert episode["action"].dtype == np.float32
    assert episode["ee_quat"].shape == (T, 4)
    assert episode["object_xy"].shape == (T, 2)

  def test_outcome_timeout(self):
    env    = _FakeEnv()
    policy = _FakePolicy(["approach", "approach", "approach", "approach", "approach"])
    player = self._make_player(env, policy)

    episode, outcome = player.run_headless(max_steps=3)
    assert outcome == "timeout"
    assert episode is not None
    assert episode["action"].shape[0] == 3

  def test_outcome_terminated(self):
    env    = _FakeEnv(terminate_at=2)
    policy = _FakePolicy(["approach", "approach", "approach", "approach"])
    player = self._make_player(env, policy)

    episode, outcome = player.run_headless(max_steps=10)
    assert outcome == "terminated"
    assert episode is not None
    assert episode["action"].shape[0] == 2

  def test_outcome_crashed_returns_partial_data(self):
    env    = _FakeEnv(crash_at=3)
    policy = _FakePolicy(["approach"] * 10)
    player = self._make_player(env, policy)

    episode, outcome = player.run_headless(max_steps=10)
    assert outcome == "crashed"
    assert episode is not None
    assert episode["action"].shape[0] == 2


class TestScenePlayerInteractive:
  def _make_player(self, env, policy):
    cfg = _make_simple_config()
    return ScenePlayer(cfg, env, policy)

  def test_shows_hints_only_while_ready(self):
    env    = _FakeEnv()
    policy = _FakePolicy(["ready", "ready", "approach", "approach"])
    player = self._make_player(env, policy)
    viewer = _FakeViewer(max_frames=3)

    player.run_interactive(viewer, step_delay=0.0)
    assert policy.insert_hints_calls >= 1
    assert viewer.sync_calls == 3

  def test_stops_on_done(self):
    env    = _FakeEnv()
    policy = _FakePolicy(["approach", "done"])
    player = self._make_player(env, policy)
    viewer = _FakeViewer(max_frames=10)

    player.run_interactive(viewer, step_delay=0.0)
    assert policy.state == "done"
    assert viewer.sync_calls == 1

  def test_stops_on_env_terminated(self):
    env    = _FakeEnv(terminate_at=2)
    policy = _FakePolicy(["approach"] * 10)
    player = self._make_player(env, policy)
    viewer = _FakeViewer(max_frames=10)

    player.run_interactive(viewer, step_delay=0.0)
    assert viewer.sync_calls == 2
