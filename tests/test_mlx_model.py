"""Shape and contract tests for the Dreamer MLX model.

These exercise the documented interfaces of every component (encoder,
decoder, RSSM, reward head, actor, critic) plus an end-to-end build
from the project's default config. They don't train anything, so they
run in well under a second on CPU/MPS.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from chuck_dreamer.config import load_config  # noqa: E402
from chuck_dreamer.dreamer import build_model  # noqa: E402
from chuck_dreamer.dreamer.mlx_model import (  # noqa: E402
  Actor,
  Critic,
  DreamerMLXModel,
  MLPDecoder,
  MLPEncoder,
  RewardHead,
  RSSM,
  feat,
)


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------


def test_mlp_encoder_output_shape():
  enc = MLPEncoder(obs_dim=15, hidden=(64, 64), embed_dim=32)
  out = enc(mx.zeros((4, 15)))
  assert out.shape == (4, 32)


def test_mlp_decoder_output_shape():
  dec = MLPDecoder(feat_dim=230, hidden=(64, 64), obs_dim=15)
  out = dec(mx.zeros((4, 230)))
  assert out.shape == (4, 15)


# ---------------------------------------------------------------------------
# feat()
# ---------------------------------------------------------------------------


def test_feat_concatenates_h_and_s():
  state = {"h": mx.zeros((4, 200)), "s": mx.ones((4, 30))}
  f = feat(state)
  assert f.shape == (4, 230)
  # First deter_dim entries come from h (zeros), rest from s (ones).
  assert float(mx.sum(f[:, :200])) == 0.0
  assert float(mx.sum(f[:, 200:])) == pytest.approx(4 * 30)


# ---------------------------------------------------------------------------
# RSSM
# ---------------------------------------------------------------------------


def _make_rssm(action_dim=6, embed_dim=32, stoch=8, deter=16, hidden=16, min_std=0.1):
  return RSSM(
    action_dim=action_dim,
    embed_dim=embed_dim,
    stoch_dim=stoch,
    deter_dim=deter,
    hidden=hidden,
    min_std=min_std,
  )


def test_rssm_initial_state_shapes():
  rssm = _make_rssm()
  st = rssm.initial_state(batch_size=5)
  assert st["h"].shape == (5, 16)
  assert st["s"].shape == (5, 8)


def test_rssm_img_step_returns_prior_only():
  rssm = _make_rssm()
  st = rssm.initial_state(3)
  out = rssm.img_step(st, mx.zeros((3, 6)))
  assert out["h"].shape == (3, 16)
  assert out["s"].shape == (3, 8)
  assert out["prior_mean"].shape == (3, 8)
  assert out["prior_std"].shape == (3, 8)
  # img_step doesn't see an observation; no posterior.
  assert "post_mean" not in out
  assert "post_std" not in out


def test_rssm_obs_step_returns_prior_and_posterior():
  rssm = _make_rssm()
  st = rssm.initial_state(3)
  out = rssm.obs_step(st, mx.zeros((3, 6)), mx.zeros((3, 32)))
  for k in ("h", "s", "prior_mean", "prior_std", "post_mean", "post_std"):
    assert k in out, f"missing key {k!r}"
  assert out["post_mean"].shape == (3, 8)
  assert out["post_std"].shape == (3, 8)


def test_rssm_std_respects_min_std():
  min_std = 0.25
  rssm = _make_rssm(min_std=min_std)
  st = rssm.initial_state(2)
  out = rssm.obs_step(st, mx.zeros((2, 6)), mx.zeros((2, 32)))
  # softplus(.) + min_std >= min_std, always.
  assert float(mx.min(out["prior_std"])) >= min_std
  assert float(mx.min(out["post_std"])) >= min_std


def test_rssm_observe_returns_one_state_per_step():
  rssm = _make_rssm()
  B, T = 2, 5
  embeds = mx.zeros((B, T, 32))
  actions = mx.zeros((B, T, 6))
  states = rssm.observe(embeds, actions)
  assert len(states) == T
  for s in states:
    assert s["h"].shape == (B, 16)
    assert s["s"].shape == (B, 8)


def test_rssm_imagine_returns_horizon_plus_one_states():
  rssm = _make_rssm()
  init = rssm.initial_state(2)

  feat_dim = rssm.deter_dim + rssm.stoch_dim
  calls: list[tuple[int, ...]] = []

  def policy_fn(f):
    calls.append(tuple(f.shape))
    return mx.zeros((f.shape[0], rssm.action_dim))

  traj = rssm.imagine(init, policy_fn, horizon=4)
  assert len(traj) == 5  # init + 4 imagined states
  assert traj[0] is init
  # Policy is called once per imagined step on the concatenated feature.
  assert calls == [(2, feat_dim)] * 4


# ---------------------------------------------------------------------------
# Reward head / Critic
# ---------------------------------------------------------------------------


def test_reward_head_squeezes_trailing_dim():
  head = RewardHead(feat_dim=230, hidden=(32, 32))
  out = head(mx.zeros((6, 230)))
  assert out.shape == (6,)


def test_critic_squeezes_trailing_dim():
  critic = Critic(feat_dim=230, hidden=(32, 32))
  out = critic(mx.zeros((6, 230)))
  assert out.shape == (6,)


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------


def _make_actor(feat_dim=230, action_dim=6, **kw):
  return Actor(feat_dim=feat_dim, action_dim=action_dim, hidden=(32, 32), **kw)


def test_actor_call_shapes_and_action_in_tanh_range():
  actor = _make_actor()
  action, mean, std = actor(mx.zeros((4, 230)))
  assert action.shape == (4, 6)
  assert mean.shape == (4, 6)
  assert std.shape == (4, 6)
  # tanh squash bounds the sampled action.
  assert float(mx.max(action)) <= 1.0
  assert float(mx.min(action)) >= -1.0


def test_actor_mode_is_in_tanh_range_and_deterministic():
  actor = _make_actor()
  f = mx.zeros((4, 230))
  a1 = actor.mode(f)
  a2 = actor.mode(f)
  assert a1.shape == (4, 6)
  assert float(mx.max(a1)) <= 1.0
  assert float(mx.min(a1)) >= -1.0
  # mode() has no sampling, so two calls must agree exactly.
  assert mx.array_equal(a1, a2)


def test_actor_std_respects_min_std():
  min_std = 0.05
  actor = _make_actor(min_std=min_std, init_std=5.0)
  _, _, std = actor(mx.zeros((4, 230)))
  assert float(mx.min(std)) >= min_std


def test_actor_initial_std_uses_init_std_offset():
  """Dropping the init_std offset would collapse std toward min_std at init.
  We don't pin a single value (random init weights make raw_std nonzero),
  but assert std lives in the band the offset is meant to produce."""
  init_std = 5.0
  actor = _make_actor(init_std=init_std, min_std=1e-4)
  _, _, std = actor(mx.zeros((4, 230)))
  # softplus is monotonic; with init_std=5 the smallest possible std is
  # well above 1.0. If the offset was dropped the std would sit near 0.7
  # (softplus(small) ≈ ln 2). 2.0 is a safe lower bound that catches that.
  assert float(mx.min(std)) >= 2.0


# ---------------------------------------------------------------------------
# DreamerMLXModel — config wiring + end-to-end forward pass
# ---------------------------------------------------------------------------


def test_dreamer_model_builds_from_default_config():
  cfg = load_config()
  model = build_model(cfg, obs_dim=15, action_dim=6)
  assert isinstance(model, DreamerMLXModel)
  assert isinstance(model.encoder, MLPEncoder)
  assert isinstance(model.decoder, MLPDecoder)
  assert isinstance(model.rssm, RSSM)
  assert isinstance(model.reward_head, RewardHead)
  assert isinstance(model.actor, Actor)
  assert isinstance(model.critic, Critic)


def test_dreamer_model_end_to_end_forward_shapes():
  cfg = load_config()
  B, T, obs_dim, action_dim = 2, 4, 15, 6
  model = build_model(cfg, obs_dim=obs_dim, action_dim=action_dim)

  obs = mx.zeros((B, T, obs_dim))
  actions = mx.zeros((B, T, action_dim))

  embeds = model.encoder(obs)
  embed_size = cfg.model.encoder.embed_size
  assert embeds.shape == (B, T, embed_size)

  states = model.rssm.observe(embeds, actions)
  assert len(states) == T

  feat_dim = cfg.model.rssm.stoch_size + cfg.model.rssm.deter_size
  f = feat(states[-1])
  assert f.shape == (B, feat_dim)

  assert model.decoder(f).shape == (B, obs_dim)
  assert model.reward_head(f).shape == (B,)

  action, mean, std = model.actor(f)
  assert action.shape == (B, action_dim)
  assert mean.shape == (B, action_dim)
  assert std.shape == (B, action_dim)

  assert model.critic(f).shape == (B,)


def test_dreamer_model_imagine_uses_actor_and_yields_horizon_plus_one():
  cfg = load_config()
  model = build_model(cfg, obs_dim=15, action_dim=6)

  init = model.rssm.initial_state(batch_size=3)

  def policy_fn(f):
    a, _, _ = model.actor(f)
    return a

  traj = model.rssm.imagine(init, policy_fn, horizon=5)
  assert len(traj) == 6
  for st in traj:
    assert st["h"].shape == (3, cfg.model.rssm.deter_size)
    assert st["s"].shape == (3, cfg.model.rssm.stoch_size)


def test_dreamer_model_rejects_unknown_encoder_type():
  cfg = load_config()
  cfg.model.encoder.type = "conv"  # not implemented
  with pytest.raises(ValueError, match="encoder"):
    build_model(cfg, obs_dim=15, action_dim=6)


def test_dreamer_model_rejects_unknown_decoder_type():
  cfg = load_config()
  cfg.model.decoder.type = "deconv"  # not implemented
  with pytest.raises(ValueError, match="decoder"):
    build_model(cfg, obs_dim=15, action_dim=6)


def test_build_model_rejects_unknown_device():
  cfg = load_config()
  cfg.hardware.device = "cuda"
  with pytest.raises(ValueError):
    build_model(cfg, obs_dim=15, action_dim=6)
