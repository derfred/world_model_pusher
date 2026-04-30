from typing import cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


# Helpers
def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, act=nn.ELU) -> nn.Sequential:
  """Standard MLP: Linear -> act -> ... -> Linear (no act on output)."""
  layers: list[nn.Module] = []
  prev = in_dim
  for h in hidden:
    layers.append(nn.Linear(prev, h))
    layers.append(act())
    prev = h
  layers.append(nn.Linear(prev, out_dim))
  return nn.Sequential(*layers)


def feat(state: dict) -> mx.array:
  """Concatenate deterministic and stochastic state into the feature fed
  to decoder, reward head, actor, and critic."""
  return mx.concatenate([state["h"], state["s"]], axis=-1)


def kl_gaussian(
  mean_q: mx.array,
  std_q:  mx.array,
  mean_p: mx.array,
  std_p:  mx.array,
) -> mx.array:
  """KL(q || p) for diagonal Gaussians, per-dimension.

  Returns the per-dim KL with the same shape as the inputs. Caller decides
  whether to sum over the latent dim (typical) or reduce differently.

  Args:
    mean_q, std_q: parameters of q, the posterior in Dreamer's ELBO.
    mean_p, std_p: parameters of p, the prior.

  All four arrays broadcast together; typical shape is (B, stoch_dim) or
  (B, T, stoch_dim).
  """
  var_q = std_q ** 2
  var_p = std_p ** 2
  return (
    mx.log(std_p / std_q)
    + (var_q + (mean_q - mean_p) ** 2) / (2.0 * var_p)
    - 0.5
  )


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------


class MLPEncoder(nn.Module):
  """For state-based observations (joint qpos + ee pose + object pose, etc.).
  Produces an embedding the RSSM consumes."""

  def __init__(self, obs_dim: int, hidden: tuple[int, ...], embed_dim: int):
    super().__init__()
    self.net = _mlp(obs_dim, hidden, embed_dim)

  def __call__(self, obs: mx.array) -> mx.array:
    """obs: (..., obs_dim) -> embedding: (..., embed_dim)"""
    return cast(mx.array, self.net(obs))


class MLPDecoder(nn.Module):
  """Reconstructs state-based observations from latent features."""

  def __init__(self, feat_dim: int, hidden: tuple[int, ...], obs_dim: int):
    super().__init__()
    self.net = _mlp(feat_dim, hidden, obs_dim)

  def __call__(self, feat_in: mx.array) -> mx.array:
    return cast(mx.array, self.net(feat_in))


# ---------------------------------------------------------------------------
# RSSM: Recurrent State Space Model
# ---------------------------------------------------------------------------


class RSSM(nn.Module):
  """Dreamer v1 RSSM.

  State = (h, s) where:
    h_t = GRU(h_{t-1}, fc([s_{t-1}, a_{t-1}]))         # deterministic
    s_t ~ N(prior(h_t))                                 # stochastic prior
    s_t ~ N(posterior(h_t, embed_t)) when observation present

  Methods:
    initial_state(B)              -> dict with zero h, s
    img_step(state, action)       -> next state using prior only (imagination)
    obs_step(state, action, embed)-> next state using posterior (training)
  """

  def __init__(
    self,
    action_dim: int,
    embed_dim:  int,
    stoch_dim:  int = 30,
    deter_dim:  int = 200,
    hidden:     int = 200,
    min_std:    float = 0.1,
  ):
    super().__init__()
    self.action_dim = action_dim
    self.stoch_dim  = stoch_dim
    self.deter_dim  = deter_dim
    self.min_std    = min_std

    # pre-GRU MLP combining (s_{t-1}, a_{t-1})
    self.pre_gru = nn.Sequential(
      nn.Linear(stoch_dim + action_dim, hidden),
      nn.ELU(),
    )
    # GRU cell. Single-step interface; we drive it manually.
    self.gru = nn.GRU(input_size=hidden, hidden_size=deter_dim)

    # Prior: h -> (mean, std) of s
    self.prior_net = nn.Sequential(
      nn.Linear(deter_dim, hidden), nn.ELU(),
      nn.Linear(hidden, 2 * stoch_dim),
    )

    # Posterior: (h, embed) -> (mean, std) of s
    self.post_net = nn.Sequential(
      nn.Linear(deter_dim + embed_dim, hidden), nn.ELU(),
      nn.Linear(hidden, 2 * stoch_dim),
    )

  # ------------------- helpers -------------------

  def initial_state(self, batch_size: int) -> dict:
    return {
      "h": mx.zeros((batch_size, self.deter_dim)),
      "s": mx.zeros((batch_size, self.stoch_dim)),
    }

  def _split_dist(self, out: mx.array) -> tuple[mx.array, mx.array]:
    mean, std = mx.split(out, 2, axis=-1)
    std = nn.softplus(std) + self.min_std
    return mean, std

  def _sample(self, mean: mx.array, std: mx.array) -> mx.array:
    return mean + std * mx.random.normal(mean.shape)

  # ------------------- core ops -------------------

  def _compute_h(self, prev_s: mx.array, prev_a: mx.array, prev_h: mx.array) -> mx.array:
    """One GRU step: h_t = GRU(h_{t-1}, mlp([s_{t-1}, a_{t-1}]))."""
    x = self.pre_gru(mx.concatenate([prev_s, prev_a], axis=-1))   # (B, hidden)
    # MLX GRU expects (B, T, D); we pass T=1 and squeeze.
    h_seq = self.gru(x[:, None, :], prev_h)                       # (B, 1, deter_dim)
    return cast(mx.array, h_seq[:, 0])                            # (B, deter_dim)

  def img_step(self, prev_state: dict, prev_action: mx.array) -> dict:
    """Imagination step: predict s_t from prior only, no observation."""
    h = self._compute_h(prev_state["s"], prev_action, prev_state["h"])
    prior_mean, prior_std = self._split_dist(self.prior_net(h))
    s = self._sample(prior_mean, prior_std)
    return {
      "h": h, "s": s,
      "prior_mean": prior_mean, "prior_std": prior_std,
    }

  def obs_step(self, prev_state: dict, prev_action: mx.array, embed: mx.array) -> dict:
    """Observation step: run prior, then refine with posterior using embed."""
    h = self._compute_h(prev_state["s"], prev_action, prev_state["h"])
    prior_mean, prior_std = self._split_dist(self.prior_net(h))
    post_in = mx.concatenate([h, embed], axis=-1)
    post_mean, post_std = self._split_dist(self.post_net(post_in))
    s = self._sample(post_mean, post_std)
    return {
      "h": h, "s": s,
      "prior_mean": prior_mean, "prior_std": prior_std,
      "post_mean":  post_mean,  "post_std":  post_std,
    }

  def observe(
    self,
    embeds:  mx.array,   # (B, T, embed_dim)
    actions: mx.array,   # (B, T, action_dim)
    init:    dict | None = None,
  ) -> list[dict]:
    """Roll the posterior forward for T steps. Returns list of T state dicts."""
    B, T = embeds.shape[0], embeds.shape[1]
    state = init if init is not None else self.initial_state(B)
    states = []
    for t in range(T):
      state = self.obs_step(state, actions[:, t], embeds[:, t])
      states.append(state)
    return states

  def imagine(
    self,
    init_state: dict,                          # (N, ...) starting points
    policy_fn,                                  # feat -> action
    horizon: int,
  ) -> list[dict]:
    """Roll the prior forward for `horizon` steps using a policy.
    policy_fn maps a feature tensor (N, feat_dim) to an action (N, action_dim).
    Returns a list of length horizon+1 (including the start state)."""
    state = init_state
    traj = [state]
    for _ in range(horizon):
      action = policy_fn(feat(state))
      state = self.img_step(state, action)
      traj.append(state)
    return traj


# ---------------------------------------------------------------------------
# Reward head
# ---------------------------------------------------------------------------


class RewardHead(nn.Module):
  """Predicts scalar reward from latent features."""

  def __init__(self, feat_dim: int, hidden: tuple[int, ...] = (200, 200)):
    super().__init__()
    self.net = _mlp(feat_dim, hidden, out_dim=1)

  def __call__(self, feat_in: mx.array) -> mx.array:
    return cast(mx.array, self.net(feat_in).squeeze(-1))


# ---------------------------------------------------------------------------
# Actor: tanh-squashed Gaussian
# ---------------------------------------------------------------------------

class Actor(nn.Module):
  """Stochastic actor for continuous control. Outputs a tanh-squashed Gaussian.

  Implementation details matching Dreamer v1:
    - mean is passed through `mean_scale * tanh(mean / mean_scale)` to keep it
      in (-mean_scale, +mean_scale) before the final tanh squash on the sample.
    - std uses softplus(raw + init_std) + min_std so initial std ~= 1.0.

  Returns (action, mean_pre_squash, std).
  """

  def __init__(
    self,
    feat_dim:    int,
    action_dim:  int,
    hidden:      tuple[int, ...] = (300, 300, 300),
    init_std:    float = 5.0,
    min_std:     float = 1e-4,
    mean_scale:  float = 5.0,
  ):
    super().__init__()
    self.action_dim = action_dim
    self.init_std   = init_std
    self.min_std    = min_std
    self.mean_scale = mean_scale
    self.net = _mlp(feat_dim, hidden, out_dim=2 * action_dim)
    # init_std offset: softplus(0 + init_std_raw) + min_std ~= 1.0
    # We add init_std as a constant inside softplus; pre-compute the offset.
    self._raw_std_offset = init_std

  def _dist(self, feat_in: mx.array) -> tuple[mx.array, mx.array]:
    out = self.net(feat_in)
    mean, raw_std = mx.split(out, 2, axis=-1)
    mean = self.mean_scale * mx.tanh(mean / self.mean_scale)
    std = nn.softplus(raw_std + self._raw_std_offset) + self.min_std
    return mean, std

  def __call__(self, feat_in: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Sample an action with reparameterization. Returns (action, mean, std).
    The action is tanh-squashed; mean/std describe the pre-squash Gaussian."""
    mean, std = self._dist(feat_in)
    eps = mx.random.normal(mean.shape)
    raw = mean + std * eps
    action = mx.tanh(raw)
    return action, mean, std

  def mode(self, feat_in: mx.array) -> mx.array:
    """Deterministic action for evaluation: tanh of the mean."""
    mean, _ = self._dist(feat_in)
    return mx.tanh(mean)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class Critic(nn.Module):
  """Predicts scalar state value V(s) from latent features."""

  def __init__(self, feat_dim: int, hidden: tuple[int, ...] = (300, 300, 300)):
    super().__init__()
    self.net = _mlp(feat_dim, hidden, out_dim=1)

  def __call__(self, feat_in: mx.array) -> mx.array:
    return cast(mx.array, self.net(feat_in).squeeze(-1))


class _WMBundle(nn.Module):
  """Helper class to bundle RSSM, decoder, and reward head for joint WM updates."""
  def __init__(self, encoder: MLPEncoder, rssm: RSSM, decoder: MLPDecoder, reward: RewardHead):
    super().__init__()
    self.encoder = encoder
    self.rssm    = rssm
    self.decoder = decoder
    self.reward  = reward


class DreamerMLXModel:
  def __init__(self, config, obs_dim: int, action_dim: int, training: bool = True):
    self.config   = config
    self.training = training
    feat_dim      = config.model.rssm.stoch_size + config.model.rssm.deter_size

    if config.model.encoder.type == "mlp":
      self.encoder = MLPEncoder(obs_dim, tuple(config.model.encoder.mlp_hidden), config.model.encoder.embed_size)
    else:
      raise ValueError(f"Unsupported encoder type: {config.model.encoder.type}")

    if config.model.decoder.type == "mlp":
      self.decoder = MLPDecoder(feat_dim, tuple(config.model.decoder.mlp_hidden), obs_dim)
    else:
      raise ValueError(f"Unsupported decoder type: {config.model.decoder.type}")

    self.rssm = RSSM(
      action_dim=action_dim,
      embed_dim=config.model.encoder.embed_size,
      stoch_dim=config.model.rssm.stoch_size,
      deter_dim=config.model.rssm.deter_size,
      hidden=config.model.rssm.hidden_size,
      min_std=config.model.rssm.min_stddev,
    )
    self.reward_head = RewardHead(feat_dim, tuple(config.model.reward.hidden))
    self.actor = Actor(
      feat_dim=feat_dim,
      action_dim=action_dim,
      hidden=tuple(config.model.actor.hidden),
      init_std=config.model.actor.init_stddev,
      min_std=config.model.actor.min_stddev,
      mean_scale=config.model.actor.mean_scale,
    )
    self.critic = Critic(feat_dim, tuple(config.model.critic.hidden))

    if training:
      self._opt_wm     = optim.Adam(learning_rate=config.training.optimizer.wm_lr,     eps=config.training.optimizer.adam_eps)
      self._opt_actor  = optim.Adam(learning_rate=config.training.optimizer.actor_lr,  eps=config.training.optimizer.adam_eps)
      self._opt_critic = optim.Adam(learning_rate=config.training.optimizer.critic_lr, eps=config.training.optimizer.adam_eps)

      self._wm_bundle = _WMBundle(self.encoder, self.rssm, self.decoder, self.reward_head)
      self._wm_grad   = nn.value_and_grad(self._wm_bundle, self._wm_loss_fn)

  def _wm_loss_fn(self, wm_modules, batch):
    obs    = batch["obs"]      # (B, T, obs_dim)
    action = batch["action"]   # (B, T, action_dim)
    reward = batch["reward"]   # (B, T)
    B, T, _ = obs.shape

    embeds = wm_modules.encoder(obs)           # (B, T, embed_dim)
    state  = wm_modules.rssm.initial_state(B)
    states = []
    kls    = []
    for t in range(T):
      state = wm_modules.rssm.obs_step(state, action[:, t], embeds[:, t])
      states.append(state)
      kl_t = kl_gaussian(
        state["post_mean"],  state["post_std"],
        state["prior_mean"], state["prior_std"],
      ).sum(-1)                                # sum over latent dim -> (B,)
      kls.append(kl_t)

    feats    = mx.stack([feat(s) for s in states], axis=1)   # (B, T, feat_dim)
    recon    = wm_modules.decoder(feats)
    rew_pred = wm_modules.reward(feats)

    recon_loss = ((recon - obs) ** 2).sum(-1).mean()
    rew_loss   = ((rew_pred - reward) ** 2).mean()

    kl = mx.stack(kls, axis=1).mean()
    kl = mx.maximum(kl, mx.array(self.config.training.losses.free_nats))

    loss = (self.config.training.losses.recon_scale * recon_loss
         + self.config.training.losses.reward_scale * rew_loss
         + self.config.training.losses.kl_scale     * kl)

    aux = {"recon": recon_loss, "rew": rew_loss, "kl": kl, "post_states": states}
    return loss, aux

  def wm_update(self, batch, tracker=None):
    """Compute model loss and gradients for a batch of sequences."""
    (loss, aux), grads = self._wm_grad(self._wm_bundle, batch)

    grad_norm = None
    if self.config.training.optimizer.gradient_clipping > 0:
      max_norm = self.config.training.optimizer.gradient_clipping
      grads, grad_norm = optim.clip_grad_norm(grads, max_norm)
    self._opt_wm.update(self._wm_bundle, grads)

    mx.eval(self._wm_bundle.parameters(), loss)

    if tracker is not None:
      logs = {
        "wm/loss":  loss.item(),
        "wm/recon": aux["recon"].item(),
        "wm/rew":   aux["rew"].item(),
        "wm/kl":    aux["kl"].item(),
      }
      if grad_norm is not None:
        gn = grad_norm.item()
        logs["wm/grad_norm"]    = gn
        logs["wm/grad_clipped"] = float(gn > max_norm)
      tracker.log(logs)

    return aux["post_states"]
