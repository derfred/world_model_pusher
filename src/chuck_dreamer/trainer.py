import logging
import os
from collections import defaultdict

from .reward import build_reward_fn
from .sim.pushing_env import PushingEnv
from .sim.episode_collector import EpisodeCollector

from .training.episode_loader import processor_for
from .training.replay_buffer import ReplayBuffer
from .training.tracker import Tracker

from .dreamer import build_model, DreamerPolicy

logger = logging.getLogger(__name__)


class Trainer:
  def __init__(self, config):
    self.config = config
    self.env    = PushingEnv(config)

    self._replay_buffer = ReplayBuffer(
      capacity_steps=config.data.buffer_size,
      min_episode_len=config.training.min_episode_len,
      processor=processor_for(self.env.obs_mode),
      reward_fn=build_reward_fn(config.reward),
      seed=config.seed,
    )

    self.model  = build_model(config, obs_dim=15, action_dim=6)
    self.policy = DreamerPolicy(self.model)
    self.collector = EpisodeCollector(self.env, self.policy)
    self.tracker = Tracker(config)
    self.tracker.init()

  def _warmup(self):
    if not os.path.exists(self.config.data.warmup_path):
      logger.warning(f"Warmup path {self.config.data.warmup_path} does not exist. Skipping warmup.")
      return
    logger.info(f"Loading warmup episodes from {self.config.data.warmup_path} (format={self.config.data.warmup_format})")
    self._replay_buffer.load_sim_episodes(self.config.data.warmup_path, self.config.data.warmup_format, progress=True)

  def _collect_phase(self):
    collect_data = defaultdict(int)
    for _ in range(self.config.training.num_collect_episodes):
      scene = self.collector.reset()
      if self.config.sim.max_steps is not None:
        scene.max_steps = int(self.config.sim.max_steps)
      episode_data, outcome = self.collector.run()
      collect_data[outcome] += 1
      if episode_data is not None:
        self._replay_buffer.add_sim_episode(episode_data)
    logger.info(f"Collect phase: collected {self.config.training.num_collect_episodes} episodes with outcomes: {dict(collect_data)}")

  def _train_phase(self):
    print("starting train phase with replay buffer size:", len(self._replay_buffer))
    if not self._replay_buffer.can_sample(self.config.training.batch_size, self.config.training.seq_len):
      logger.warning("Buffer too small to sample (have %d steps); skipping train phase.", len(self._replay_buffer))
      return

    for epoch in range(self.config.training.num_gradient_steps):
      batch = self._replay_buffer.sample(self.config.training.batch_size, self.config.training.seq_len)

      # get the world model predictions for this batch
      self.model.wm_update(batch, tracker=self.tracker.derive({"phase": "train", "epoch": epoch}))

  def _eval_phase(self):
    pass

  def _checkpoint(self):
    pass

  def train(self):
    self._warmup()
    for i in range(self.config.training.num_iterations):
      self._collect_phase()
      self._train_phase()
      if i % self.config.training.eval_every == 0:
        self._eval_phase()
      if i % self.config.training.save_every == 0:
        self._checkpoint()


if __name__ == "__main__":
  from chuck_dreamer.config import load_config
  config = load_config()

  trainer = Trainer(config)
  trainer.train()
