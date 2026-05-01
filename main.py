#!/usr/bin/env python3
"""
Chuck Dreamer: A Robotics and Machine Learning Project

This is the main entry point for the project. It provides a CLI interface
for training models, processing data, and running experiments.
"""

import click
import logging

import numpy as np
from omegaconf import DictConfig

from chuck_dreamer.config import load_config, merge_overrides

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', type=str, help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Chuck Dreamer CLI - Robotics ML with MLX."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)


def _resolve_cfg(ctx, overrides: dict) -> DictConfig:
  """Merge the loaded config with a nested dict of CLI overrides, defaulting seed if unset."""
  cfg = merge_overrides(ctx.obj["config"], overrides)
  if cfg.get("seed") is None:
    cfg.seed = int(np.random.default_rng().integers(0, 2**31))
  return cfg


@cli.command("generate-scenes")
@click.option("--episodes", default=10, type=int, help="Number of episodes to collect")
@click.option("--output", default=None, type=str, help="Output directory")
@click.option("--difficulty", default=None, type=str, help="Scene difficulty (easy/medium/hard)")
@click.option("--render-size", default=None, type=str, help="Render size WxH (e.g. 128x128)")
@click.option("--seed", default=None, type=int, help="Random seed")
@click.option("--max-steps", default=None, type=int, help="Per-episode step cap (overrides scene config)")
@click.option("--format", "fmt", default=None, type=click.Choice(["hdf5", "rerun"]),
              help="Episode output format (hdf5 or rerun)")
@click.pass_context
def generate_scenes(ctx, episodes, output, difficulty, render_size, seed, max_steps, fmt):
  """Generate pushing scenes using the scripted heuristic policy."""
  from dataclasses import asdict
  from tqdm import tqdm

  from chuck_dreamer.sim import (
    EpisodeCollector,
    EpisodeWriter,
    PushingEnv,
    ScriptedPolicy,
  )

  cfg = _resolve_cfg(ctx, {
    "seed": seed,
    "sim": {
      "output_dir": output,
      "difficulty": difficulty,
      "render_size": render_size,
      "format": fmt,
      "max_steps": max_steps,
    },
  })

  click.echo(f"Collecting {episodes} episodes → {cfg.sim.output_dir}  (difficulty={cfg.sim.difficulty}, format={cfg.sim.format}, seed={cfg.seed})")
  outcome_counts = {"done": 0, "terminated": 0, "timeout": 0, "crashed": 0}

  env    = PushingEnv(cfg)
  policy = ScriptedPolicy(auto_advance_from_ready=True)
  collector = EpisodeCollector(env, policy)
  writer = EpisodeWriter(cfg.sim.output_dir, format=cfg.sim.format)
  for ep_idx in tqdm(range(episodes), desc="Collecting"):
    scene = collector.reset()
    if cfg.sim.max_steps is not None:
      scene.max_steps = int(cfg.sim.max_steps)
    episode_data, outcome = collector.run()
    outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

    if episode_data is None:
      click.echo(f"[ep {ep_idx}] crashed on reset, skipping")
      continue

    writer.write_episode(
      episode_data,
      metadata={
        "config":  asdict(scene),
        "seed":    cfg.seed + ep_idx,
        "source":  "sim",
        "outcome": outcome,
        "goal_xy": scene.goal_pos,
      },
    )

  env.close()
  click.echo(f"Done. Outcomes: {outcome_counts}")


@cli.command("show-scene")
@click.option("--difficulty", default=None, type=str, help="Scene difficulty (easy/medium/hard)")
@click.option("--seed", default=None, type=int, help="Random seed (random if omitted)")
@click.option("--render-size", default=None, type=str, help="Render size WxH (e.g. 128x128)")
@click.option("--step-delay", default=0.05, type=float, help="Seconds to sleep between steps (default 0.05)")
@click.pass_context
def show_scene(ctx, difficulty, seed, render_size, step_delay):
  """Generate a scene and run it in the interactive MuJoCo viewer.

  Drives the scripted policy directly so the user can press Space to
  advance from ``ready`` → ``approach`` and see the heuristic push play
  out, with hint geoms drawn while the policy is in ``ready``.
  """
  import time
  import mujoco
  import mujoco.viewer

  from chuck_dreamer.sim import PushingEnv, ScriptedPolicy

  cfg = _resolve_cfg(ctx, {
    "seed": seed,
    "sim": {"difficulty": difficulty, "render_size": render_size},
  })

  click.echo(f"difficulty={cfg.sim.difficulty}  seed={cfg.seed}")
  env    = PushingEnv(cfg)
  policy = ScriptedPolicy()
  scene  = env.generate_scene()
  obs, _ = env.reset(scene=scene)
  policy.reset(scene)

  def key_callback(keycode):
    if keycode == 32 and policy.state == "ready":  # Space bar
      policy.advance_from_ready()
      print("Policy state changed: ready → approach")
    return True

  click.echo("Launching MuJoCo viewer — close the window to exit.")
  with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as v:
    while v.is_running():
      prev_state = policy.state
      action = policy.act(obs)
      obs, _, terminated, truncated, _ = env.step(action)
      if policy.state != prev_state:
        print(f"Policy state changed: {prev_state} → {policy.state}")

      policy.insert_hints(v)
      v.sync()
      if step_delay > 0:
        time.sleep(step_delay)

      if terminated or truncated or policy.state == "done":
        break

  env.close()


@cli.command("train")
@click.option("--warmup_path", default=None, type=str, help="Path to warmup episodes")
@click.option("--seed", default=None, type=int, help="Random seed (random if omitted)")
@click.pass_context
def train(ctx, warmup_path, seed):
  """Train a model using the specified configuration."""
  from chuck_dreamer.trainer import Trainer

  cfg = _resolve_cfg(ctx, {"seed": seed, "data": {"warmup_path": warmup_path}})
  click.echo(f"Training with config: {cfg}")
  trainer = Trainer(cfg)
  trainer.train()


if __name__ == "__main__":
    cli()
