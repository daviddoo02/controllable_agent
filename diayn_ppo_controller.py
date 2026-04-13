"""
DIAYN PPO Skill Controller
==========================
Post-pretraining controller for EECS 567 Team 14.

This script keeps DIAYN pretraining unchanged, loads a pretrained DIAYN
checkpoint, freezes the low-level DIAYN policy, and trains a PPO controller
over the discrete DIAYN skill space.

The high-level PPO policy picks a skill every `skill_horizon` environment
steps. The low-level DIAYN actor then executes continuous actions conditioned
on that fixed skill until the horizon expires or the episode terminates.

Usage (from the controllable_agent root):
    python diayn_ppo_controller.py \\
        --snapshot /path/to/snapshot_2000000.pt \\
        --task walker_walk \\
        --seed 1 \\
        --outdir /content/ppo_runs/diayn_ppo_walker_walk_s1
"""

import argparse
import csv
import json
import math
import os
import time
from collections import OrderedDict
from pathlib import Path
import typing as tp

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

os.environ['MUJOCO_GL'] = os.environ.get('MUJOCO_GL', 'egl')
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

from url_benchmark import dmc, utils


def make_env(task: str, seed: int = 1) -> dmc.EnvWrapper:
    return dmc.make(task, obs_type='states', frame_stack=1,
                    action_repeat=1, seed=seed)


def load_diayn_agent(snapshot_path: str, device: str = 'cuda'):
    """Load the pretrained DIAYN agent and place it on the requested device."""
    path = Path(snapshot_path)

    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    print(f"[Load] {snapshot_path}")

    with path.open('rb') as f:
        payload = torch.load(f, map_location=device, weights_only=False)

    agent = payload['agent']
    agent.device = torch.device(device)
    agent.reward_free = False
    return agent


def freeze_diayn_agent(agent) -> None:
    """Freeze the low-level DIAYN parameters so only PPO is trained."""
    modules = [
        agent.encoder,
        agent.actor,
        agent.critic,
        agent.critic_target,
    ]

    if getattr(agent, 'diayn', None) is not None:
        modules.append(agent.diayn)

    if getattr(agent, 'reward_model', None) is not None:
        modules.append(agent.reward_model)

    for module in modules:

        for param in module.parameters():
            param.requires_grad_(False)

        module.eval()


def make_skill_meta(skill_idx: int, skill_dim: int) -> OrderedDict:
    """Build the one-hot DIAYN skill input expected by the low-level actor."""
    skill = np.zeros(skill_dim, dtype=np.float32)
    skill[skill_idx] = 1.0
    meta = OrderedDict()
    meta['skill'] = skill
    return meta


class SkillController(nn.Module):
    """Discrete actor-critic network over the DIAYN skill set."""

    def __init__(self, obs_dim: int, skill_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, skill_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tp.Tuple[Categorical, torch.Tensor]:
        hidden = self.trunk(obs)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return Categorical(logits=logits), value


class RolloutBuffer:
    """On-policy storage for PPO high-level transitions."""

    def __init__(self) -> None:
        self.obs: tp.List[np.ndarray] = []
        self.actions: tp.List[int] = []
        self.log_probs: tp.List[float] = []
        self.values: tp.List[float] = []
        self.rewards: tp.List[float] = []
        self.discounts: tp.List[float] = []
        self.next_obs: tp.List[np.ndarray] = []
        self.dones: tp.List[float] = []

    def __len__(self) -> int:
        return len(self.obs)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        discount: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs.append(np.array(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.discounts.append(float(discount))
        self.next_obs.append(np.array(next_obs, dtype=np.float32))
        self.dones.append(float(done))

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.discounts.clear()
        self.next_obs.clear()
        self.dones.clear()


def sample_skill(
    controller: SkillController,
    obs: np.ndarray,
    device: torch.device,
) -> tp.Tuple[int, float, float]:
    """Sample one high-level DIAYN skill from the PPO controller."""
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        dist, value = controller(obs_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action)

    return int(action.item()), float(log_prob.item()), float(value.item())


def greedy_skill(
    controller: SkillController,
    obs: np.ndarray,
    device: torch.device,
) -> int:
    """Pick the most likely skill for evaluation."""
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        dist, _ = controller(obs_tensor)
        return int(torch.argmax(dist.logits, dim=-1).item())


def execute_skill(
    agent,
    env: dmc.EnvWrapper,
    obs: np.ndarray,
    skill_idx: int,
    skill_horizon: int,
    remaining_frames: int,
    global_frame: int,
    deterministic: bool,
) -> tp.Tuple[np.ndarray, float, bool, int]:
    """
    Roll out a fixed skill for up to `skill_horizon` env steps.

    Returns the next observation, cumulative reward, done flag, and the number
    of low-level environment frames consumed by this high-level action.
    """
    meta = make_skill_meta(skill_idx, agent.skill_dim)
    frames = 0
    reward_sum = 0.0
    time_step = None
    current_obs = obs
    max_steps = min(skill_horizon, remaining_frames)

    while frames < max_steps:

        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(
                current_obs,
                meta,
                global_frame + frames,
                eval_mode=deterministic,
            )

        time_step = env.step(action)
        reward_sum += float(time_step.reward)
        current_obs = np.array(time_step.observation, dtype=np.float32)
        frames += 1

        if time_step.last():
            break

    done = bool(time_step.last()) if time_step is not None else False
    return current_obs, reward_sum, done, frames


def run_eval(
    controller: SkillController,
    agent,
    env: dmc.EnvWrapper,
    skill_horizon: int,
    global_frame: int,
    device: torch.device,
    num_episodes: int = 5,
) -> float:
    """Evaluate the greedy PPO skill controller on downstream task reward."""
    total_reward = 0.0

    for _ in range(num_episodes):

        time_step = env.reset()
        obs = np.array(time_step.observation, dtype=np.float32)
        episode_done = False

        while not episode_done:

            skill_idx = greedy_skill(controller, obs, device)
            obs, reward, episode_done, _ = execute_skill(
                agent=agent,
                env=env,
                obs=obs,
                skill_idx=skill_idx,
                skill_horizon=skill_horizon,
                remaining_frames=skill_horizon,
                global_frame=global_frame,
                deterministic=True,
            )
            total_reward += reward

    return total_reward / num_episodes


def log_eval(csv_path: Path, frame: int, episode: int,
             reward: float, step: int) -> None:
    """Append one evaluation row to eval.csv."""
    with csv_path.open('a', newline='') as f:
        csv.writer(f).writerow([frame, episode, round(reward, 4), step])


def compute_gae(
    controller: SkillController,
    buffer: RolloutBuffer,
    device: torch.device,
    gae_lambda: float,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """Compute generalized advantage estimates over high-level transitions."""
    obs_tensor = torch.as_tensor(
        np.stack(buffer.next_obs), dtype=torch.float32, device=device
    )

    with torch.no_grad():
        _, next_values = controller(obs_tensor)

    values = torch.as_tensor(buffer.values, dtype=torch.float32, device=device)
    rewards = torch.as_tensor(buffer.rewards, dtype=torch.float32, device=device)
    discounts = torch.as_tensor(buffer.discounts, dtype=torch.float32, device=device)

    advantages = torch.zeros(len(buffer), dtype=torch.float32, device=device)
    gae = torch.zeros(1, dtype=torch.float32, device=device)

    for index in range(len(buffer) - 1, -1, -1):

        delta = rewards[index] + discounts[index] * next_values[index] - values[index]
        gae = delta + discounts[index] * gae_lambda * gae
        advantages[index] = gae

    returns = advantages + values
    return advantages, returns


def ppo_update(
    controller: SkillController,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    device: torch.device,
    gae_lambda: float,
    ppo_epochs: int,
    batch_size: int,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> tp.Dict[str, float]:
    """Run PPO updates on the collected high-level rollout."""
    advantages, returns = compute_gae(controller, buffer, device, gae_lambda)
    obs_tensor = torch.as_tensor(np.stack(buffer.obs), dtype=torch.float32, device=device)
    actions = torch.as_tensor(buffer.actions, dtype=torch.int64, device=device)
    old_log_probs = torch.as_tensor(buffer.log_probs, dtype=torch.float32, device=device)

    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    sample_count = obs_tensor.size(0)
    losses: tp.Dict[str, float] = {
        'policy_loss': 0.0,
        'value_loss': 0.0,
        'entropy': 0.0,
    }
    updates = 0

    for _ in range(ppo_epochs):

        permutation = torch.randperm(sample_count, device=device)

        for start in range(0, sample_count, batch_size):

            end = min(start + batch_size, sample_count)
            batch_idx = permutation[start:end]
            batch_obs = obs_tensor[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]

            dist, values = controller(batch_obs)
            new_log_probs = dist.log_prob(batch_actions)
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            policy_loss = -torch.min(
                ratio * batch_advantages,
                clipped_ratio * batch_advantages,
            ).mean()
            value_loss = F.mse_loss(values, batch_returns)
            entropy = dist.entropy().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), max_grad_norm)
            optimizer.step()

            losses['policy_loss'] += float(policy_loss.item())
            losses['value_loss'] += float(value_loss.item())
            losses['entropy'] += float(entropy.item())
            updates += 1

    if updates:

        for key in losses:
            losses[key] /= updates

    return losses


def train_controller(args) -> float:
    """Train a PPO skill controller on top of a frozen pretrained DIAYN agent."""
    device_name = args.device if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    print(f"\n{'=' * 64}")
    print("[Setup] DIAYN PPO Skill Controller")
    print(f"  Device        : {device_name}")
    print(f"  Task          : {args.task}   seed={args.seed}")
    print(f"  Snapshot      : {args.snapshot}")
    print(f"  Total frames  : {args.total_frames}")
    print(f"  Skill horizon : {args.skill_horizon}")
    print(f"{'=' * 64}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    eval_csv_path = outdir / 'eval.csv'
    summary_path = outdir / 'summary.json'

    utils.set_seed_everywhere(args.seed)
    train_env = make_env(args.task, seed=args.seed)
    eval_env = make_env(args.task, seed=args.seed + 100)

    agent = load_diayn_agent(args.snapshot, device=device_name)
    freeze_diayn_agent(agent)

    obs_dim = int(np.prod(train_env.observation_spec().shape))
    skill_dim = int(agent.skill_dim)
    controller = SkillController(obs_dim, skill_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=args.lr)

    with eval_csv_path.open('w', newline='') as f:
        csv.writer(f).writerow(['frame', 'episode', 'episode_reward', 'step'])

    time_step = train_env.reset()
    obs = np.array(time_step.observation, dtype=np.float32)
    buffer = RolloutBuffer()
    global_frame = 0
    global_episode = 0
    eval_count = 0
    train_episode_reward = 0.0
    last_losses: tp.Dict[str, float] = {}
    start_time = time.time()

    baseline_reward = run_eval(
        controller=controller,
        agent=agent,
        env=eval_env,
        skill_horizon=args.skill_horizon,
        global_frame=0,
        device=device,
        num_episodes=args.eval_episodes,
    )
    log_eval(eval_csv_path, frame=0, episode=eval_count,
             reward=baseline_reward, step=0)
    print(f"  [Eval @{0:>7} frames] reward = {baseline_reward:7.2f}")
    eval_count += 1

    next_eval_frame = args.eval_every

    while global_frame < args.total_frames:

        remaining_frames = args.total_frames - global_frame
        skill_idx, log_prob, value = sample_skill(controller, obs, device)
        next_obs, reward, done, frames_used = execute_skill(
            agent=agent,
            env=train_env,
            obs=obs,
            skill_idx=skill_idx,
            skill_horizon=args.skill_horizon,
            remaining_frames=remaining_frames,
            global_frame=global_frame,
            deterministic=True,
        )

        discount = 0.0 if done else float(args.gamma ** frames_used)
        buffer.add(
            obs=obs,
            action=skill_idx,
            log_prob=log_prob,
            value=value,
            reward=reward,
            discount=discount,
            next_obs=next_obs,
            done=done,
        )

        train_episode_reward += reward
        global_frame += frames_used
        obs = next_obs

        while global_frame >= next_eval_frame:

            eval_reward = run_eval(
                controller=controller,
                agent=agent,
                env=eval_env,
                skill_horizon=args.skill_horizon,
                global_frame=global_frame,
                device=device,
                num_episodes=args.eval_episodes,
            )
            log_eval(eval_csv_path, frame=global_frame, episode=eval_count,
                     reward=eval_reward, step=global_frame)
            elapsed = time.time() - start_time
            print(
                f"  [Eval @{global_frame:>7} frames] reward = {eval_reward:7.2f} "
                f"({elapsed:.0f}s elapsed)"
            )
            eval_count += 1
            next_eval_frame += args.eval_every

        if done:

            global_episode += 1
            print(
                f"  [Train Episode {global_episode:>4}] reward = {train_episode_reward:7.2f} "
                f"frames = {global_frame}"
            )
            time_step = train_env.reset()
            obs = np.array(time_step.observation, dtype=np.float32)
            train_episode_reward = 0.0

        if len(buffer) >= args.rollout_decisions or global_frame >= args.total_frames:

            last_losses = ppo_update(
                controller=controller,
                optimizer=optimizer,
                buffer=buffer,
                device=device,
                gae_lambda=args.gae_lambda,
                ppo_epochs=args.ppo_epochs,
                batch_size=args.batch_size,
                clip_eps=args.clip_eps,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm,
            )
            buffer.clear()

    final_reward = run_eval(
        controller=controller,
        agent=agent,
        env=eval_env,
        skill_horizon=args.skill_horizon,
        global_frame=global_frame,
        device=device,
        num_episodes=args.eval_episodes,
    )
    log_eval(eval_csv_path, frame=global_frame, episode=eval_count,
             reward=final_reward, step=global_frame)

    summary = {
        'task': args.task,
        'seed': args.seed,
        'total_frames': args.total_frames,
        'skill_horizon': args.skill_horizon,
        'baseline_reward': baseline_reward,
        'final_reward': final_reward,
        'last_losses': last_losses,
    }

    with summary_path.open('w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 64}")
    print("[Done]")
    print(f"  Task           : {args.task}   seed={args.seed}")
    print(f"  Baseline eval  : {baseline_reward:.2f}")
    print(f"  Final eval     : {final_reward:.2f}  "
          f"(delta={final_reward - baseline_reward:+.2f})")
    print(f"  Output         : {outdir}")
    print(f"{'=' * 64}\n")

    return final_reward


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a PPO controller over pretrained DIAYN skills',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--snapshot', required=True,
                        help='Path to pretrained DIAYN snapshot .pt file')
    parser.add_argument('--task', required=True,
                        help='DMC task name, e.g. walker_walk or cheetah_run')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--outdir', required=True,
                        help='Output directory for eval.csv and summary.json')
    parser.add_argument('--total_frames', type=int, default=100_000,
                        help='Total downstream frame budget')
    parser.add_argument('--skill_horizon', type=int, default=50,
                        help='Environment steps per PPO skill decision')
    parser.add_argument('--eval_every', type=int, default=4_000,
                        help='Frames between evaluation rollouts')
    parser.add_argument('--eval_episodes', type=int, default=5,
                        help='Episodes per evaluation rollout')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden width for the PPO controller network')
    parser.add_argument('--rollout_decisions', type=int, default=256,
                        help='High-level decisions collected before each PPO update')
    parser.add_argument('--ppo_epochs', type=int, default=4,
                        help='Number of PPO epochs per rollout')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Mini-batch size for PPO updates')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate for the PPO controller')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor over low-level environment frames')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda for PPO advantage estimation')
    parser.add_argument('--clip_eps', type=float, default=0.2,
                        help='PPO clipping epsilon')
    parser.add_argument('--value_coef', type=float, default=0.5,
                        help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy bonus coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Gradient clipping norm for PPO')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


if __name__ == '__main__':
    train_controller(parse_args())
