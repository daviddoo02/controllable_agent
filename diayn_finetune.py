"""
DIAYN Skill-Selection Fine-tuning Script
=========================================
Phase 2 improvement for EECS 567 Team 14.

The standard URLB fine-tuning for DIAYN re-randomises the skill vector z at
the start of every episode, which averages gradient updates across 16 completely
different behaviours and destroys the pretrained skill structure.

Our approach — within the same 100k frame budget as URLB:
  Phase A  — Skill Selection (20% of total_frames = 20k frames by default)
             Distribute frames_per_skill = 20k // 16 = 1,250 frames per skill.
             Roll out each skill until its frame budget is exhausted.
             z* = argmax over mean episode reward.

  Phase B  — Skill-Conditioned Fine-tuning (80% of total_frames = 80k frames)
             Set reward_free=False  ->  agent.update() uses extrinsic task reward.
             Fix z=z* throughout all episodes. Log eval.csv every eval_every frames.

Total environment interactions = 100k frames (identical to URLB baseline).

Usage (from the controllable_agent root):
    python -m url_benchmark.diayn_finetune \\
        --snapshot /path/to/snapshot_2000000.pt \\
        --task walker_walk \\
        --seed 1 \\
        --outdir /content/finetune_runs/diayn_ss_walker_walk_s1

Optional flags (defaults shown):
    --total_frames        100000   total environment frame budget
    --selection_fraction     0.2   fraction for skill selection (20%)
    --eval_every            4000   frames between evaluations during fine-tuning
    --eval_episodes            5   episodes per evaluation rollout
    --device               cuda
"""

import os
import csv
import json
import argparse
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch

os.environ['MUJOCO_GL'] = os.environ.get('MUJOCO_GL', 'egl')
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

from url_benchmark import dmc, utils
from url_benchmark.in_memory_replay_buffer import ReplayBuffer


# ------------------------------------------------------------------------------
# Environment helper
# ------------------------------------------------------------------------------

def make_env(task: str, seed: int = 1) -> dmc.EnvWrapper:
    return dmc.make(task, obs_type='states', frame_stack=1,
                    action_repeat=1, seed=seed)


# ------------------------------------------------------------------------------
# Checkpoint loading
# ------------------------------------------------------------------------------

def load_diayn_agent(snapshot_path: str, device: str = 'cuda'):
    """
    Load a DIAYNAgent from a pretrain.py checkpoint.

    Checkpoint format (from BaseWorkspace.save_checkpoint):
        {'agent': DIAYNAgent, 'global_step': int,
         'global_episode': int, 'replay_loader': ReplayBuffer}
    """
    path = Path(snapshot_path)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    print(f"[Load] {snapshot_path}")
    with path.open('rb') as f:
        payload = torch.load(f, map_location=device, weights_only=False)

    agent = payload['agent']
    agent.device = torch.device(device)

    # Switch to extrinsic reward so agent.update() uses task reward
    agent.reward_free = False

    print(f"[Load] OK  skill_dim={agent.skill_dim}")
    return agent


# ------------------------------------------------------------------------------
# Skill meta helper
# ------------------------------------------------------------------------------

def make_skill_meta(skill_idx: int, skill_dim: int) -> OrderedDict:
    """Return the one-hot skill meta dict that the DIAYN actor expects."""
    skill = np.zeros(skill_dim, dtype=np.float32)
    skill[skill_idx] = 1.0
    meta = OrderedDict()
    meta['skill'] = skill
    return meta


# ------------------------------------------------------------------------------
# Phase A: Skill Selection (frame-budgeted)
# ------------------------------------------------------------------------------

def evaluate_skill_budget(agent, env: dmc.EnvWrapper,
                           skill_idx: int, frame_budget: int,
                           device: str = 'cuda') -> float:
    """
    Roll out skill `skill_idx` until `frame_budget` environment steps are used.

    Partial episodes at the boundary are included in the mean if at least one
    step was taken. Returns mean episode reward across all episodes attempted.

    This is the key methodological improvement over a fixed episode count:
    every skill gets exactly the same number of environment interactions,
    regardless of episode length variability.
    """
    meta = make_skill_meta(skill_idx, agent.skill_dim)
    frames_used     = 0
    episode_rewards = []

    while frames_used < frame_budget:
        ts = env.reset()
        ep_reward = 0.0
        ep_frames  = 0

        while not ts.last() and frames_used < frame_budget:
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(ts.observation, meta, 0, eval_mode=True)
            ts = env.step(action)
            ep_reward  += ts.reward
            frames_used += 1
            ep_frames   += 1

        if ep_frames > 0:
            episode_rewards.append(ep_reward)

    return float(np.mean(episode_rewards)) if episode_rewards else 0.0


def select_best_skill(agent, env: dmc.EnvWrapper,
                       selection_frames: int,
                       device: str = 'cuda'):
    """
    Distribute selection_frames evenly across all skills.
    Returns (best_skill_idx, list_of_per_skill_mean_rewards).

    Each skill receives floor(selection_frames / skill_dim) frames.
    Any remainder is unused (at most skill_dim - 1 frames, negligible).
    """
    skill_dim        = agent.skill_dim
    frames_per_skill = selection_frames // skill_dim

    print(f"\n{'='*64}")
    print(f"[Skill Selection]")
    print(f"  Skills          : {skill_dim}")
    print(f"  Total budget    : {selection_frames} frames")
    print(f"  Per-skill budget: {frames_per_skill} frames "
          f"(~{frames_per_skill / 1000:.1f} episodes of 1000 steps)")
    print(f"{'='*64}")

    rewards = []
    for z in range(skill_dim):
        r = evaluate_skill_budget(agent, env, z, frames_per_skill, device)
        rewards.append(r)
        marker = " <-- current best" if r == max(rewards) else ""
        print(f"  skill {z:2d}: mean_reward = {r:7.2f}{marker}")

    best_idx  = int(np.argmax(rewards))
    mean_all  = float(np.mean(rewards))
    gain      = rewards[best_idx] / max(mean_all, 1e-6)

    print(f"\n[Skill Selection] RESULT")
    print(f"  Best skill : z={best_idx}  reward={rewards[best_idx]:.2f}")
    print(f"  Mean (all) : {mean_all:.2f}")
    print(f"  Gain vs mean: {gain:.2f}x")
    print(f"{'='*64}\n")

    return best_idx, rewards


# ------------------------------------------------------------------------------
# Evaluation rollout (used during Phase B)
# ------------------------------------------------------------------------------

def run_eval(agent, env: dmc.EnvWrapper, meta: OrderedDict,
             global_step: int, device: str = 'cuda',
             num_episodes: int = 5) -> float:
    """Run num_episodes with fixed meta and return mean episode reward."""
    total = 0.0
    for _ in range(num_episodes):
        ts = env.reset()
        while not ts.last():
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(ts.observation, meta,
                                   global_step, eval_mode=True)
            ts = env.step(action)
            total += ts.reward
    return total / num_episodes


# ------------------------------------------------------------------------------
# CSV logging helper
# ------------------------------------------------------------------------------

def log_eval(csv_path: Path, frame: int, episode: int,
             reward: float, step: int) -> None:
    """Append one row to eval.csv. Column format matches pretrain.py output."""
    with csv_path.open('a', newline='') as f:
        csv.writer(f).writerow([frame, episode, round(reward, 4), step])


# ------------------------------------------------------------------------------
# Main fine-tuning routine
# ------------------------------------------------------------------------------

def finetune(args):
    device = args.device if torch.cuda.is_available() else 'cpu'

    # Budget split — this is the key fairness constraint
    # Both phases together equal exactly args.total_frames (= URLB's 100k)
    selection_frames = int(args.total_frames * args.selection_fraction)
    finetune_frames  = args.total_frames - selection_frames

    print(f"\n{'='*64}")
    print(f"[Setup] DIAYN Skill-Selection Fine-tuning")
    print(f"  Device          : {device}")
    print(f"  Task            : {args.task}   seed={args.seed}")
    print(f"  Snapshot        : {args.snapshot}")
    print(f"  Total budget    : {args.total_frames} frames  "
          f"(matches URLB 100k protocol)")
    print(f"  Selection (A)   : {selection_frames} frames  "
          f"({args.selection_fraction*100:.0f}%)")
    print(f"  Fine-tuning (B) : {finetune_frames} frames  "
          f"({(1-args.selection_fraction)*100:.0f}%)")
    print(f"{'='*64}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    eval_csv_path = outdir / 'eval.csv'

    utils.set_seed_everywhere(args.seed)

    # Separate envs so eval rollouts don't contaminate training state
    train_env = make_env(args.task, seed=args.seed)
    eval_env  = make_env(args.task, seed=args.seed + 100)

    # Load pretrained agent
    agent = load_diayn_agent(args.snapshot, device=device)

    # ------------------------------------------------------------------
    # Phase A: Skill Selection
    # ------------------------------------------------------------------
    best_skill_idx, all_rewards = select_best_skill(
        agent, eval_env,
        selection_frames=selection_frames,
        device=device,
    )

    # Persist selection results for notebook analysis
    with (outdir / 'skill_selection.json').open('w') as f:
        json.dump({
            'task':               args.task,
            'seed':               args.seed,
            'total_frames':       args.total_frames,
            'selection_frames':   selection_frames,
            'finetune_frames':    finetune_frames,
            'selection_fraction': args.selection_fraction,
            'best_skill':         best_skill_idx,
            'all_rewards':        {str(i): float(r)
                                   for i, r in enumerate(all_rewards)},
        }, f, indent=2)

    # Fixed meta dict — reused for every step of fine-tuning
    fixed_meta = make_skill_meta(best_skill_idx, agent.skill_dim)

    # ------------------------------------------------------------------
    # Phase B: Fine-tuning with fixed z*
    # ------------------------------------------------------------------
    print(f"[Fine-tuning] Starting  frames={finetune_frames}  "
          f"z={best_skill_idx}  eval_every={args.eval_every}")

    agent.reward_free = False   # belt-and-suspenders
    agent.train()

    # Fresh replay buffer — no pretrain data carried over
    replay_buffer = ReplayBuffer(max_episodes=2000, discount=0.99, future=1.0)

    # Initialise CSV
    with eval_csv_path.open('w', newline='') as f:
        csv.writer(f).writerow(['frame', 'episode', 'episode_reward', 'step'])

    global_step    = 0
    global_episode = 0
    eval_count     = 0
    episode_step   = 0
    episode_reward = 0.0
    seed_frames    = 4000   # collect experience before first gradient update

    # Prime the replay buffer
    ts = train_env.reset()
    replay_buffer.add(ts, fixed_meta)

    # Baseline eval — reward before any gradient updates (frame 0 of phase B)
    r0 = run_eval(agent, eval_env, fixed_meta, 0, device, args.eval_episodes)
    log_eval(eval_csv_path, frame=0, episode=eval_count, reward=r0, step=0)
    print(f"  [Eval @{0:>7} frames]  reward = {r0:7.2f}  (pre-finetune baseline)")
    eval_count += 1

    t0 = time.time()

    while global_step < finetune_frames:

        # End of episode
        if ts.last():
            global_episode += 1
            ts = train_env.reset()
            replay_buffer.add(ts, fixed_meta)
            episode_reward = 0.0
            episode_step   = 0

        # Periodic evaluation
        if global_step > 0 and global_step % args.eval_every == 0:
            r = run_eval(agent, eval_env, fixed_meta,
                         global_step, device, args.eval_episodes)
            log_eval(eval_csv_path, frame=global_step, episode=eval_count,
                     reward=r, step=global_step)
            elapsed = time.time() - t0
            pct     = 100 * global_step / finetune_frames
            print(f"  [Eval @{global_step:>7} frames]  "
                  f"reward = {r:7.2f}  "
                  f"({pct:.0f}% done, {elapsed:.0f}s)")
            eval_count += 1

        # Act with fixed skill z*
        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(ts.observation, fixed_meta,
                               global_step, eval_mode=False)

        # Gradient update after seed period
        # agent.update() uses extrinsic reward because reward_free=False
        if global_step >= seed_frames:
            agent.update(replay_buffer, global_step)

        # Step environment
        ts = train_env.step(action)
        replay_buffer.add(ts, fixed_meta)
        episode_reward += ts.reward
        episode_step   += 1
        global_step    += 1

    # Final evaluation
    r_final = run_eval(agent, eval_env, fixed_meta,
                       global_step, device, args.eval_episodes)
    log_eval(eval_csv_path, frame=global_step, episode=eval_count,
             reward=r_final, step=global_step)

    total_time = time.time() - t0
    print(f"\n{'='*64}")
    print(f"[Done]")
    print(f"  Task              : {args.task}   seed={args.seed}")
    print(f"  Best skill        : z={best_skill_idx}  "
          f"(selection reward={all_rewards[best_skill_idx]:.2f})")
    print(f"  Pre-finetune eval : {r0:.2f}")
    print(f"  Final eval        : {r_final:.2f}  (delta={r_final - r0:+.2f})")
    print(f"  Wall time         : {total_time:.0f}s")
    print(f"  Output            : {outdir}")
    print(f"{'='*64}\n")

    return r_final


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='DIAYN Skill-Selection Fine-tuning (Phase 2, EECS 567 Team 14)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument('--snapshot',  required=True,
                   help='Path to pretrained DIAYN snapshot .pt file')
    p.add_argument('--task',      required=True,
                   help='DMC task name, e.g. walker_walk or cheetah_run')
    p.add_argument('--seed',      type=int, default=1)
    p.add_argument('--outdir',    required=True,
                   help='Output directory for eval.csv and skill_selection.json')

    # Single frame budget — both phases draw from this pool
    p.add_argument('--total_frames', type=int, default=100_000,
                   help='Total environment frame budget (matches URLB 100k protocol)')
    p.add_argument('--selection_fraction', type=float, default=0.2,
                   help='Fraction of total_frames for skill selection '
                        '(0.2 => 20k selection + 80k fine-tuning)')

    # Evaluation
    p.add_argument('--eval_every',    type=int, default=4_000,
                   help='Fine-tuning frames between evaluation rollouts')
    p.add_argument('--eval_episodes', type=int, default=5,
                   help='Episodes per evaluation rollout')

    p.add_argument('--device', default='cuda')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    finetune(args)
