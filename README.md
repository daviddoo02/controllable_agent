[![CircleCI](https://dl.circleci.com/status-badge/img/gh/facebookresearch/controllable_agent/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/facebookresearch/controllable_agent/tree/main)

# Controllable Agent — EECS 567 Team 14

This repo extends the original [facebookresearch/controllable_agent](https://github.com/facebookresearch/controllable_agent) with URLB pretraining baselines for **FB-DDPG** and **DIAYN**, and three improvements to DIAYN's downstream adaptation within the URLB 100k fine-tuning budget.

Original work builds a controllable agent based on the Forward-Backward representation:
- A. Touati, J. Rapin, Y. Ollivier, [Does Zero-Shot Reinforcement Learning Exist?](https://arxiv.org/abs/2209.14935)
- A. Touati, Y. Ollivier, [Learning One Representation to Optimize All Rewards (Neurips 2021)](https://arxiv.org/abs/2103.07945)


## Setup

### Install (first time only)

```bash
source env.sh install ca
```

This installs MuJoCo 2.1.1 and all Python dependencies into a conda environment named `ca`.

### Activate

```bash
conda activate ca
```

### MuJoCo rendering

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco-2.1.1/lib
export MUJOCO_GL=egl   # use glfw if egl fails
```

**GPU note:** PyTorch 2.10+ dropped support for Pascal GPUs (GTX 1080 Ti, sm_61). If you have a Pascal GPU, install a compatible version:
```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```


## Phase 1 — Pretraining (2M frames, reward-free)

### FB-DDPG

```bash
python -m url_benchmark.pretrain agent=fb_ddpg task=walker_walk seed=1 use_tb=1
python -m url_benchmark.pretrain agent=fb_ddpg task=walker_walk seed=2 use_tb=1
python -m url_benchmark.pretrain agent=fb_ddpg task=walker_walk seed=3 use_tb=1

python -m url_benchmark.pretrain agent=fb_ddpg task=quadruped_walk seed=1 use_tb=1
python -m url_benchmark.pretrain agent=fb_ddpg task=quadruped_walk seed=2 use_tb=1
python -m url_benchmark.pretrain agent=fb_ddpg task=quadruped_walk seed=3 use_tb=1
```

### DIAYN

```bash
python -m url_benchmark.pretrain agent=diayn task=walker_walk seed=1 use_tb=1
python -m url_benchmark.pretrain agent=diayn task=walker_walk seed=2 use_tb=1
python -m url_benchmark.pretrain agent=diayn task=walker_walk seed=3 use_tb=1

python -m url_benchmark.pretrain agent=diayn task=cheetah_run seed=1 use_tb=1
python -m url_benchmark.pretrain agent=diayn task=cheetah_run seed=2 use_tb=1
python -m url_benchmark.pretrain agent=diayn task=cheetah_run seed=3 use_tb=1
```

Snapshots are saved to `exp_local/<timestamp>_<agent>_<task>_online/` at 100k, 200k, 500k, 800k, 1M, 1.5M, and 2M frames.

### Monitor with TensorBoard

```bash
tensorboard --logdir exp_local
```

### Check available config parameters

```bash
python -m url_benchmark.pretrain --cfg job
python -m url_benchmark.pretrain agent=fb_ddpg --cfg job
```


## Phase 2 — DIAYN Improvements (100k frame budget)

All three improvements take a pretrained DIAYN snapshot as input. Replace `<snapshot>` with the path to your `snapshot_2000000.pt`.

### Improvement 1 — DIAYN+SS (Skill Selection, zero-shot)

Evaluates all 16 skills with equal frame budgets and picks the best one. No gradient updates.

```bash
python run_diayn_eval_suite.py \
    --snapshot exp_local/<run>/snapshot_2000000.pt \
    --task walker_walk \
    --seed 1 \
    --outdir outputs/eval_suite/walker_walk_s1
```

### Improvement 2 — DIAYN+SS+FT (Skill Selection + Fine-tuning)

Spends 20k frames on skill selection, then fine-tunes the selected skill with task reward for 80k frames.

```bash
python diayn_finetune.py \
    --snapshot exp_local/<run>/snapshot_2000000.pt \
    --task walker_walk \
    --seed 1 \
    --outdir outputs/finetune/diayn_ss_walker_walk_s1
```

### Improvement 3 — DIAYN+PPO (PPO Skill Controller)

Freezes the pretrained DIAYN low-level policy and trains a PPO controller over the skill space for 100k frames.

```bash
python diayn_ppo_controller.py \
    --snapshot exp_local/<run>/snapshot_2000000.pt \
    --task walker_walk \
    --seed 1 \
    --outdir outputs/diayn_ppo/walker_walk/seed-1
```

### Run all three back-to-back

```bash
python run_diayn_eval_suite.py \
    --snapshot exp_local/<run>/snapshot_2000000.pt \
    --task walker_walk \
    --seed 1 \
    --outdir outputs/full_suite/walker_walk_s1 \
    --ppo_total_frames 100000
```


## Analysis Scripts

```bash
# PPO reward summary table (outputs ppo_reward_table.md)
python scripts/analysis/make_ppo_table.py --ppo_root outputs/diayn_ppo

# DIAYN vs PPO learning curves
python scripts/analysis/plot_diayn_vs_ppo.py --ppo_root outputs/diayn_ppo

# PPO-only learning curves
python scripts/analysis/plot_ppo_only.py --ppo_root outputs/diayn_ppo
```


## Output Structure

```
exp_local/
  <timestamp>_<agent>_<task>_online/
    eval.csv                  # eval reward every 10k frames
    train.csv                 # training metrics
    snapshot_<N>.pt           # checkpoint at N frames
    test_rewards.json         # downstream task rewards at end of training

outputs/
  diayn_ppo/<task>/seed-<N>/
    eval.csv
    summary.json
    checkpoint.pt
    best_checkpoint.pt

  finetune/<run>/
    eval.csv
    skill_selection.json      # selected skill and all per-skill rewards
```


## Agents

| Agent | Command | Description |
|---|---|---|
| FB-DDPG | `agent=fb_ddpg` | Forward-Backward representation, zero-shot adaptation |
| DIAYN | `agent=diayn` | Skill discovery via mutual information (discrete, 16 skills) |
| DIAYN Continuous | `agent=diayn_continuous` | Continuous Gaussian skill space (experimental) |
| RND | `agent=rnd` | Random Network Distillation |
| ICM | `agent=icm` | Intrinsic Curiosity Module |
| APS | `agent=aps` | Active Pre-Training with Successor features |
| ProtoRL | `agent=proto` | Prototypical RL |
| SMM | `agent=smm` | State Marginal Matching |


## Supported Tasks

| Domain | Tasks |
|---|---|
| `walker` | `stand`, `walk`, `run`, `flip` |
| `cheetah` | `run`, `walk`, `walk_backward`, `run_backward` |
| `quadruped` | `stand`, `walk`, `run`, `jump` |
| `jaco` | `reach_top_left`, `reach_top_right`, `reach_bottom_left`, `reach_bottom_right` |

### Offline Training
Here are some instructions to train FB agent offline on a dataset of transitions generated by RND in the Walker domain. 
+ Download RND dataset using the download.sh script from https://github.com/denisyarats/exorl.
```
./download.sh walker proto
```
+ Build the replay buffer and save it as torch pickle.

```python
from pathlib import Path
import torch
from controllable_agent import runner
hp = runner.HydraEntryPoint("url_benchmark/pretrain.py")
buffer_dir = Path("./walker/rnd/buffer/")
ws = hp.workspace(task="walker_walk", replay_buffer_episodes=5000)
ws.replay_loader.load(ws.train_env, buffer_dir, relabel=True)
with Path("walker/rnd/replay.pt").open('wb') as f:
    torch.save(ws.replay_loader, f)
```
+ Finally, train FB using the stored replay buffer
```
python -m url_benchmark.train_online  agent=fb_ddpg task=walker_walker load_replay_buffer=./walker/rnd/replay.pt
```

#### Using hiplot for Monitoring Training
From your device containing the logs, run the following command from the root folder: \
`python -m hiplot url_benchmark.hiplogs.load --port=XXXX`

Then connect to the path that is printed (make sure you have forwarded your port if you don't have the logs locally), and print the folder containing the logs in the text box. The server will parse the folder recursively and plot all train.csv and eval.csv files.



## Demo

A demo is available at [`https://controllable-agent.metademolab.com/`](https://controllable-agent.metademolab.com/) for testing custom rewards on the walker agent.

The demo is based on a replay buffer generated through:
`python -m url_benchmark.anytrain reward_free=1 num_train_episodes=2000 replay_buffer_episodes=2000 agent=fb_ddpg task=walker_walk goal_space=walker_pos_speed_z append_goal_to_observation=1 update_replay_buffer=1 load_replay_buffer=...`
with a replay buffer generated through `rnd`.


### Overview

The agent was trained in the Walker environment. We follow the algorithms outlined in the papers, with a restricted control space of 6 variables: x, z, vx, vz, up, am (horizontal and vertical positions of the torso, horizontal and vertical velocities, cosine of torso angle, angular momentum). We also augment the replay buffer dynamically by following the learned policies with various z parameters.

Again, this is a single agent, it wasn't trained on any of those rewards, and there is no finetuning when the reward function is specified. Based on the reward function, a task parameter is computed via an explicit formula, and a policy is applied using this task parameter.

By varying the reward function, we can train the agent to optimize various combinations of variables, as can be seen below. Multiplicative rewards are the easiest way to mix several constraints. 

Rewards must be provided as a Python equation. Here are a few reward examples:
- `vx`: run as fast as possible
- `x < -4`: go to the left until x<-4
- `1 / z`: be close to the ground
- `-up`: be upside down
- `-up * x * (x > 0)`: be to the right and upside down
- `exp(-abs(x - 8)) * up / z`: be around x=8, upright, and close to the ground: crouch at x=8
- `exp(-abs(x - 10)) * up * z**4`: be around x=10, upright, and very high: jump at x=10
- `vx/z**2`: crawl
- `exp(-abs(vx - 2)) * up`: move slowly (speed=2) and stay upright
- `vx * (1 - up) / z`: move as fast as possible, upside down, close to the ground
- `-am * exp (-abs(x - 10))`: go to x=10 and do backward spins
- `vx * (1 + up * cos(x / 4))`: run upright or rolling depending on cos(x/4)


### Running the Demo Locally

Provided you have access to a replay buffer and model checkpoint:

1. Create/activate your environment
2. Update the `CASES` variable to have one entry pointing to your checkpoint
3. From the root of the repo, run `streamlit run demo/main.py --server.port=8501`
4. Connect instead to `localhost:8501` in your browser (don't forget port forwarding if the demo runs on a server)
5. write a formula to be maximized as Python code.


## Contributing 

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
`controllable_agent` is MIT licensed, as found in the LICENSE file.