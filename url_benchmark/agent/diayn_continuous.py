# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Continuous skill extension of DIAYN.
# Instead of discrete one-hot skills, skills are sampled from N(0, I).
# The discriminator is a regressor (MSE loss) instead of a classifier.
# Intrinsic reward = negative MSE between predicted and true skill.

import dataclasses
import typing as tp
from typing import Any, Dict, Tuple
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
import omegaconf

from url_benchmark import utils
from url_benchmark.dmc import TimeStep
from .ddpg import DDPGAgent, MetaDict, DDPGAgentConfig
from url_benchmark.in_memory_replay_buffer import ReplayBuffer


@dataclasses.dataclass
class DIAYNContinuousAgentConfig(DDPGAgentConfig):
    _target_: str = "url_benchmark.agent.diayn_continuous.DIAYNContinuousAgent"
    name: str = "diayn_continuous"
    update_encoder: bool = omegaconf.II("update_encoder")
    skill_dim: int = 16
    diayn_scale: float = 1.0
    update_skill_every_step: int = 50


cs = ConfigStore.instance()
cs.store(group="agent", name="diayn_continuous", node=DIAYNContinuousAgentConfig)


class DIAYNContinuous(nn.Module):
    """Discriminator network that regresses continuous skill z from observations."""
    def __init__(self, obs_dim: int, skill_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.skill_pred_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, skill_dim)  # outputs skill_dim values, no softmax
        )
        self.apply(utils.weight_init)

    def forward(self, obs) -> Any:
        return self.skill_pred_net(obs)


class DIAYNContinuousAgent(DDPGAgent):
    def __init__(self, **kwargs) -> None:
        cfg = DIAYNContinuousAgentConfig(**kwargs)

        # create actor and critic with augmented obs (obs + skill)
        super().__init__(**kwargs, meta_dim=cfg.skill_dim)
        self.cfg = cfg

        # create continuous discriminator
        self.diayn = DIAYNContinuous(
            self.obs_dim - self.skill_dim,
            self.skill_dim,
            kwargs['hidden_dim']
        ).to(kwargs['device'])

        # regression loss instead of classification
        self.diayn_criterion = nn.MSELoss()
        self.diayn_opt = torch.optim.Adam(self.diayn.parameters(), lr=self.lr)

        self.diayn.train()

    def init_meta(self) -> tp.Dict[str, np.ndarray]:
        # sample skill from standard Gaussian instead of one-hot
        skill = np.random.randn(self.cfg.skill_dim).astype(np.float32)
        skill = skill / (np.linalg.norm(skill) + 1e-8)
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    # pylint: disable=unused-argument
    def update_meta(
        self,
        meta: MetaDict,
        global_step: int,
        time_step: TimeStep,
        finetune: bool = False,
        replay_loader: tp.Optional[ReplayBuffer] = None
    ) -> MetaDict:
        if global_step % self.cfg.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    def compute_intr_reward(self, skill, next_obs, step) -> Any:
        predicted_z = self.diayn(next_obs)
        # reward = negative MSE: higher when discriminator can reconstruct skill well
        mse = F.mse_loss(predicted_z, skill, reduction='none').mean(dim=1)
        reward = -mse.reshape(-1, 1)
        return reward * self.cfg.diayn_scale

    def compute_diayn_loss(self, next_state, skill) -> Tuple[Any, Any]:
        predicted_z = self.diayn(next_state)
        loss = self.diayn_criterion(predicted_z, skill)
        # regression error as proxy for "accuracy" (lower = better)
        regression_error = loss.item()
        return loss, regression_error

    def update_diayn(self, skill, next_obs, step) -> Dict[str, Any]:
        metrics: tp.Dict[str, float] = {}

        loss, regression_error = self.compute_diayn_loss(next_obs, skill)

        self.diayn_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.diayn_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['diayn_loss'] = loss.item()
            metrics['diayn_regression_error'] = regression_error

        return metrics

    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        if step % self.update_every_steps != 0:
            return metrics

        batch = replay_loader.sample(self.cfg.batch_size).to(self.device)
        obs, action, extr_reward, discount, next_obs = batch.unpack()
        skill = batch.meta["skill"]

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            metrics.update(self.update_diayn(skill, next_obs, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(skill, next_obs, step)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.cfg.critic_target_tau)

        return metrics
