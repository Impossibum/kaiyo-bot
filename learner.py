import wandb
import torch.jit
from torch.nn import Linear, Sequential, GELU
from torch import nn
from torch.nn.init import xavier_uniform_
from redis import Redis
from shared_agent import SharedAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from rlgym_tools.extra_obs.advanced_padder import AdvancedObsPadder
from rlgym.utils.gamestates import PlayerData, GameState
from N_Parser import NectoAction
import numpy as np
from kaiyo_rewards import KaiyoRewards
from pathlib import Path
import os
from torch import set_num_threads
set_num_threads(1)


if __name__ == "__main__":
    frame_skip = 8
    half_life_seconds = 8
    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))
    print(f"_gamma is: {gamma}")
    run_id = None
    config = dict(
        actor_lr=1e-4,
        critic_lr=1e-4,
        shared_lr=1e-4,
        n_steps=1_000_000,
        batch_size=100_000,
        minibatch_size=50_000,
        epochs=30,
        gamma=gamma,
        iterations_per_save=20,
        ent_coef=0.01,
    )

    wandb.login(key="key")
    logger = wandb.init(name="name", project="project name", entity="kaiyo", id=run_id, config=config)

    r = Redis(password="password")
    rollout_gen = RedisRolloutGenerator(r, lambda: AdvancedObsPadder(team_size=3, expanding=True), lambda: KaiyoRewards(), lambda: NectoAction(),
                                        save_every=config.iterations_per_save, logger=logger, clear=run_id is None, max_age=1)

    shared = Sequential(Linear(237, 512), GELU(), Linear(512, 512), GELU())

    critic = Sequential(Linear(512, 512), GELU(), Linear(512, 512), GELU(), Linear(512, 512),
                        GELU(), Linear(512, 512), GELU(), Linear(512, 1))

    actor = Sequential(Linear(512, 512), GELU(), Linear(512, 512), GELU(), Linear(512, 90))

    actor = DiscretePolicy(actor, (90,))

    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": config.actor_lr},
        {"params": critic.parameters(), "lr": config.critic_lr},
        {"params": shared.parameters(), "lr": config.shared_lr}
    ])

    agent = SharedAgent(actor=actor, critic=critic, shared=shared, optimizer=optim)
    model_parameters = filter(lambda p: p.requires_grad, agent.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"There are {params} trainable parameters")
    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=config.ent_coef,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        minibatch_size=config.minibatch_size,
        epochs=config.epochs,
        gamma=config.gamma,
        logger=logger,
        zero_grads_with_none=True,
    )

    alg.run(iterations_per_save=config.iterations_per_save, save_dir="kaiyo-bot")



