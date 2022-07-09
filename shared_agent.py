
import torch as th
from torch import nn

from rocket_learn.agent.policy import Policy


class SharedAgent(nn.Module):
    def __init__(self, actor: Policy, critic: nn.Module, shared: nn.Module, optimizer: th.optim.Optimizer):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.shared = shared
        self.optimizer = optimizer

    def forward(self, *args, **kwargs):
        x = self.shared(*args, **kwargs)
        return self.actor(x), self.critic(x)
