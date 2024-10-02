import numpy as np

import gymnasium as gym
from gymnasium.spaces.utils import flatdim

from ..gym_interface import GymInterface


class GymnasiumGym(GymInterface):

    def __init__(
            self,
            id: str,
            seed: int = 42,
            render_mode = None,
            **kwargs
    ):
        self.gym_impl = gym.make(id, render_mode=render_mode)

        self._obs, self._info = self.gym_impl.reset(seed=seed)

    @property
    def state_dim(self):
        return flatdim(self.gym_impl.observation_space)

    @property
    def action_dim(self):
        return flatdim(self.gym_impl.action_space)

    @property
    def name(self):
        return self.gym_impl.spec.id

    # def seed(self, x):
    #     self.gym_impl.seed(x)

    # def reset(self, **kwargs):
    #     return self.gym_impl.reset(**kwargs)

    # def step(self, action, **kwargs):
    #     return self.gym_impl.step(action, **kwargs)

    # def sample(self):
    #     return self.gym_impl.sample()

    # def render(self):
    #     return self.gym_impl.render()

    # def close(self):
    #     self.gym_impl.close()

    def reset(self):
        return self.gym_impl.reset()
