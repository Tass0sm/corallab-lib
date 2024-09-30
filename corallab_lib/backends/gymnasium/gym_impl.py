import numpy as np

import gymnasium as gym

from ..gym_interface import GymInterface


class GymnasiumGym(GymInterface):

    def __init__(
            self,
            id: str,
            seed: int = 42,
            **kwargs
    ):
        self.gym_impl = gym.make(id, render_mode="human")

        self._obs, self._info = self.gym_impl.reset(seed=seed)

    # @property
    # def action_space(self):
    #     return self.gym_impl.action_space

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