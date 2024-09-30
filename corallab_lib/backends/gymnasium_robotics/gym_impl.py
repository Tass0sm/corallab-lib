import numpy as np

# from . import gyms

import gymnasium as gym
import gymnasium_robotics

from ..gym_interface import GymInterface


class GymnasiumRoboticsGym(GymInterface):

    def __init__(
            self,
            id: str,
            seed: int = 42,
            **kwargs
    ):
        assert id == "FetchPickAndPlace-v3"

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

    def reset(self):
        return self.gym_impl.reset()
