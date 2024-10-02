import numpy as np

import gymnasium_robotics

from ..gymnasium.gym_impl import GymnasiumGym


class GymnasiumRoboticsGym(GymnasiumGym):

    def __init__(
            self,
            id: str,
            seed: int = 42,
            render_mode: str = "human",
            **kwargs
    ):
        super().__init__(id, seed=seed, render_mode=render_mode)
