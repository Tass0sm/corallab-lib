import numpy as np

from . import robots
from ..robot_interface import RobotInterface


class PybulletRobot(RobotInterface):
    def __init__(
            self,
            id: str,
            pos=np.array([0, 0, 0]),
            ori=np.array([0, 0, 0]),
            **kwargs
    ):
        RobotClass = getattr(robots, id)
        self.robot_impl = RobotClass(
            pos, ori,
            **kwargs
        )

    def random_q(self, n_samples=10):
        return self.robot_impl.random_q(n_samples=n_samples)
