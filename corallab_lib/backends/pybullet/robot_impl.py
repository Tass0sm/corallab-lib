import numpy as np

from . import robots
from ..robot_interface import RobotInterface


class PybulletRobot(RobotInterface):
    def __init__(
            self,
            id: str,
            pos=np.array([0, 0, 0]),
            ori=np.array([0, 0, 0]),
            urdf_override=None,
            **kwargs
    ):
        self.urdf_override = urdf_override

        RobotClass = getattr(robots, id)
        self.robot_impl = RobotClass(
            pos, ori,
            **kwargs
        )

    def random_q(self, gen):
        return self.robot_impl.random_q(gen)

    def ik(self, pos, orn, max_niter=200):
        return self.robot_impl.ik(pos, orn, max_niter=200)

    def set_q(self, q):
        return self.robot_impl.set_q(q)

    def get_q(self):
        return self.robot_impl.get_q()

    def get_qd(self):
        return self.robot_impl.get_qd()
