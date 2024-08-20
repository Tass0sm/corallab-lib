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
            # **kwargs
        )

    @property
    def q_dim(self):
        return self.robot_impl.arm_num_dofs

    def get_position(self, trajs):
        return trajs[..., :self.robot_impl.arm_num_dofs]

    def get_velocity(self, trajs):
        return trajs[..., self.robot_impl.arm_num_dofs:]

    def get_n_dof(self):
        return self.robot_impl.arm_num_dofs

    def get_q_dim(self):
        return self.robot_impl.arm_num_dofs

    def get_q_min(self):
        return np.array(self.robot_impl.arm_lower_limits)

    def get_q_max(self):
        return np.array(self.robot_impl.arm_upper_limits)

    def random_q(self, gen, n_samples):
        return self.robot_impl.random_q(gen, n_samples)

    def ik(self, pos, orn, max_niter=200):
        return self.robot_impl.ik(pos, orn, max_niter=200)

    def set_q(self, q):
        return self.robot_impl.set_q(q)

    def get_q(self):
        return self.robot_impl.get_q()

    def get_qd(self):
        return self.robot_impl.get_qd()

    def _for_states(self, q, per_state_f, **kwargs):
        assert q.ndim == 2

        results = []

        for q_i in q:
            result = per_state_f(q_i, **kwargs)
            results.append(result)


        return torch.tensor(results, **self.tensor_args)
