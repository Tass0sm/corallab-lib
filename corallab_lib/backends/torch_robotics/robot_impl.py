from corallab_lib import Robot

from torch_robotics import robots
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch

from ..robot_interface import RobotInterface


class TorchRoboticsRobot(RobotInterface):
    def __init__(
            self,
            id: str,
            impl = None,
            tensor_args: dict = DEFAULT_TENSOR_ARGS,
            **kwargs
    ):
        self.tensor_args = tensor_args

        if impl:
            self.robot_impl = impl
        else:
            RobotClass = getattr(robots, id)
            self.robot_impl = RobotClass(
                **kwargs,
                tensor_args=tensor_args
            )

    @classmethod
    def from_impl(cls, impl, **kwargs):
        # id is None
        return cls(None, impl=impl, **kwargs)

    @property
    def robot_id(self):
        return self.robot_impl.name

    @property
    def name(self):
        return self.robot_impl.name

    def get_position(self, trajs):
        return self.robot_impl.get_position(trajs)

    def get_velocity(self, trajs):
        return self.robot_impl.get_velocity(trajs)

    def get_n_dof(self):
        return self.robot_impl.q_dim

    def get_q_min(self):
        return self.robot_impl.q_min

    def get_q_max(self):
        return self.robot_impl.q_max

    def random_q(self, n_samples=10):
        return self.robot_impl.random_q(n_samples=n_samples)

    def fk(self, q, **kwargs):
        return self.robot_impl.fk_map_collision(q, **kwargs)

    def get_margins(self):
        return to_torch(self.robot_impl.link_margins_for_object_collision_checking, **self.tensor_args)

    # Multi-Agent API

    def is_multi_agent(self):
        return self.robot_impl.name == "MultiRobot"

    def get_subrobots(self):
        robs = []

        for subrob in self.robot_impl.subrobots:
            csubrobot = Robot(
                from_impl=subrob,
                backend="torch_robotics"
            )

            robs.append(csubrobot)

        return robs

    def separate_joint_state(self, q):
        states = []

        for i, r in enumerate(self.robot_impl.subrobots):
            subrobot_state = r.get_position(joint_state)
            states.append(subrobot_state)

            joint_state = joint_state[..., r.get_n_dof():]

        return states
