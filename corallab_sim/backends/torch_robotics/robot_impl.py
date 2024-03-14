from torch_robotics import robots
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


class TorchRoboticsRobot:
    def __init__(
            self,
            id: str,
            tensor_args: dict = DEFAULT_TENSOR_ARGS
    ):
        RobotClass = getattr(robots, id)
        self.robot_impl = RobotClass(
            tensor_args=tensor_args
        )

    def random_q(self, n_samples=10):
        return self.robot_impl.random_q(n_samples=n_samples)
