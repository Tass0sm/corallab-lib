from torch_robotics import environments
from torch_robotics.torch_utils.torch_utils import to_numpy, DEFAULT_TENSOR_ARGS


class TorchRoboticsEnv:
    def __init__(
            self,
            id: str,
            tensor_args: dict = DEFAULT_TENSOR_ARGS
    ):
        EnvClass = getattr(environments, id)
        self.env_impl = EnvClass(
            tensor_args=tensor_args
        )
