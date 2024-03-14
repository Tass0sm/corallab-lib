from torch_robotics import tasks
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


class TorchRoboticsTask:
    def __init__(
            self,
            id: str,
            env,
            robot,
            tensor_args: dict = DEFAULT_TENSOR_ARGS
    ):
        TaskClass = getattr(tasks, id)
        self.task_impl = TaskClass(
            env=env,
            robot=robot,
            tensor_args=tensor_args
        )
