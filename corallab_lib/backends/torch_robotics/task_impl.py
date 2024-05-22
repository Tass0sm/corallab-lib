from torch_robotics.tasks import tasks
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from .env_impl import TorchRoboticsEnv
from .robot_impl import TorchRoboticsRobot


class TorchRoboticsTask:
    def __init__(
            self,
            id: str,
            env=None,
            robot=None,
            impl=None,
            tensor_args: dict = DEFAULT_TENSOR_ARGS,
            **kwargs
    ):
        assert isinstance(env, TorchRoboticsEnv) or env is None
        assert isinstance(robot, TorchRoboticsRobot) or robot is None

        if impl:
            self.task_impl = impl
        else:
            TaskClass = getattr(tasks, id)
            self.task_impl = TaskClass(
                env=env.env_impl,
                robot=robot.robot_impl,
                tensor_args=tensor_args,
                **kwargs
            )

    @classmethod
    def from_impl(cls, impl, **kwargs):
        # id is None
        return cls(None, impl=impl, **kwargs)

    def get_q_dim(self):
        return self.task_impl.robot.q_dim

    def get_q_min(self):
        return self.task_impl.robot.q_min.cpu()

    def get_q_max(self):
        return self.task_impl.robot.q_max.cpu()

    def random_coll_free_q(self, *args, **kwargs):
        return self.task_impl.random_coll_free_q(*args, **kwargs)

    def compute_collision(self, q, **kwargs):
        return self.task_impl.compute_collision(q, **kwargs)