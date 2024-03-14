import sys
from .backend_manager import backend_manager


class Task:
    """A task bundles together an env and a robot from a particular
    backend.

    """

    def __init__(
            self,
            id: str,
            env,
            robot,
            **kwargs
    ):
        TaskImpl = backend_manager.get_backend_attr("TaskImpl")
        self.task_impl = TaskImpl(
            id,
            env.env_impl,
            robot.robot_impl,
            **kwargs
        )

    def __getattr__(self, name):
        if hasattr(self.task_impl, name):
            return getattr(self.task_impl, name)
        else:
            # Default behaviour
            raise AttributeError
