import sys
from .backend_manager import backend_manager


class Task:
    """A task bundles together an env and a robot from a particular
    backend.

    """

    def __init__(
            self,
            *args,
            env=None,
            env_impl=None,
            robot=None,
            robot_impl=None,
            from_impl=None,
            backend=None,
            **kwargs
    ):
        TaskImpl = backend_manager.get_backend_attr(
            "TaskImpl",
            backend=backend
        )

        if from_impl:
            self.task_impl = TaskImpl.from_impl(
                from_impl,
                *args,
                env=env.env_impl,
                robot=robot.robot_impl,
                **kwargs
            )
        else:
            env_impl = env_impl or env.env_impl
            robot_impl = robot_impl or robot.robot_impl

            self.task_impl = TaskImpl(
                *args,
                env=env_impl,
                robot=robot_impl,
                **kwargs
            )

    def __getattr__(self, name):
        if hasattr(self.task_impl, name):
            return getattr(self.task_impl, name)
        else:
            # Default behaviour
            raise AttributeError
