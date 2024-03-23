import numpy as np

from corallab_sim.backend_manager import backend_manager
from .env_impl import MultiBackendEnv
from .robot_impl import MultiBackendRobot


class MultiBackendTask:
    def __init__(
            self,
            env=None,
            robot=None,
            **kwargs
    ):
        assert isinstance(env, MultiBackendEnv)
        assert isinstance(robot, MultiBackendRobot)

        backends = backend_manager.get_backend_kwarg("backends")
        self.task_impls = {}

        for backend in backends:
            backend_args, backend_kwargs = kwargs.get(backend, ([], {}))
            TaskImpl = backend_manager.get_backend_attr(
                "TaskImpl",
                backend=backend
            )

            task_impl = TaskImpl(
                *backend_args,
                env=env.get_backend_impl(backend),
                robot=robot.get_backend_impl(backend),
                **backend_kwargs
            )
            self.task_impls[backend] = task_impl

    def get_backend_impl(self, backend):
        return self.task_impls[backend]
