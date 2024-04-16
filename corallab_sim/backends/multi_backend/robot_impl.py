import numpy as np

from corallab_sim.backend_manager import backend_manager
from ..robot_interface import RobotInterface


class MultiBackendRobot(RobotInterface):
    def __init__(
            self,
            subrobot_args: list = [],
            **kwargs
    ):
        backends = backend_manager.get_backend_kwarg("backends")
        self.robot_impls = {}

        for backend in backends:
            backend_args, backend_kwargs = kwargs.get(backend, ([], {}))
            RobotImpl = backend_manager.get_backend_attr(
                "RobotImpl",
                backend=backend
            )

            robot_impl = RobotImpl(*backend_args, **backend_kwargs)
            self.robot_impls[backend] = robot_impl

    def get_backend_impl(self, backend):
        return self.robot_impls[backend]
