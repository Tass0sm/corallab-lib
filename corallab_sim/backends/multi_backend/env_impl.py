import numpy as np

from corallab_sim.backend_manager import backend_manager


class MultiBackendEnv:
    def __init__(
            self,
            subrobot_args: list = [],
            **kwargs
    ):
        backends = backend_manager.get_backend_kwarg("backends")
        self.env_impls = {}

        for backend in backends:
            backend_args, backend_kwargs = kwargs.get(backend, ([], {}))
            EnvImpl = backend_manager.get_backend_attr(
                "EnvImpl",
                backend=backend
            )

            env_impl = EnvImpl(*backend_args, **backend_kwargs)
            self.env_impls[backend] = env_impl

    def get_backend_impl(self, backend):
        return self.env_impls[backend]
