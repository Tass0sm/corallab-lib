from .backend_manager import backend_manager


class Gym:
    def __init__(
            self,
            *args,
            backend=None,
            from_impl=None,
            **kwargs
    ):
        GymImpl = backend_manager.get_backend_attr(
            "GymImpl",
            backend=backend
        )

        if from_impl:
            self.gym_impl = GymImpl.from_impl(from_impl, *args, **kwargs)
        else:
            self.gym_impl = GymImpl(*args, **kwargs)

    def __getattr__(self, name):
        if hasattr(self.gym_impl, name):
            return getattr(self.gym_impl, name)
        else:
            # Default behaviour
            raise AttributeError
