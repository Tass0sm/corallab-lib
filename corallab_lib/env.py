from .entity import Entity


class Env(Entity):
    def __init__(
            self,
            *args,
            backend=None,
            from_impl=None,
            **kwargs
    ):
        super().__init__("EnvImpl", *args, backend=backend, from_impl=from_impl, **kwargs)
