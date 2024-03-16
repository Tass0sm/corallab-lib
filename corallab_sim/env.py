from .backend_manager import backend_manager


class Env:
    def __init__(self, *args, **kwargs):
        EnvImpl = backend_manager.get_backend_attr("EnvImpl")
        self.env_impl = EnvImpl(*args, **kwargs)

    def __getattr__(self, name):
        if hasattr(self.env_impl, name):
            return getattr(self.env_impl, name)
        else:
            # Default behaviour
            raise AttributeError