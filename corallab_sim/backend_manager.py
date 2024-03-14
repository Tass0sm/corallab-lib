# import importlib
from .backends import backends_dict


class BackendManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.backend = None

    def set_backend(self, backend: str):
        assert backend in backends_dict
        self.backend = backend

    def get_backend(self):
        try:
            return backends_dict[self.backend]
        except KeyError:
            raise Exception("Need to select a backend!")

    def get_backend_attr(self, attr):
        backend_module = self.get_backend()
        return getattr(backend_module, attr)


backend_manager = BackendManager()
