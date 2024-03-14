from .backends import backend_list
import importlib


class BackendManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.backend = None

    def set_backend(self, backend: str):
        assert backend in backend_list
        self.backend = backend

    def get_backend(self):
        try:
            return importlib.import_module("." + self.backend, package="corallab_sim.backends")
        except KeyError:
            raise Exception("Need to select a backend!")

    def get_backend_attr(self, attr):
        backend_module = self.get_backend()
        return getattr(backend_module, attr)


backend_manager = BackendManager()
