# Third Party
import torch

# cuRobo
from curobo.types.base import TensorDeviceType


class EnvOpen3D:

    def __init__(self, **kwargs):
        tensor_args = TensorDeviceType()

        self.config_file = None

        # create a world from a dictionary of objects
        self.config = {}
