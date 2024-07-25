# Third Party
import torch

# cuRobo
from curobo.types.base import TensorDeviceType


class EnvFloor3D:

    def __init__(self, **kwargs):
        tensor_args = TensorDeviceType()

        self.config_file = None

        # create a world from a dictionary of objects
        # cuboid: {} # dictionary of objects that are cuboids
        # mesh: {} # dictionary of objects that are meshes
        self.config = {
            "cuboid": {
                "table": {"dims": [4, 4, 0.2], "pose": [0.0, 0.0, -0.2, 1, 0, 0, 0]}
            },
        }
