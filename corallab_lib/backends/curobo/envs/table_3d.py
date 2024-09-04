# Third Party
import torch

# cuRobo
from curobo.types.base import TensorDeviceType


class EnvTable3D:

    def __init__(self, **kwargs):
        tensor_args = TensorDeviceType()

        self.config_file = None

        self.table_center = torch.tensor([0.0, 0.0, 0.0])
        self.table_pose = [*self.table_center.tolist(), 1., 0., 0., 0.]
        self.table_dimensions = torch.tensor([0.3, 1.2, 0.2])
        table_half_dimensions = self.table_dimensions / 2

        self.table_bounds = torch.vstack([self.table_center - table_half_dimensions,
                                          self.table_center + table_half_dimensions])
        self.table_height = self.table_bounds[1, 2]

        # create a world from a dictionary of objects
        # cuboid: {} # dictionary of objects that are cuboids
        # mesh: {} # dictionary of objects that are meshes
        self.config = {
            "cuboid": {
                "floor": {"dims": [6, 6, 0.2], "pose": [0.0, 0.0, -0.2, 1, 0, 0, 0]},
                "table": {"dims": self.table_dimensions.tolist(), "pose": self.table_pose}
            },
        }

