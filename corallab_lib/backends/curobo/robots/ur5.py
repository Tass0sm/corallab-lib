import os

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig

from .utils import find_config_dict


class UR5:

    def __init__(self, **kwargs):
        self.tensor_args = TensorDeviceType()

        config_file_basename = "ur5e.yml"
        config_dict = find_config_dict(config_file_basename)
        self.config = RobotConfig.from_dict(config_dict, self.tensor_args)
        self.kin_model = CudaRobotModel(self.config.kinematics)

    # Multi-Agent API
    def is_multi_agent(self):
        return False

    def get_subrobots(self):
        return []
