import os

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig

from .utils import find_config_dict


class UR10:

    def __init__(self, base_pos=None, **kwargs):
        self.tensor_args = TensorDeviceType()

        config_file_basename = "ur10e.yml"
        config_dict = find_config_dict(config_file_basename)
        self.config = RobotConfig.from_dict(config_dict, self.tensor_args)

        if base_pos is not None:
            self.config.kinematics.kinematics_config.fixed_transforms[0, :3, -1] = base_pos

        self.kin_model = CudaRobotModel(self.config.kinematics)
        self.retract_config = self.kin_model.retract_config

    @property
    def robot_id(self):
        if self.base_pos is None:
            return "UR10"
        else:
            base_pos_str = str(self.base_pos.cpu().tolist())
            return f"UR10_at_{base_pos_str}"

    # Multi-Agent API
    def is_multi_agent(self):
        return False

    def get_subrobots(self):
        return []
