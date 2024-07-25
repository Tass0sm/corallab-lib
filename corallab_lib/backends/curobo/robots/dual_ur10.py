import os

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig

from corallab_lib import Robot
from .utils import find_config_dict


class DualUR10:

    def __init__(self, **kwargs):
        self.tensor_args = TensorDeviceType()

        config_file_basename = "dual_ur10/dual_ur10.py"
        config_dict = find_config_dict(config_file_basename)
        self.config = RobotConfig.from_dict(config_dict, self.tensor_args)
        self.kin_model = CudaRobotModel(self.config.kinematics)

    # Multi-Agent API
    def is_multi_agent(self):
        return True

    def get_subrobots(self):
        link_name_to_idx_map = self.kin_model.kinematics_config.link_name_to_idx_map
        fixed_transforms = self.kin_model.kinematics_config.fixed_transforms
        l0 = link_name_to_idx_map["base_link"]
        l1 = link_name_to_idx_map["base_link_1"]

        base_pos_0 = fixed_transforms[l0, :3, 3]
        base_pos_1 = fixed_transforms[l1, :3, 3]

        UR10_0 = Robot(
            "UR10",
            base_pos=base_pos_0,
            backend="curobo"
        )

        UR10_1 = Robot(
            "UR10",
            base_pos=base_pos_1,
            backend="curobo"
        )

        return [UR10_0, UR10_1]
