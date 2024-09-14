import os
import torch

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig

from corallab_lib import Robot
from .utils import find_config_dict


class TripleUR5eWithGrippers:

    def __init__(self, **kwargs):
        self.tensor_args = TensorDeviceType()

        config_file_basename = "triple_ur5/triple_ur5e_with_grippers.py"
        config_dict = find_config_dict(config_file_basename)
        self.config = RobotConfig.from_dict(config_dict, self.tensor_args)
        self.kin_model = CudaRobotModel(self.config.kinematics)
        self.retract_config = self.kin_model.retract_config

    # Multi-Agent API
    def is_multi_agent(self):
        return True

    def get_subrobots(self):
        link_name_to_idx_map = self.kin_model.kinematics_config.link_name_to_idx_map
        fixed_transforms = self.kin_model.kinematics_config.fixed_transforms
        l0 = link_name_to_idx_map["base_link_0"]
        l1 = link_name_to_idx_map["base_link_1"]
        l2 = link_name_to_idx_map["base_link_2"]

        base_pos_0 = fixed_transforms[l0, :3, 3]
        base_pos_1 = fixed_transforms[l1, :3, 3]
        base_pos_2 = fixed_transforms[l2, :3, 3]

        UR5_0 = Robot(
            "UR5e",
            base_pos=base_pos_0,
            backend="curobo"
        )
        UR5_0.set_id("UR5_0")

        UR5_1 = Robot(
            "UR5e",
            base_pos=base_pos_1,
            backend="curobo"
        )
        UR5_1.set_id("UR5_1")

        UR5_2 = Robot(
            "UR5e",
            base_pos=base_pos_2,
            backend="curobo"
        )
        UR5_2.set_id("UR5_2")

        return [UR5_0, UR5_1, UR5_2]
