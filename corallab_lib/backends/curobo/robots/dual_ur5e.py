import os
import torch

from corallab_lib import Robot
from .utils import RobotBase


class DualUR5e(RobotBase):

    def __init__(self, **kwargs):
        super().__init__("dual_ur5/dual_ur5e.py", **kwargs)

    # Multi-Agent API
    def is_multi_agent(self):
        return True

    def get_subrobots(self):
        link_name_to_idx_map = self.kin_model.kinematics_config.link_name_to_idx_map
        fixed_transforms = self.kin_model.kinematics_config.fixed_transforms
        l0 = link_name_to_idx_map["base_link_0"]
        l1 = link_name_to_idx_map["base_link_1"]

        base_pos_0 = fixed_transforms[l0, :3, 3]
        base_pos_1 = fixed_transforms[l1, :3, 3]

        UR5e_0 = Robot(
            "UR5e",
            base_pos=base_pos_0,
            backend="curobo"
        )

        UR5e_1 = Robot(
            "UR5e",
            base_pos=base_pos_1,
            backend="curobo"
        )

        return [UR5e_0, UR5e_1]
