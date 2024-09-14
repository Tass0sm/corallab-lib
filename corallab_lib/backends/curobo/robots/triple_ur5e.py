import os
import torch

from corallab_lib import Robot
from .utils import RobotBase


class TripleUR5e(RobotBase):

    def __init__(self, **kwargs):
        super().__init__("triple_ur5/triple_ur5e.py", **kwargs)

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
            "UR5",
            base_pos=base_pos_0,
            backend="curobo"
        )

        UR5_1 = Robot(
            "UR5",
            base_pos=base_pos_1,
            backend="curobo"
        )

        UR5_2 = Robot(
            "UR5",
            base_pos=base_pos_2,
            backend="curobo"
        )

        return [UR5_0, UR5_1, UR5_2]
