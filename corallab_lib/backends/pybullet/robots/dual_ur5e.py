"""Credit to: https://github.com/ElectronicElephant/pybullet_ur5_robotiq"""

import pybullet as pb
import math
import numpy as np
import corallab_assets

from corallab_lib import Robot
from .dual_ur5 import DualUR5

DUAL_UR5E_ASSET_PATH = str(corallab_assets.get_resource_path("dual_ur5"))
DUAL_UR5E_URDF_PATH = str(corallab_assets.get_resource_path("dual_ur5/dual_ur5e.urdf"))


class DualUR5e(DualUR5):
    def __init_robot__(self, p=pb, urdf_override=None):
        self.eef_id = 6
        self.arm_num_dofs = 12
        self.arm_rest_poses = [
            0.0000, -2.2000, 1.9000, -1.3830, -1.5700,  0.0000,
            0.0000, -2.2000, 1.9000, -1.3830, -1.5700,  0.0000
        ]

        self._p = p
        self._p.setAdditionalSearchPath(DUAL_UR5E_ASSET_PATH)
        self.id = self._p.loadURDF(urdf_override or DUAL_UR5E_URDF_PATH, self.base_pos, self.base_ori,
                                   useFixedBase=True, flags=pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    
    # Multi-Agent API
    # def get_subrobots(self):
    #     base_link_0_idx = 0
    #     base_link_1_idx = 19

    #     base_pos_0 = self._p.getLinkState(self.id, base_link_0_idx)[0]
    #     base_pos_1 = self._p.getLinkState(self.id, base_link_1_idx)[0]

    #     UR5_0 = Robot(
    #         "UR5",
    #         pos=base_pos_0,
    #         backend="pybullet"
    #     )
    #     UR5_0.set_id("UR5_0")

    #     UR5_1 = Robot(
    #         "UR5",
    #         pos=base_pos_1,
    #         backend="pybullet"
    #     )
    #     UR5_1.set_id("UR5_1")

    #     return [UR5_0, UR5_1]
