"""Credit to: https://github.com/ElectronicElephant/pybullet_ur5_robotiq"""

import pybullet as p
import math
import numpy as np
import corallab_assets

from .robot_base import RobotBase

TRIPLE_UR5_ASSET_PATH = str(corallab_assets.get_resource_path("triple_ur5"))
TRIPLE_UR5_URDF_PATH = str(corallab_assets.get_resource_path("triple_ur5/triple_ur5e.urdf"))


class TripleUR5e(RobotBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.retract_config = np.array([
            0.00, -2.2, 1.9, -1.383, -1.57, 0.00,
            0.00, -2.2, 1.9, -1.383, -1.57, 0.00,
            0.00, -2.2, 1.9, -1.383, -1.57, 0.00
        ])

    def __init_robot__(self, p=p, urdf_override=None):
        self.eef_id = 6
        self.arm_num_dofs = 18
        self.arm_rest_poses = [
            0.0000, -2.2000, 1.9000, -1.3830, -1.5700,  0.0000,
            0.0000, -2.2000, 1.9000, -1.3830, -1.5700,  0.0000,
            0.0000, -2.2000, 1.9000, -1.3830, -1.5700, 0.0000
        ]

        self._p = p
        self._p.setAdditionalSearchPath(TRIPLE_UR5_ASSET_PATH)
        self.id = self._p.loadURDF(urdf_override or TRIPLE_UR5_URDF_PATH, self.base_pos, self.base_ori,
                                   useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

    def __post_load__(self):
        pass

    def disable_collisions(self, objects):
        enableCollision = 0
        gripper_link_ids = [joint.id for joint in self.joints if ("pad" in joint.name or
                                                                  "finger" in joint.name or
                                                                  "knuckle" in joint.name)]
        print(objects)
        print(gripper_link_ids)

        for oid in objects:
            for l in gripper_link_ids:
                self._p.setCollisionFilterPair(self.id, int(oid), l, -1, enableCollision)

    def get_link_names(self):
        return ["ee_link_0", "ee_link_1", "ee_link_2"]
