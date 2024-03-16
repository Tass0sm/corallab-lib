"""Credit to: https://github.com/ElectronicElephant/pybullet_ur5_robotiq"""

import pybullet as p
import math
import corallab_assets

from .robot_base import RobotBase


UR5_URDF_PATH = str(corallab_assets.get_resource_path("ur5/ur5_robotiq_85.urdf"))


class UR5(RobotBase):
    def __init_robot__(self, urdf_override=None):
        self.eef_id = 6
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                               -1.5707970583733368, 0.0009377758247187636]

        self.id = p.loadURDF(urdf_override or UR5_URDF_PATH, self.base_pos, self.base_ori,
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
                p.setCollisionFilterPair(self.id, int(oid), l, -1, enableCollision)
