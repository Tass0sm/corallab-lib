import pybullet as p
import math
import corallab_assets

from .robot_base import RobotBase


CUBOID_URDF_PATH = str(corallab_assets.get_resource_path("cuboid/cuboid.urdf"))


class Cuboid(RobotBase):
    def __init_robot__(self, p=p, urdf_override=None):
        self.eef_id = 0
        self.arm_num_dofs = 6
        self.arm_rest_poses = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self._p = p
        self.id = self._p.loadURDF(urdf_override or CUBOID_URDF_PATH, self.base_pos, self.base_ori,
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
