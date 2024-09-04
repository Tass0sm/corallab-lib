"""Credit to: https://github.com/ElectronicElephant/pybullet_ur5_robotiq"""

import pybullet as pb
import math
import numpy as np
import corallab_assets

from corallab_lib import Robot
from .robot_base import RobotBase
from .triple_ur5 import TripleUR5

TRIPLE_UR5_ASSET_PATH = str(corallab_assets.get_resource_path("triple_ur5"))
TRIPLE_UR5_URDF_PATH = str(corallab_assets.get_resource_path("triple_ur5/triple_ur5_with_grippers.urdf"))


class TripleUR5WithGrippers(TripleUR5):
    def __init_robot__(self, p=pb, urdf_override=None):
        self.eef_id = 6
        self.arm_num_dofs = 18
        self.arm_rest_poses = [
            3.1415, -2.2000, 1.9000, -1.3830, -1.5700,  0.0000,
            0.0000, -2.2000, 1.9000, -1.3830, -1.5700,  0.0000,
            3.1415, -2.2000, 1.9000, -1.3830, -1.5700, 0.0000
        ]

        self._p = p
        self._p.setAdditionalSearchPath(TRIPLE_UR5_ASSET_PATH)
        self.id = self._p.loadURDF(urdf_override or TRIPLE_UR5_URDF_PATH, self.base_pos, self.base_ori,
                                   useFixedBase=True, flags=pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

    def __post_load__(self):
        # To control the gripper
        mimic_parent_name = 'finger_joint_0'
        mimic_children_names = {'right_outer_knuckle_joint_0': 1,
                                'left_inner_knuckle_joint_0': 1,
                                'right_inner_knuckle_joint_0': 1,
                                'left_inner_finger_joint_0': -1,
                                'right_inner_finger_joint_0': -1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

        mimic_parent_name = 'finger_joint_1'
        mimic_children_names = {'right_outer_knuckle_joint_1': 1,
                                'left_inner_knuckle_joint_1': 1,
                                'right_inner_knuckle_joint_1': 1,
                                'left_inner_finger_joint_1': -1,
                                'right_inner_finger_joint_1': -1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

        mimic_parent_name = 'finger_joint_2'
        mimic_children_names = {'right_outer_knuckle_joint_2': 1,
                                'left_inner_knuckle_joint_2': 1,
                                'right_inner_knuckle_joint_2': 1,
                                'left_inner_finger_joint_2': -1,
                                'right_inner_finger_joint_2': -1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = self._p.createConstraint(self.id, self.mimic_parent_id,
                                         self.id, joint_id,
                                         jointType=pb.JOINT_GEAR,
                                         jointAxis=[0, 1, 0],
                                         parentFramePosition=[0, 0, 0],
                                         childFramePosition=[0, 0, 0])
            self._p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)  # Note: the mysterious `erp` is of EXTREME importance

    # Multi-Agent API

    def get_subrobots(self):
        base_link_0_idx = 0
        base_link_1_idx = 19
        base_link_1_idx = 38

        base_pos_0 = self._p.getLinkState(self.id, base_link_0_idx)[0]
        base_pos_1 = self._p.getLinkState(self.id, base_link_1_idx)[0]
        base_pos_2 = self._p.getLinkState(self.id, base_link_2_idx)[0]

        UR5_0 = Robot(
            "UR5",
            pos=base_pos_0,
            backend="pybullet"
        )
        UR5_0.set_id("UR5_0")

        UR5_1 = Robot(
            "UR5",
            pos=base_pos_1,
            backend="pybullet"
        )
        UR5_1.set_id("UR5_1")

        UR5_2 = Robot(
            "UR5",
            pos=base_pos_2,
            backend="pybullet"
        )
        UR5_2.set_id("UR5_2")

        return [UR5_0, UR5_1, UR5_2]
