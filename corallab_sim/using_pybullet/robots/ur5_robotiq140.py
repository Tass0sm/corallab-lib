"""Credit to: https://github.com/ElectronicElephant/pybullet_ur5_robotiq"""

import pybullet as p

from .ur5_robotiq85 import UR5Robotiq85


class UR5Robotiq140(UR5Robotiq85):
    def __init_robot__(self):
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                               -1.5707970583733368, 0.0009377758247187636]
        self.id = p.loadURDF('./urdf/ur5_robotiq_140.urdf', self.base_pos, self.base_ori,
                             useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.gripper_range = [0, 0.085]
        # TODO: It's weird to use the same range and the same formula to calculate open_angle as Robotiq85.

    def __post_load__(self):
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': -1,
                                'left_inner_knuckle_joint': -1,
                                'right_inner_knuckle_joint': -1,
                                'left_inner_finger_joint': 1,
                                'right_inner_finger_joint': 1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)
