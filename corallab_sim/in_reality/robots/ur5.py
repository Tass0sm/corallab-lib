from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from corallab_sim.utilities.spatial import get_transform


class UR5:
    def __init__(
        self,
        ip_address,
        base_pos=(0, 0, 0),
        base_orn=(0, 0, 0, 1),
        gripper=None,
        move_timestep=0,
    ):
        """
        base_pos - position of robot base in world frame
        base_orn - orientation of robot base in world frame
        """
        self.con = RTDEControlInterface(ip_address)
        self.rec = RTDEReceiveInterface(ip_address)
        self.con.setTcp([0, 0, 0.145, 0, 0, 0])

        self.base_pos = base_pos
        self.base_orn = base_orn
        self.base_transform = get_transform(rotq=base_orn, pos=base_pos)
        self.move_timestep = move_timestep

    def move_q(self, configuration):
        self.con.moveJ(configuration)

    def test(self):
        actual_q = self.rec.getActualQ()

        position1 = [-0.343, -0.435, 0.50, -0.001, 3.12, 0.04]
        position2 = [-0.243, -0.335, 0.20, -0.001, 3.12, 0.04]

        self.con.moveL(position1)
        self.con.moveL(position2)
        self.con.stopL(10.0, False)

        if self.gripper is not None:
            print("Testing gripper...")
            self.gripper.move_and_wait_for_pos(255, 255, 255)
            log_info(self.gripper)
            self.gripper.move_and_wait_for_pos(0, 255, 255)
            log_info(self.gripper)

    def go_home(self):
        pass

    def enter_teach_mode(self):
        self.con.teachMode()

    def exit_teach_mode(self):
        self.con.endTeachMode()

    def convert_pose_to_base_frame(self, pos, rot):
        """convert position (meters given in the world frame) and rot (a scipy
        spatial rotation) to a pose (x, y, z, rx, ry, rz) in the robot's base
        frame

        T_w * p = T_b * p'
        T_b^(-1) * T_w * p = p'
        T_b^(-1) * I * p = p'
        T_b^(-1) * p = p'

        """

        T_b_inv = invert_transform(self.base_transform)
        R_b_inv = get_rotation(t_matrix=T_b_inv)

        # change basis of pos
        pos_B = transform_point(T_b_inv, pos)

        # change basis of orn
        new_rot = rot * R_b_inv
        rotvec = new_rot.as_rotvec(degrees=False)

        # combine
        pose = np.hstack([pos, rotvec])
        return pose

    def get_joint_positions(self):
        return self.rec.getActualQ()

    def ee_pose(self):
        fk = self.con.getForwardKinematics(
            self.rec.getActualQ(), self.con.getTCPOffset()
        )

        pos = fk[0:3]
        rot = R.from_rotvec(fk[3:6], degrees=False)
        q = rot.as_quat()

        return pos, q

    def move_ee(self, pos, orn, flip=True):
        """Move end effector to position POS (meters given in the world frame)
        with orientation ORN (a quaternion).

        In pybullet the orientations of the end effector are left-handed and
        defined with Z pointing into the robot. That way, you can give an
        object's orientation and the robot will assume an orientation that can
        interface that object from the top. On the robot, this frame is
        apparently flipped upside down. So, by default, we flip the input
        orientation by 180 degrees about its x-axis to get something that
        matches this convention.

        """

        if flip:
            rot = R.from_quat(orn)
            rot_x_180 = R.from_euler("xyz", [180, 0, 0], degrees=True)
            rot = rot * rot_x_180

        pose_base = self.convert_pose_to_base_frame(pos, rot)
        self.con.moveL(pose_base)

    def move_ee_down(self, pos, orn=(1, 0, 0, 0), **kwargs):
        """
        moves down from `pos` to z=0 until it detects object
        returns: pose=(pos, orn) at which it detected contact"""
        speed = 0.1
        assert speed > 0
        # self.con.moveUntilContact([0, 0, -speed, 0, 0, 0])
        return self.ee_pose()

    def move_ee_above(self, pos, orn=(1, 0, 0, 0), above_offt=(0, 0, 0.2)):
        a_pos = np.add(pos, above_offt)
        self.move_ee(a_pos, orn)

    def get_gripper_state(self):
        if self.gripper is not None:
            return self.gripper.get_state()

    def toggle_gripper(self):
        if self.gripper is not None:
            return self.gripper.toggle()

    def __del__(self):
        self.con.stopScript()
        print("Stopping robot control script")


def teach_and_record_trajectory(robot):
    """Set robot to freedrive mode, and repeatedly prompt user to set positions
    for a trajectory. Return the recorded list of poses."""

    poses = []
    configurations = []
    robot.enter_teach_mode()

    print("- Robot is now in free-drive mode.")
    print("- Move the robot to the desired positions.")
    print("- Record a pose with 'r' + RET")
    print("- Toggle the gripper with 't' + RET")
    print("- Quit the loop with 'q' + RET")

    i = 0
    gripper_action = 1
    while True:
        command = input("> ")

        try:
            if command == "r":
                pose = robot.ee_pose()
                poses.append((*pose, gripper_action))
                print(f"Recorded position {i}: {pose}, {gripper_action}")

                config = robot.get_joint_positions()
                configurations.append(config)
                print(f"Recorded configuration {i}")
                gripper_action = 1

                i += 1
            elif command == "t":
                new_state = robot.toggle_gripper()

                if new_state == 0:
                    # open
                    gripper_action = 0
                elif new_state == 1:
                    # close
                    gripper_action = 2
                else:
                    # no change
                    gripper_action = 1

                print(f"Gripper is now in state {new_state}")
            elif command == "q":
                break
            else:
                print("Unknown command")

        except KeyboardInterrupt:
            break
        except:
            print("Error: quitting")
            sys.exit()

    print(f"Recorded {i} poses")

    robot.exit_teach_mode()

    return poses, configurations
