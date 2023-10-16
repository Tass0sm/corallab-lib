"""Module to control Robotiq's grippers - tested with HAND-E"""

import socket
import threading
import time
import math
import numpy as np
import pybullet as p
from enum import Enum
from typing import Union, Tuple, OrderedDict, Iterable
from corallab_sim.robots.gripper import Gripper
from corallab_sim.utilities.bullet import load_urdf, draw_text
from importlib.resources import files


class RobotiqGripper(Gripper):
    """Base gripper class.

    Args:
      robot: int representing PyBullet ID of robot.
      ee: int representing PyBullet ID of end effector link.
      obj_ids: list of PyBullet IDs of all suctionable objects in the env.
    """

    def __init__(self, robot, ee, obj_ids):
        super().__init__()

        # TODO: investigate
        # Load suction gripper base model (visual only).
        pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))

        # corallab_sim.robots "assets/grippers/robotiq/robotiq_85.urdf
        robotiq_gripper_urdf = files("corallab_sim.robots").joinpath("assets/ur5/gripper/robotiq_2f_85.urdf")
        self.id = load_urdf(p, str(robotiq_gripper_urdf), pose[0], pose[1])
        # TODO: investigate useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        c = p.createConstraint(
            parentBodyUniqueId=robot.id,
            parentLinkIndex=ee,
            childBodyUniqueId=self.id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0.01))
        p.changeConstraint(c, maxForce=10000)

        self.gripper_range = [0, 0.085]

        n_joints = p.getNumJoints(self.id)
        self.joints = [p.getJointInfo(self.id, i) for i in range(n_joints)]
        self.__create_mimic_joints__()
        self.move_timestep = robot.move_timestep

        # self.activated = False
        # self.arm_rest_poses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
        #                        -1.5707970583733368, 0.0009377758247187636]

    def __create_mimic_joints__(self):
        # These are defined within the robotiq_gripper urdf. See
        # http://wiki.ros.org/urdf/XML/joint for more info.
        joint_to_mimic_name = b"robotiq_2f_85_right_driver_joint"
        # negated from urdf because relationship is inverted (to_mimic ->
        # mimicking_joint)
        mimic_joints_and_multipliers = {
            b"robotiq_2f_85_right_follower_joint":     1,
            b"robotiq_2f_85_right_spring_link_joint": -1,
            b"robotiq_2f_85_left_driver_joint":       -1,
            b"robotiq_2f_85_left_follower_joint":      1,
            b"robotiq_2f_85_left_spring_link_joint":  -1,
        }

        self.joint_to_mimic_id = [j[0] for j in self.joints if j[1] == joint_to_mimic_name][0]

        for j_name, mult in mimic_joints_and_multipliers.items():
            self.__create_mimic_constraint__(joint_to_mimic_name, j_name, mult)

    def __create_mimic_constraint__(
            self,
            joint_to_mimic_name,
            joint_name,
            multiplier
    ):
        joint_id = [j[0] for j in self.joints if j[1] == joint_name][0]

        c = p.createConstraint(self.id, joint_id,
                               self.id, self.joint_to_mimic_id,
                               jointType=p.JOINT_GEAR,
                               jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=multiplier, maxForce=10000, erp=1)

    def move_gripper(self, open_length):
        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        # Control the mimic gripper joint(s)
        joint_to_mimic_info = self.joints[self.joint_to_mimic_id]
        joint_max_force = joint_to_mimic_info[10]
        joint_max_velocity = joint_to_mimic_info[11]
        p.setJointMotorControl2(self.id, self.joint_to_mimic_id, p.POSITION_CONTROL, targetPosition=open_angle,
                                force=joint_max_force,
                                maxVelocity=joint_max_velocity)

    def activate(self):
        """Simulate suction using a rigid fixed constraint to contacted object."""
        pass
        # TODO(andyzeng): check deformables logic.
        # del def_ids

        # if not self.activated:
        #     points = p.getContactPoints(bodyA=self.id, linkIndexA=0)
        #     # print(points)
        #     if points:

        #         # Handle contact between suction with a rigid object.
        #         for point in points:
        #             obj_id, contact_link = point[2], point[4]
        #         # if obj_id in self.obj_ids['rigid']:
        #         body_pose = p.getLinkState(self.id, 0)
        #         obj_pose = p.getBasePositionAndOrientation(obj_id)
        #         world_to_body = p.invertTransform(body_pose[0], body_pose[1])
        #         obj_to_body = p.multiplyTransforms(world_to_body[0],
        #                                            world_to_body[1],
        #                                            obj_pose[0], obj_pose[1])
        #         self.contact_constraint = p.createConstraint(
        #             parentBodyUniqueId=self.id,
        #             parentLinkIndex=0,
        #             childBodyUniqueId=obj_id,
        #             childLinkIndex=contact_link,
        #             jointType=p.JOINT_FIXED,
        #             jointAxis=(0, 0, 0),
        #             parentFramePosition=obj_to_body[0],
        #             parentFrameOrientation=obj_to_body[1],
        #             childFramePosition=(0, 0, 0),
        #             childFrameOrientation=(0, 0, 0))

        #         self.activated = True

    def get_contact_points(self):
        """Detects a contact with a rigid object."""
        left_pad_link_id = [j[0] for j in self.joints if j[1] == b'robotiq_2f_85_left_pad_joint'][0]
        right_pad_link_id = [j[0] for j in self.joints if j[1] == b'robotiq_2f_85_right_pad_joint'][0]

        left_contact_points = p.getContactPoints(bodyA=self.id, linkIndexA=left_pad_link_id)
        right_contact_points = p.getContactPoints(bodyA=self.id, linkIndexA=right_pad_link_id)
        points = [p for p in left_contact_points
                  if p[2] != self.id] + \
                 [p for p in right_contact_points
                  if p[2] != self.id]

        return points

    def detect_contact(self):
        contact_points = self.get_contact_points()
        return len(contact_points) > 0

    def check_grasp(self):
        return True

    def get_q(self):
        """q is expressed in length"""
        open_angle = p.getJointState(self.id, self.joint_to_mimic_id)[0]
        open_length = (math.sin(0.715 - open_angle) * 0.1143) + 0.010
        return open_length

    @classmethod
    def _norm(cls, it: Iterable) -> float:
        return np.linalg.norm(it)

    @classmethod
    def _unit_vec(cls, lst: np.ndarray) -> np.ndarray:
        mag = cls._norm(lst)
        return (lst / mag) if mag > 0 else 0

    def pose(self):
        return p.getLinkState(self.id, 0)[:2]

    def move_q(self, tar_q, error_thresh=1e-4, speed=0.01, break_cond=lambda: False, max_iter=600, **kwargs):
        """ Written with help of TransporterNet code: https://arxiv.org/pdf/2010.14406.pdf"""
        i = 0
        assert i < max_iter
        while i < max_iter:
            cur_q = self.get_q()
            err_q = tar_q - cur_q

            draw_text(self.pose()[0], "max force: {}", self.grasp_force(), lifeTime=0.1)

            if break_cond() or (np.abs(err_q) < error_thresh).all():
                # p.removeBody(marker)
                return True, tar_q, cur_q

            u = self._unit_vec(err_q)
            step_q = cur_q + u * speed
            self.move_gripper(step_q)

            p.stepSimulation()
            i += 1
            time.sleep(self.move_timestep)

        # p.removeBody(marker)
        return False, tar_q, cur_q

    def grasp_force(self):
        contact_points = self.get_contact_points()
        normalForcePos = 9

        return max([p[normalForcePos] for p in contact_points], default=0)

    def grasp(self, on, colmask):
        # collision masks for simplifying testing
        # obj = self.check_grasp()

        def bc():
            return self.grasp_force() > 15

        if on:
            closed_position = self.gripper_range[0]
            self.move_q(closed_position, break_cond=bc)
        else:
            open_position = self.gripper_range[1]
            self.move_q(open_position)

    def release(self):
        return


class RealRobotiqGripper:
    """
    Communicates with the gripper directly, via socket with string commands, leveraging string names for variables.
    """
    # WRITE VARIABLES (CAN ALSO READ)
    ACT = 'ACT'  # act : activate (1 while activated, can be reset to clear fault status)
    GTO = 'GTO'  # gto : go to (will perform go to with the actions set in pos, for, spe)
    ATR = 'ATR'  # atr : auto-release (emergency slow move)
    ADR = 'ADR'  # adr : auto-release direction (open(1) or close(0) during auto-release)
    FOR = 'FOR'  # for : force (0-255)
    SPE = 'SPE'  # spe : speed (0-255)
    POS = 'POS'  # pos : position (0-255), 0 = open
    # READ VARIABLES
    STA = 'STA'  # status (0 = is reset, 1 = activating, 3 = active)
    PRE = 'PRE'  # position request (echo of last commanded position)
    OBJ = 'OBJ'  # object detection (0 = moving, 1 = outer grip, 2 = inner grip, 3 = no object at rest)
    FLT = 'FLT'  # fault (0=ok, see manual for errors if not zero)

    ENCODING = 'UTF-8'  # ASCII and UTF-8 both seem to work

    class GripperStatus(Enum):
        """Gripper status reported by the gripper. The integer values have to match what the gripper sends."""
        RESET = 0
        ACTIVATING = 1
        # UNUSED = 2  # This value is currently not used by the gripper firmware
        ACTIVE = 3

    class ObjectStatus(Enum):
        """Object status reported by the gripper. The integer values have to match what the gripper sends."""
        MOVING = 0
        STOPPED_OUTER_OBJECT = 1
        STOPPED_INNER_OBJECT = 2
        AT_DEST = 3

    def __init__(self):
        """Constructor."""
        self.socket = None
        self.command_lock = threading.Lock()
        self._min_position = 0
        self._max_position = 255
        self._min_speed = 0
        self._max_speed = 255
        self._min_force = 0
        self._max_force = 255

    def connect(self, hostname: str, port: int, socket_timeout: float = 2.0) -> None:
        """Connects to a gripper at the given address.
        :param hostname: Hostname or ip.
        :param port: Port.
        :param socket_timeout: Timeout for blocking socket operations.
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((hostname, port))
        self.socket.settimeout(socket_timeout)

    def disconnect(self) -> None:
        """Closes the connection with the gripper."""
        self.socket.close()

    def _set_vars(self, var_dict: OrderedDict[str, Union[int, float]]):
        """Sends the appropriate command via socket to set the value of n variables, and waits for its 'ack' response.
        :param var_dict: Dictionary of variables to set (variable_name, value).
        :return: True on successful reception of ack, false if no ack was received, indicating the set may not
        have been effective.
        """
        # construct unique command
        cmd = "SET"
        for variable, value in var_dict.items():
            cmd += f" {variable} {str(value)}"
        cmd += '\n'  # new line is required for the command to finish
        # atomic commands send/rcv
        with self.command_lock:
            self.socket.sendall(cmd.encode(self.ENCODING))
            data = self.socket.recv(1024)
        return self._is_ack(data)

    def _set_var(self, variable: str, value: Union[int, float]):
        """Sends the appropriate command via socket to set the value of a variable, and waits for its 'ack' response.
        :param variable: Variable to set.
        :param value: Value to set for the variable.
        :return: True on successful reception of ack, false if no ack was received, indicating the set may not
        have been effective.
        """
        return self._set_vars(OrderedDict([(variable, value)]))

    def _get_var(self, variable: str):
        """Sends the appropriate command to retrieve the value of a variable from the gripper, blocking until the
        response is received or the socket times out.
        :param variable: Name of the variable to retrieve.
        :return: Value of the variable as integer.
        """
        # atomic commands send/rcv
        with self.command_lock:
            cmd = f"GET {variable}\n"
            # print(cmd)
            self.socket.sendall(cmd.encode(self.ENCODING))
            data = self.socket.recv(1024)

        # expect data of the form 'VAR x', where VAR is an echo of the variable name, and X the value
        # note some special variables (like FLT) may send 2 bytes, instead of an integer. We assume integer here
        var_name, value_str = data.decode(self.ENCODING).split()
        if var_name != variable:
            raise ValueError(f"Unexpected response {data} ({data.decode(self.ENCODING)}): does not match '{variable}'")
        # print(value_str)
        value = int(value_str)
        return value

    @staticmethod
    def _is_ack(data: str):
        return data == b'ack'

    def _reset(self):
        """
        Reset the gripper.
        The following code is executed in the corresponding script function
        def rq_reset(gripper_socket="1"):
            rq_set_var("ACT", 0, gripper_socket)
            rq_set_var("ATR", 0, gripper_socket)

            while(not rq_get_var("ACT", 1, gripper_socket) == 0 or not rq_get_var("STA", 1, gripper_socket) == 0):
                rq_set_var("ACT", 0, gripper_socket)
                rq_set_var("ATR", 0, gripper_socket)
                sync()
            end

            sleep(0.5)
        end
        """
        self._set_var(self.ACT, 0)
        self._set_var(self.ATR, 0)
        while (not self._get_var(self.ACT) == 0 or not self._get_var(self.STA) == 0):
            self._set_var(self.ACT, 0)
            self._set_var(self.ATR, 0)
        time.sleep(0.5)

    def activate(self, auto_calibrate: bool = True):
        """Resets the activation flag in the gripper, and sets it back to one, clearing previous fault flags.
        :param auto_calibrate: Whether to calibrate the minimum and maximum positions based on actual motion.
        The following code is executed in the corresponding script function
        def rq_activate(gripper_socket="1"):
            if (not rq_is_gripper_activated(gripper_socket)):
                rq_reset(gripper_socket)

                while(not rq_get_var("ACT", 1, gripper_socket) == 0 or not rq_get_var("STA", 1, gripper_socket) == 0):
                    rq_reset(gripper_socket)
                    sync()
                end

                rq_set_var("ACT",1, gripper_socket)
            end
        end
        def rq_activate_and_wait(gripper_socket="1"):
            if (not rq_is_gripper_activated(gripper_socket)):
                rq_activate(gripper_socket)
                sleep(1.0)

                while(not rq_get_var("ACT", 1, gripper_socket) == 1 or not rq_get_var("STA", 1, gripper_socket) == 3):
                    sleep(0.1)
                end

                sleep(0.5)
            end
        end
        """
        if not self.is_active():
            self._reset()
            while (not self._get_var(self.ACT) == 0 or not self._get_var(self.STA) == 0):
                time.sleep(0.01)

            self._set_var(self.ACT, 1)
            time.sleep(1.0)
            while (not self._get_var(self.ACT) == 1 or not self._get_var(self.STA) == 3):
                time.sleep(0.01)

        # auto-calibrate position range if desired
        if auto_calibrate:
            self.auto_calibrate()

    def is_active(self):
        """Returns whether the gripper is active."""
        status = self._get_var(self.STA)
        return RobotiqGripper.GripperStatus(status) == RobotiqGripper.GripperStatus.ACTIVE

    def get_min_position(self) -> int:
        """Returns the minimum position the gripper can reach (open position)."""
        return self._min_position

    def get_max_position(self) -> int:
        """Returns the maximum position the gripper can reach (closed position)."""
        return self._max_position

    def get_open_position(self) -> int:
        """Returns what is considered the open position for gripper (minimum position value)."""
        return self.get_min_position()

    def get_closed_position(self) -> int:
        """Returns what is considered the closed position for gripper (maximum position value)."""
        return self.get_max_position()

    def is_open(self):
        """Returns whether the current position is considered as being fully open."""
        return self.get_current_position() <= self.get_open_position()

    def is_closed(self):
        """Returns whether the current position is considered as being fully closed."""
        return self.get_current_position() >= self.get_closed_position()

    def get_current_position(self) -> int:
        """Returns the current position as returned by the physical hardware."""
        return self._get_var(self.POS)

    def auto_calibrate(self, log: bool = True) -> None:
        """Attempts to calibrate the open and closed positions, by slowly closing and opening the gripper.
        :param log: Whether to print the results to log.
        """
        # first try to open in case we are holding an object
        (position, status) = self.move_and_wait_for_pos(self.get_open_position(), 64, 1)
        if RobotiqGripper.ObjectStatus(status) != RobotiqGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed opening to start: {str(status)}")

        # try to close as far as possible, and record the number
        (position, status) = self.move_and_wait_for_pos(self.get_closed_position(), 64, 1)
        if RobotiqGripper.ObjectStatus(status) != RobotiqGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed because of an object: {str(status)}")
        assert position <= self._max_position
        self._max_position = position

        # try to open as far as possible, and record the number
        (position, status) = self.move_and_wait_for_pos(self.get_open_position(), 64, 1)
        if RobotiqGripper.ObjectStatus(status) != RobotiqGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed because of an object: {str(status)}")
        assert position >= self._min_position
        self._min_position = position

        if log:
            print(f"Gripper auto-calibrated to [{self.get_min_position()}, {self.get_max_position()}]")

    def move(self, position: int, speed: int, force: int) -> Tuple[bool, int]:
        """Sends commands to start moving towards the given position, with the specified speed and force.
        :param position: Position to move to [min_position, max_position]
        :param speed: Speed to move at [min_speed, max_speed]
        :param force: Force to use [min_force, max_force]
        :return: A tuple with a bool indicating whether the action it was successfully sent, and an integer with
        the actual position that was requested, after being adjusted to the min/max calibrated range.
        """

        def clip_val(min_val, val, max_val):
            return max(min_val, min(val, max_val))

        clip_pos = clip_val(self._min_position, position, self._max_position)
        clip_spe = clip_val(self._min_speed, speed, self._max_speed)
        clip_for = clip_val(self._min_force, force, self._max_force)

        # moves to the given position with the given speed and force
        var_dict = OrderedDict([(self.POS, clip_pos), (self.SPE, clip_spe), (self.FOR, clip_for), (self.GTO, 1)])
        return self._set_vars(var_dict), clip_pos

    def move_and_wait_for_pos(self, position: int, speed: int, force: int) -> Tuple[int, ObjectStatus]:  # noqa
        """Sends commands to start moving towards the given position, with the specified speed and force, and
        then waits for the move to complete.
        :param position: Position to move to [min_position, max_position]
        :param speed: Speed to move at [min_speed, max_speed]
        :param force: Force to use [min_force, max_force]
        :return: A tuple with an integer representing the last position returned by the gripper after it notified
        that the move had completed, a status indicating how the move ended (see ObjectStatus enum for details). Note
        that it is possible that the position was not reached, if an object was detected during motion.
        """
        set_ok, cmd_pos = self.move(position, speed, force)
        if not set_ok:
            raise RuntimeError("Failed to set variables for move.")

        # wait until the gripper acknowledges that it will try to go to the requested position
        while self._get_var(self.PRE) != cmd_pos:
            time.sleep(0.001)

        # wait until not moving
        cur_obj = self._get_var(self.OBJ)
        while RobotiqGripper.ObjectStatus(cur_obj) == RobotiqGripper.ObjectStatus.MOVING:
            cur_obj = self._get_var(self.OBJ)

        # report the actual position and the object status
        final_pos = self._get_var(self.POS)
        final_obj = cur_obj
        return final_pos, RobotiqGripper.ObjectStatus(final_obj)

    def toggle(self):
        min_p = self.get_min_position()
        max_p = self.get_max_position()

        if self.get_current_position() > min_p:
            self.move_and_wait_for_pos(min_p, 255, 255)
            self.state = 0
        else:
            self.move_and_wait_for_pos(max_p, 255, 255)
            self.state = 1

        return self.state

    def get_state(self):
        return self.state
