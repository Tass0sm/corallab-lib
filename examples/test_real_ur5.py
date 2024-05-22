import time
from corallab_lib.in_reality.robots.ur5 import UR5
from corallab_lib.in_reality.robots.robotiq import RobotiqGripper


ip_address = "192.168.1.123"
real_robot = UR5(ip_address)
real_gripper = RobotiqGripper()
real_gripper.connect(ip_address, 63352)
real_gripper.activate()

real_robot.gripper = real_gripper

for _ in range(100):
    real_robot.test()
    time.sleep(10)
