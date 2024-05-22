import corallab_lib
from corallab_lib import Robot, Env, Task

corallab_lib.backend_manager.set_backend("pybullet")

robot = Robot("UR5")
env = Env(add_plane=False)
task = Task(robot=robot, env=env)
