import corallab_lib
from corallab_lib import Robot, Env, Task

corallab_lib.backend_manager.set_backend("curobo")

robot = Robot("UR5")
env = Env(None)
task = Task(None, robot=robot, env=env)
