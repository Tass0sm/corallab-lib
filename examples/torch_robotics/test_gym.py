import torch
import corallab_lib
from corallab_lib import Gym, Robot, Env, Task

corallab_lib.backend_manager.set_backend("torch_robotics")

robot = Robot("RobotPointMass")
env = Env("EnvSquare2D")
task = Task("PlanningTask", robot=robot, env=env)
gym = Gym(task, render_mode="human")

obs, info = gym.reset(seed=1)

action = torch.tensor(gym.action_space.sample())
# gym.step(action)
