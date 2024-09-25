import corallab_lib
from corallab_lib import Robot, Env, DynamicPlanningProblem

corallab_lib.backend_manager.set_backend("corallab")

robot = Robot(
    "RobotPointMass",
    backend="torch_robotics"
)

env = Env(
    "EnvSquare2D",
    backend="torch_robotics"
)

problem = DynamicPlanningProblem(robot=robot, env=env)
