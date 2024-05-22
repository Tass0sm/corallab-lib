import corallab_lib
from corallab_lib import Robot

corallab_lib.backend_manager.set_backend("multi_backend", backends=["pybullet", "torch_robotics"])

robot = Robot(
    pybullet=(["UR5"], {}),
    torch_robotics=(["RobotUR5"], {})
)

env = Env(
    pybullet=([], { "add_plane": False }),
    torch_robotics=(["EnvOpen3D"], {})
)

task = Task(
    torch_robotics=(["PlanningTask"],
                    { "robot": robot,
                      "env": env }),
    pybullet=([],
              { "robot": robot,
                "env": env })
)
