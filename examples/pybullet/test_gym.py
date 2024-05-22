import corallab_lib
from corallab_lib import Gym

corallab_lib.backend_manager.set_backend("pybullet")

gym = Gym("InvertedPendulumBulletEnv", render_mode="human")
