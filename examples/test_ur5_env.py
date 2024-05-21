"""Credit to: https://github.com/lyfkyle/pybullet_ompl/"""

import time
import os.path as osp
import pybullet as p
import math
import sys

import corallab_sim
from corallab_sim import Gym

corallab_sim.backend_manager.set_backend("pybullet")


if __name__ == '__main__':
    # 0. create env object
    env = Gym("UR5BulletEnv", render_mode="human")

    # 1. seed env for initial state.
    # Seed 0-200 are used for the demonstration dataset.
    env.seed(1000)

    # 2. must reset before use
    obs, IGNORE_GIT_FOLDER_PATTERNS = env.reset()

    for _ in range(1000):
        # 3. 2D positional action space [0,512]
        action = env.action_space.sample()

        # 4. Standard gym step method
        obs, reward, terminated, truncated, info = env.step(action)

        time.sleep(0.033)
