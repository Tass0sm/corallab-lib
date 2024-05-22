"""Credit to: https://github.com/lyfkyle/pybullet_ompl/"""

import os.path as osp
import pybullet as p
import math
import sys

import corallab_lib
from corallab_lib import Gym

corallab_lib.backend_manager.set_backend("pybullet")


if __name__ == '__main__':
    # 0. create env object
    env = Gym("ReacherBulletEnv", render_mode="human")

    # 1. seed env for initial state.
    # Seed 0-200 are used for the demonstration dataset.
    env.seed(1000)

    # 2. must reset before use
    obs, IGNORE_GIT_FOLDER_PATTERNS = env.reset()

    # 3. 2D positional action space [0,512]
    action = env.action_space.sample()

    # 4. Standard gym step method
    obs, reward, terminated, truncated, info = env.step(action)
