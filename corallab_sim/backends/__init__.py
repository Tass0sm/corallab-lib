from . import curobo
from . import mujoco
from . import pybullet
from . import torch_robotics

backends_dict = {
    "curobo": curobo,
    "mujoco": mujoco,
    "pybullet": pybullet,
    "torch_robotics": torch_robotics
}

backends_list = [k for k, v in backends_dict.items()]
