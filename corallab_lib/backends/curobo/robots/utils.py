import os

# cuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml

import importlib
import corallab_assets


def find_config_dict(config_file_basename):
    path = join_path(get_robot_path(), config_file_basename)

    if os.path.exists(path):
        return load_yaml(path)["robot_cfg"]

    path = corallab_assets.get_resource_path(config_file_basename)


    if os.path.exists(path):
        if path.suffix == ".yml" or path.suffix == ".yaml":
            return load_yaml(path)["robot_cfg"]
        elif path.suffix == ".py":
            loader = importlib.machinery.SourceFileLoader(path.stem, str(path))
            spec = importlib.util.spec_from_loader(path.stem, loader)
            robot_config_module = importlib.util.module_from_spec(spec)
            loader.exec_module(robot_config_module)

            return robot_config_module.robot_cfg
    else:
        raise FileNotFoundError
