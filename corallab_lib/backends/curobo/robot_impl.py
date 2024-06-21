# Third Party
import torch

# cuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml


curobo_config_map = {
    "Panda": "franka.yml",
    "UR5": "ur5e_robotiq_2f_140.yml",
}

class CuroboRobot:
    def __init__(
            self,
            id: str,
            **kwargs
    ):

        tensor_args = TensorDeviceType()

        config_file = load_yaml(join_path(get_robot_path(), "franka.yml"))["robot_cfg"]
        self.config = RobotConfig.from_dict(config_file, tensor_args)

        self.kin_model = CudaRobotModel(self.config.kinematics)

    def get_n_dof(self):
        return self.kin_model.kinematics_config.n_dof

    def get_q_min(self):
        return self.kin_model.kinematics_config.joint_limits.position[0]

    def get_q_max(self):
        return self.kin_model.kinematics_config.joint_limits.position[1]

    # def random_q(self, n_samples=10):
    #     return self.robot_impl.random_q(n_samples=n_samples)

    # def tmp(self):
    #     # compute forward kinematics:
    #     # torch random sampling might give values out of joint limits
    #     q = torch.rand((10, kin_model.get_dof()), **vars(tensor_args))
    #     out = self.kin_model.get_state(q)
