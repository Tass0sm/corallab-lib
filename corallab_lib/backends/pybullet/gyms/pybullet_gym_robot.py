# import numpy as np

# import gymnasium as gym
# from gymnasium import spaces
# from gymnasium import utils

# from ..robot_impl import PybulletRobot


# class PybulletGymRobot(PybulletRobot):
#     def __init__(self, id, action_dim=None, obs_dim=None, **kwargs):
#         super().__init__(id, **kwargs)
#         self.done_loading = 0

#         action_dim = 6
#         action_lb = -100 * np.ones([action_dim], dtype=np.float32)
#         action_up = 100 * np.ones([action_dim], dtype=np.float32)
#         self.action_space = spaces.Box(-action_lb, action_up, dtype=np.float32)

#         obs_dim = 6
#         obs_lb = -10 * np.ones([obs_dim], dtype=np.float32)
#         obs_up = 10 * np.ones([obs_dim], dtype=np.float32)
#         self.observation_space = spaces.Box(-obs_lb, obs_up, dtype=np.float32)

#     def reset(self, bullet_client):
#         self._p = bullet_client

#         if self.done_loading == 0:
#             self.robot_impl.load(self._p)
#             self.done_loading = 1

#         self.robot_impl.set_q(self.robot_impl.arm_rest_poses)
#         s = self.calc_state()
#         return s

#     def apply_action(self, a, action_type="torque"):
#         assert np.isfinite(a).all()

#         if action_type == "torque":
#             self.robot_impl.apply_torque(a)
#         elif action_type == "velocity":
#             self.robot_impl.velocity_control(a)
#         elif action_type == "position":
#             self.robot_impl.position_control(a)

#     def calc_state(self):
#         s = self.robot_impl.get_joint_obs()
#         return np.concatenate([
#             s["positions"],
#             s["velocities"],
#         ])

#         # s["ee_pos"]

#     def calc_potential(self):
#         pass
