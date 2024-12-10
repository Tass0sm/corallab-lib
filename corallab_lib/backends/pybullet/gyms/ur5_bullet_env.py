# import time
# import pybullet
# import pybullet_data
# import numpy as np

# from pybullet_envs_gymnasium.env_bases import MJCFBaseBulletEnv
# from pybullet_envs_gymnasium.scene_abstract import SingleRobotEmptyScene
# from .pybullet_gym_robot import PybulletGymRobot


# class UR5BulletEnv(MJCFBaseBulletEnv):
#     def __init__(self, should_render=True, render_mode=None):
#         self.robot = PybulletGymRobot("UR5")
#         MJCFBaseBulletEnv.__init__(self, self.robot, render_mode)
#         self.should_render = render_mode in ["human", "rgb_array"]

#     def create_single_player_scene(self, bullet_client):
#         return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0165, frame_skip=1)

#     def reset(self, seed=None, options={}):
#         s, info = super().reset(seed=seed, options=options)

#         self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
#         self._p.loadURDF("plane.urdf", basePosition=(0, 0, 0))

#         # TODO: Set Fixed or Random Start
#         if "start_position" in options:
#             self.robot.set_q(options["start_position"])

#         # TODO: Set Fixed or Random Goal
#         if "goal_position" in options:
#             self.goal_position = options["goal_position"]

#         return s, info

#     def step(self, a, action_type="torque"):
#         assert not self.scene.multiplayer
#         self.robot.apply_action(a, action_type=action_type)
#         self.scene.global_step()
#         state = self.robot.calc_state()

#         self.rewards = [0]
#         self.HUD(state, a, False)

#         return state.astype(np.float32), sum(self.rewards), False, False, {}

#     def camera_adjust(self):
#         # Target:
#         target = (0, 0, 0.1)
#         dist = 2.0
#         yaw = 142.0
#         pitch = -20
#         self._p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

#         # move_and_look_at(self, 0.15, 0, 0.15, 0, 0, 0)

#     def render(self):
#         if self.render_mode == "human":
#             self.should_render = True
#         if self.physicsClientId >= 0:
#             self.camera_adjust()

#         if self.render_mode != "rgb_array":
#             time.sleep(1 / self.metadata["render_fps"])
#             return

#         base_pos = [0, 0, 0.5]
#         if hasattr(self, "robot"):
#             if hasattr(self.robot, "body_real_xyz"):
#                 base_pos = self.robot.body_real_xyz

#         if self.physicsClientId >= 0:
#             view_matrix = self._p.computeViewMatrix(
#                 cameraEyePosition=(1.4, 1.4, 0.5),
#                 cameraTargetPosition=base_pos,
#                 cameraUpVector=(0, 0, 1)
#             )
#             proj_matrix = self._p.computeProjectionMatrixFOV(
#                 fov=60, aspect=1.0, nearVal=1, farVal=10
#             )
#             (_, _, px, _, _) = self._p.getCameraImage(
#                 width=96 * 2,
#                 height=96 * 2,
#                 viewMatrix=view_matrix,
#                 projectionMatrix=proj_matrix,
#                 renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
#             )

#             self._p.configureDebugVisualizer(self._p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
#         else:
#             px = np.array([[[255, 255, 255, 255]] * self._render_width] * self._render_height, dtype=np.uint8)
#         rgb_array = np.array(px, dtype=np.uint8)
#         rgb_array = np.reshape(np.array(px), (96 * 2, 96 * 2, -1))
#         rgb_array = rgb_array[:, :, :3]
#         return rgb_array
