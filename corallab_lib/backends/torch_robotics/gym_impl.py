import torch

from torch_robotics import environments
from torch_robotics.robots import RobotPointMass, RobotPointMass3D
from torch_robotics.torch_utils.torch_utils import to_numpy, DEFAULT_TENSOR_ARGS
from torch_robotics.torch_planning_objectives.fields.distance_fields import (
    CollisionWorkspaceBoundariesDistanceField,
    CollisionSelfField,
    CollisionObjectDistanceField
)

import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import pygame

from corallab_lib.utilities.blit_manager import BlitManager
from corallab_lib.backends.gym_interface import GymInterface


class TorchRoboticsGym(GymInterface, gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, task, render_mode=None):
        self.task = task

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        q_min = self.task.get_q_min().numpy()
        q_max = self.task.get_q_max().numpy()
        q_dim = self.task.get_q_dim()
        self.observation_space = spaces.Box(q_min, q_max, shape=(q_dim,))
        self.action_space = spaces.Box(q_min, q_max, shape=(q_dim,))

        # State
        self._threshold_start_goal_pos = 0
        self._start_state = None
        self._goal_state = None
        self._current_state = None

        # Rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.ax = None
        self.agent_artist = None
        self.fig_manager = None
        self.clock = None

    def _sample_state_and_goal_states(self):
        # Random initial and final positions
        n_tries = 100
        start_state_pos, goal_state_pos = None, None
        for _ in range(n_tries):
            q_free, _ = self.task.random_coll_free_q(n_samples=2)
            start_state_pos = q_free[0]
            goal_state_pos = q_free[1]

            if torch.linalg.norm(start_state_pos - goal_state_pos) > self._threshold_start_goal_pos:
                break

        if start_state_pos is None or goal_state_pos is None:
            raise ValueError(f"No collision free configuration was found\n"
                             f"start_state_pos: {start_state_pos}\n"
                             f"goal_state_pos:  {goal_state_pos}\n")

        return start_state_pos, goal_state_pos

    def _get_obs(self):
        return {"agent": self._current_state, "goal": self._goal_state}

    def _get_info(self):
        return {
            "distance": torch.norm(self._current_state - self._goal_state)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            torch.manual_seed(seed)

        self._start_state, self._goal_state = self._sample_state_and_goal_states()
        self._current_state = self._start_state

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _forward_dynamics(state, action):
        if not self.task.compute_collision(action):
            return action

    def step(self, action, **kwargs):
        # assert action.cpu().numpy() in self.action_space

        self._current_state = self._forward_dynamics(self._current_state, action)

        terminated = torch.allclose(self._current_state, self._goal_state)
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.fig_manager is None and self.render_mode == "human":
            fig = plt.figure()
            self.ax = fig.add_axes((0, 0, 1, 1))
            self.task.task_impl.task_impl.env.render(ax=self.ax)

            self.agent_artist = self.task.task_impl.task_impl.robot.render(self.ax, q=self._current_state)
            self.fig_manager = BlitManager(fig.canvas, [self.agent_artist])

            # make sure our window is on the screen and drawn
            plt.show(block=False)
            plt.pause(.1)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self.agent_artist.set_center(self._current_state)

        if self.render_mode == "human":
            self.fig_manager.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:
            return None
            # np.transpose(
            #     np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            # )
