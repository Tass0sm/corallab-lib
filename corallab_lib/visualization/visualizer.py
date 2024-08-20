import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from torch_robotics.torch_utils.torch_utils import to_numpy
import matplotlib.collections as mcoll


class Visualizer:

    def __init__(
            self,
            problem=None,
            planner=None
    ):
        self.problem = problem
        self.env = self.problem.env
        self.robot = self.problem.robot
        self.planner = planner

        self.colors = {'collision': 'black', 'free': 'orange'}
        self.colors_robot = {'collision': 'black', 'free': 'darkorange'}
        self.cmaps = {'collision': 'Greys', 'free': 'Oranges'}
        self.cmaps_robot = {'collision': 'Greys', 'free': 'YlOrRd'}


    def plot_joint_space_state_trajectories(
            self,
            fig=None, axs=None,
            trajs=None,
            traj_best=None,
            pos_start_state=None, pos_goal_state=None,
            vel_start_state=None, vel_goal_state=None,
            set_joint_limits=True,
            skip_selection=False,
            **kwargs
    ):
        if trajs is None:
            return
        trajs_np = to_numpy(trajs)

        assert trajs_np.ndim == 3
        B, H, D = trajs_np.shape

        # Separate trajectories in collision and free (not in collision)
        trajs_coll, trajs_free = self.problem.get_trajs_collision_and_free(trajs)

        trajs_coll_pos_np = to_numpy([])
        trajs_coll_vel_np = to_numpy([])
        if trajs_coll is not None:
            trajs_coll_pos_np = to_numpy(self.robot.get_position(trajs_coll))
            trajs_coll_vel_np = to_numpy(self.robot.get_velocity(trajs_coll))

        trajs_free_pos_np = to_numpy([])
        trajs_free_vel_np = to_numpy([])
        if trajs_free is not None:
            trajs_free_pos_np = to_numpy(self.robot.get_position(trajs_free))
            trajs_free_vel_np = to_numpy(self.robot.get_velocity(trajs_free))

        if pos_start_state is not None:
            pos_start_state = to_numpy(pos_start_state)
        if vel_start_state is not None:
            vel_start_state = to_numpy(vel_start_state)
        if pos_goal_state is not None:
            pos_goal_state = to_numpy(pos_goal_state)
        if vel_goal_state is not None:
            vel_goal_state = to_numpy(vel_goal_state)

        if fig is None or axs is None:
            dim = self.robot.get_n_dof()
            fig, axs = plt.subplots(dim, 2, squeeze=False)

        axs[0, 0].set_title('Position')
        axs[0, 1].set_title('Velocity')
        axs[-1, 0].set_xlabel('Timesteps')
        axs[-1, 1].set_xlabel('Timesteps')
        timesteps = np.arange(H).reshape(1, -1)
        for i, ax in enumerate(axs):
            for trajs_filtered, color in zip([(trajs_coll_pos_np, trajs_coll_vel_np), (trajs_free_pos_np, trajs_free_vel_np)],
                                             ['black', 'orange']):
                # Positions and velocities
                for j, trajs_filtered_ in enumerate(trajs_filtered):
                    # print(f"currently at {i}, {j}")
                    # print(f"trajs {trajs_filtered_.shape}")
                    if trajs_filtered_.size > 0:
                        timesteps_ = np.repeat(timesteps, trajs_filtered_.shape[0], axis=0)
                        plot_multiline(ax[j], timesteps_, trajs_filtered_[..., i], color=color, **kwargs)

            if traj_best is not None:
                traj_best_pos = self.robot.get_position(traj_best)
                traj_best_vel = self.robot.get_velocity(traj_best)
                traj_best_pos_np = to_numpy(traj_best_pos)
                traj_best_vel_np = to_numpy(traj_best_vel)
                plot_multiline(ax[0], timesteps, traj_best_pos_np[..., i].reshape(1, -1), color='blue', **kwargs)
                plot_multiline(ax[1], timesteps, traj_best_vel_np[..., i].reshape(1, -1), color='blue', **kwargs)

            # Start and goal
            if pos_start_state is not None:
                ax[0].scatter(0, pos_start_state[i], color='green')
            if vel_start_state is not None:
                ax[1].scatter(0, vel_start_state[i], color='green')
            if pos_goal_state is not None:
                ax[0].scatter(H-1, pos_goal_state[i], color='purple')
            if vel_goal_state is not None:
                ax[1].scatter(H-1, vel_goal_state[i], color='purple')
            # Y label
            ax[0].set_ylabel(f'q_{i}')
            # Set limits
            if set_joint_limits:
                q_min = self.robot.get_q_min()
                if isinstance(q_min, torch.Tensor):
                    q_min_np = q_min.cpu().numpy()
                else:
                    q_min_np = q_min

                q_max = self.robot.get_q_max()
                if isinstance(q_max, torch.Tensor):
                    q_max_np = q_max.cpu().numpy()
                else:
                    q_max_np = q_max

                ax[0].set_ylim(q_min_np[i], q_max_np[i])
                # ax[1].set_ylim(self.robot.q_vel_min_np[i], self.robot.q_vel_max_np[i])

        return fig, axs


    def animate_opt_iters_joint_space_state(
            self, trajs=None, traj_best=None, n_frames=10, **kwargs
    ):
        # trajs: steps, batch, horizon, q_dim
        if trajs is None:
            return

        assert trajs.ndim == 4

        S, B, H, D = trajs.shape

        idxs = np.round(np.linspace(0, S - 1, n_frames)).astype(int)
        trajs_selection = trajs[idxs]

        fig, axs = self.plot_joint_space_state_trajectories(
            trajs=trajs_selection[0],
            skip_selection=True,
            **kwargs
        )

        def animate_fn(i):
            [ax.clear() for ax in axs.ravel()]
            fig.suptitle(f"iter: {idxs[i]}/{S-1}")
            self.plot_joint_space_state_trajectories(
                fig=fig, axs=axs,
                trajs=trajs_selection[i],
                skip_selection=True,
                **kwargs
            )
            if i == n_frames -1 and traj_best is not None:
                self.plot_joint_space_state_trajectories(
                    fig=fig, axs=axs,
                    trajs=trajs_selection[i],
                    traj_best=traj_best,
                    skip_selection=True,
                    **kwargs
                )

        create_animation_video(fig, animate_fn, n_frames=n_frames, **kwargs)


def create_animation_video(fig, animate_fn, anim_time=5, n_frames=100, video_filepath='video.mp4', **kwargs):
    str_start = "Creating animation"
    print(f'{str_start}...')
    ani = FuncAnimation(
        fig,
        animate_fn,
        frames=n_frames,
        interval=anim_time * 1000 / n_frames,
        repeat=False
    )
    print(f'...finished {str_start}')

    str_start = "Saving video..."
    print(f'{str_start}...')
    ani.save(os.path.join(video_filepath), fps=max(1, int(n_frames / anim_time)), dpi=90)
    print(f'...finished {str_start}')


def plot_multiline(ax, X, Y, color='blue', linestyle='solid', **kwargs):
    segments = np.stack((X, Y), axis=-1)
    line_segments = mcoll.LineCollection(segments, colors=[color] * len(segments), linestyle=linestyle)
    ax.add_collection(line_segments)
    points = np.reshape(segments, (-1, 2))
    ax.scatter(points[:, 0], points[:, 1], color=color, s=2 ** 2)
