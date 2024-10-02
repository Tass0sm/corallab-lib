import numpy as np

from corallab_lib import Gym


if __name__ == "__main__":

    # initialize the task
    gym = Gym("Ant-v4", render_mode="human", backend="gymnasium")
    gym.reset()
    # gym.gym_impl.gym_impl.viewer.set_camera(camera_id=0)

    # TODO: Decide on interface
    env = gym.gym_impl

    # Get action limits
    action_space = env.action_space
    low, high = action_space.low, action_space.high

    # do visualization
    for i in range(10000):
        action = np.random.uniform(low, high)
        obs, reward, done, _, _ = env.step(action)
        env.render()

