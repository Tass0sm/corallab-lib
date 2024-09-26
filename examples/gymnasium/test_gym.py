from corallab_lib import Gym

env_name = "Ant-v4"
gym = Gym(env_name, backend="gymnasium")

if __name__ == "__main__":

    # initialize the task
    gym = Gym("Ant-v4", backend="gymnasium")
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec

    # do visualization
    for i in range(10000):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()

