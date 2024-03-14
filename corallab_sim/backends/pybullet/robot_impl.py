from . import robots


class PybulletRobot:
    def __init__(
            self,
            id: str
    ):
        RobotClass = getattr(robots, id)
        self.robot_impl = RobotClass()

    def random_q(self, n_samples=10):
        return self.robot_impl.random_q(n_samples=n_samples)
