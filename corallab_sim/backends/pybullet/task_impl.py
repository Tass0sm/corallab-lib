from .env_impl import PybulletEnv
from .robot_impl import PybulletRobot


class PybulletTask:
    def __init__(
            self,
            env=None,
            robot=None,
            **kwargs
    ):
        assert isinstance(env, PybulletEnv)
        assert isinstance(robot, PybulletRobot)

        self.env = env
        self.robot = robot
        self.robot.robot_impl.load()

    def get_q_dim(self):
        # return self.task_impl.robot.q_dim
        pass

    def get_q_min(self):
        # return self.task_impl.robot.q_min.cpu()
        pass

    def get_q_max(self):
        # return self.task_impl.robot.q_max.cpu()
        pass

    def random_coll_free_q(self, *args, **kwargs):
        # return self.task_impl.random_coll_free_q(*args, **kwargs)
        pass

    def compute_collision(self, q):
        # return self.task_impl.compute_collision(q)
        pass
