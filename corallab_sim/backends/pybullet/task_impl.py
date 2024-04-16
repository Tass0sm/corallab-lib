import numpy as np
import corallab_sim.backends.pybullet.ompl.utils as pb_utils

from itertools import product

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


        self._setup_collision_detection(self.robot, self.env)

    def _setup_collision_detection(self, robot, env, allow_collision_links=[]):
        pybullet_robots = [self.robot.robot_impl]

        self.intra_robot_link_pairs = []
        for pybullet_bot in pybullet_robots:
            pairs = pb_utils.get_self_link_pairs(
                pybullet_bot.id,
                pybullet_bot.arm_controllable_joints
            )

            self.intra_robot_link_pairs.extend(pairs)

        moving_bodies = []
        all_moving_links = []
        self.inter_robot_link_pairs = []
        for pybullet_bot in pybullet_robots:
            bot_moving_links = pb_utils.get_moving_links(
                pybullet_bot.id,
                pybullet_bot.arm_controllable_joints
            )

            full_bot_moving_links = [pb_utils.Link(pybullet_bot.id, x) for x in bot_moving_links]
            self.inter_robot_link_pairs.extend(product(all_moving_links, full_bot_moving_links))
            all_moving_links.extend(full_bot_moving_links)

            moving_bodies.append((pybullet_bot.id, bot_moving_links))

        self.robot_object_body_pairs = list(product(moving_bodies, self.env.pb_objs))




    def get_q_dim(self):
        # return self.task_impl.robot.q_dim
        pass

    def get_q_min(self):
        # return self.task_impl.robot.q_min.cpu()
        pass

    def get_q_max(self):
        # return self.task_impl.robot.q_max.cpu()
        pass

    def random_coll_free_q(self, gen):
        n_samples=1
        max_tries=1000

        # Random position in configuration space not in collision
        reject = True
        samples = np.zeros((n_samples, self.robot.robot_impl.arm_num_dofs))
        idx_begin = 0
        for i in range(max_tries):
            qs = np.expand_dims(self.robot.random_q(gen), 0)

            in_collision = self.compute_collision(qs)
            idxs_not_in_collision = np.argwhere(in_collision == False)
            if idxs_not_in_collision.size == 0:
                # all points are in collision
                continue

            idxs_not_in_collision = idxs_not_in_collision.squeeze(axis=0)
            idx_random = gen.permutation(len(idxs_not_in_collision))[:n_samples]
            free_qs = qs[idxs_not_in_collision[idx_random]]
            idx_end = min(idx_begin + free_qs.shape[0], samples.shape[0])
            samples[idx_begin:idx_end] = free_qs[:idx_end - idx_begin]
            idx_begin = idx_end
            if idx_end >= n_samples:
                reject = False
                break

        if reject:
            sys.exit("Could not find a collision free configuration")

        return samples.squeeze()


    def compute_collision(self, qs, **kwargs):

        # Helper
        def set_state(robot, q, **kwargs):
            robot.set_q(q)

        # Helper
        def check_pb_collision(q, **kwargs):
            # Intra-robot collision detection
            for link1, link2 in self.intra_robot_link_pairs:
                if pb_utils.pairwise_link_collision(link1.body_id, link1.link_id,
                                                    link2.body_id, link2.link_id):
                    return True

            # Inter-robot collision detection
            for link1, link2 in self.inter_robot_link_pairs:
                if pb_utils.pairwise_link_collision(link1.body_id, link1.link_id,
                                                    link2.body_id, link2.link_id):
                    return True

            # Object collision detection
            for body1, body2 in self.robot_object_body_pairs:
                if pb_utils.pairwise_collision(body1, body2):
                    return True

            return False

        def for_qs(qs, per_robot_f, per_state_f, **kwargs):
            assert qs.ndim == 2
    
            results = []
    
            for q_i in qs:
                # q_offset = 0
                # for base_pose, robot in zip(self.base_poses, self.subrobots):
                #     subrobot_q = q_i[..., q_offset:q_offset+robot.q_dim]
                #     per_robot_f(robot, q=subrobot_q, **kwargs)
                #     q_offset += robot.q_dim
    
                per_robot_f(self.robot, q_i, **kwargs)
                result = per_state_f(q_i, **kwargs)
                results.append(result)
    
            return np.array(results)


        
        if qs.ndim == 1:
            qs = np.expand_dims(qs, 0)
            results = for_qs(qs, set_state, check_pb_collision, **kwargs)
        elif qs.ndim == 2:
            results = for_qs(qs, set_state, check_pb_collision, **kwargs)
        elif qs.ndim == 3:

            results_l = []
            for b in qs:
                r = self.robot.for_states(b, set_state, check_pb_collision, **kwargs)
                results_l.append(r)

            results = np.stack(results_l)

        return results
