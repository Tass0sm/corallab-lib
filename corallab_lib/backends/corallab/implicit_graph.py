import math
import torch
import numpy as np
import itertools
import random

try:
    from collections import Mapping
except ImportError:
    from collections.abc import Mapping


class ExplicitVertex:

    def __init__(self, implicit_nodes):
        self.implicit_nodes = implicit_nodes


class ImplicitGraph(Mapping):
    """Defines implicit graph (composite of PRM roadmaps)"""

    def __init__(
            self,
            task,
            prms,
    ):
        """Loads PRM roadmap that will define implicit graph."""
        self.task = task
        self.prms = prms
        self.roadmaps = [prm.planner_impl.planner_impl.roadmap for prm in self.prms]

    def __getitem__(self, q):
        subrobot_states = self._separate_subrobot_states(q)
        return [r_i[q_i] for r_i, q_i in zip(self.roadmaps, subrobot_states)]

    def __contains__(self, q, **kwargs):
        subrobot_states = self._separate_subrobot_states(q)
        in_roadmaps = all([q_i in r_i for r_i, q_i in zip(self.roadmaps, subrobot_states)])
        free = not self.task.check_collision(q, **kwargs).item()
        return in_roadmaps and free

    def __len__(self):
        return math.prod([len(r_i) for r_i in self.roadmaps])

    def __iter__(self):
        return itertools.product([iter(r_i) for r_i in self.roadmaps])

    def _separate_subrobot_states(self, joint_state):
        states = []

        for r in self.task.robot.get_subrobots():
            subrobot_state = r.get_position(joint_state)
            states.append(subrobot_state)
            joint_state = joint_state[r.get_n_dof():]

        return states

    def check_local_motion(self, q1, q2, step=None, no_max_dist=False, **kwargs):
        return self.task.check_local_motion(q1, q2, step=step, no_max_dist=no_max_dist, **kwargs)

    def get_n_dof(self):
        return self.task.get_q_dim()

    def _compute_composite_distance(self, config1, config2):
        """Computes distance in "composite configuration space".
        Defined as sum of Euclidean distances between PRM nodes in two configs.
        """
        dist = 0
        for i in range(len(config1)):
            dist += self.env.ComputeDistance(config1[i], config2[i])
        return dist

    def _nearest_node_in_graph(self, subrobot_states_l):
        """Returns nearest node in implicit graph to a composite configuration.
        Input: list of configurations
        Output: list of node IDs of closest node on implicit graph
        """
        nearest = []

        for prm, q in zip(self.prms, subrobot_states_l):
            nearest.append(prm.get_nearest_node(q))

        return nearest

    def get_neighbors(self, state):
        """Returns list of neighbors for node in implicit graph.
        """

        neighbors_of_each = []  # list of lists
        subrobot_states = self._separate_subrobot_states(state)

        for roadmap, state in zip(self.roadmaps, subrobot_states):
            node = roadmap[state]
            neighbor_and_self_states = [n.q for n in node.edges.keys()] # if n is not node]
            neighbors_of_each.append(neighbor_and_self_states)

        # Return all possible combinations of neighbors
        neighbors = list(itertools.product(*neighbors_of_each))
        neighbors = list(map(lambda p: torch.cat(p), neighbors))
        neighbors = torch.stack(neighbors)

        return neighbors

    def get_subrobot_neighbors(self, roadmap, a):
        """Returns closest neighbor for state a in a single graph to state b. It
        can also return state a itself.
        """

        node_a = roadmap[a]
        neighbor_states = [n.q for n in node_a.edges.keys() if n is not node_a]
        neighbor_states = torch.stack(neighbor_states)
        return neighbor_states

    def get_closest_subrobot_neighbor(self, roadmap, a, b):
        """Returns closest neighbor for state a in a single graph to state b. It
        can also return state a itself.
        """

        node_a = roadmap[a]
        neighbor_states = [n.q for n in node_a.edges.keys() if n is not node_a]

        if len(neighbor_states) == 0:
            return None

        neighbor_states = torch.stack(neighbor_states)

        b = b.expand(neighbor_states.shape)
        dist = neighbor_states.add(-b).square().sum(dim=-1).sqrt()
        min_dist_idx = torch.argmin(dist)

        return neighbor_states[min_dist_idx]

    def get_closest_composite_neighbor(self, a, b):
        """ Given randomly sampled comp config and
        nearest config on current tree, find neighbor of qnear that
        is closest to qrand
        """

        subrobot_states_a = self._separate_subrobot_states(a)
        subrobot_states_b = self._separate_subrobot_states(b)

        neighbors_l = []

        for roadmap, a_i, b_i in zip(self.roadmaps, subrobot_states_a, subrobot_states_b):
            neighbor_i = self.get_closest_subrobot_neighbor(roadmap, a_i, b_i)
            neighbors_l.append(neighbor_i)

        neighbor = torch.cat(neighbors_l)

        return neighbor
