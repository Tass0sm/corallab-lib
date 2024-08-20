import torch
import pickle
import numpy as np

try:
    from collections import Mapping
except ImportError:
    from collections.abc import Mapping

from corallab_lib import MotionPlanningProblem

import matplotlib.pyplot as plt
from matplotlib import patches


def to_tuple(q):
    return tuple([x.item() for x in q])


class Vertex:

    def __init__(self, q):
        self.q = q
        self.time = None
        self.edges = {}
        self._handle = None
        self.total_connection_attempts = 0
        self.successful_connection_attempts = 0

    def __lt__(self, other):
        return (self.q.norm() < other.q.norm()).item()
        # return (np.linalg.norm(self.q) < np.linalg.norm(other.q))

    def __le__(self, other):
        return (self.q.norm() <= other.q.norm()).item()
        # return (np.linalg.norm(self.q) <= np.linalg.norm(other.q))

    def __hash__(self):
        key = to_tuple(self.q)
        return hash((key, self.time))

    def __eq__(self, other):
        return (self.q == other.q).all() and (self.time == other.time)

    def q_tuple(self):
        return to_tuple(self.q)

    def clear(self):
        self._handle = None

    def draw(self, ax, color="red"):
        assert len(self.q) == 2
        p = self.q.cpu().numpy()
        x, y = p[0], p[1]
        circle = plt.Circle((x, y), 0.005, color="black", linewidth=0, alpha=1)
        ax.add_patch(circle)

    def __str__(self):
        return 'Vertex(' + str(self.q) + ')'

    __repr__ = __str__


class Edge:

    def __init__(self, v1, v2, path):
        self.v1, self.v2 = v1, v2
        self.v1.edges[v2], self.v2.edges[v1] = self, self
        self._path = path
        #self._handle = None
        self._handles = []

        self.in_shortest_path = False

    def end(self, start):
        if self.v1 == start:
            return self.v2
        if self.v2 == start:
            return self.v1
        assert False

    def path(self, start):
        if self._path is None:
            return [self.end(start).q]
        if self.v1 == start:
            return self._path + [self.v2.q]
        if self.v2 == start:
            return self._path[::-1] + [self.v1.q]
        assert False

    def configs(self):
        if self._path is None:
            return []
        return [self.v1.q] + self._path + [self.v2.q]

    def clear(self):
        #self._handle = None
        self._handles = []

    def draw(self, ax, color="red"):
        if self._path is None:
            return

        assert len(self.v1.q) == 2 and len(self.v2.q) == 2
        p1 = self.v1.q.cpu().numpy()
        p2 = self.v2.q.cpu().numpy()
        dx, dy = p2 - p1

        color = "orange" if self.in_shortest_path else "black"
        width = 0.01 if self.in_shortest_path else 0.00001
        zorder = 100 if self.in_shortest_path else 0

        line = patches.FancyArrow(p1[0], p1[1], dx, dy, width=width, head_length=0,
                                  color=color, zorder=zorder)
        ax.add_patch(line)

    def __str__(self):
        return 'Edge(' + str(self.v1.q) + ' - ' + str(self.v2.q) + ')'

    __repr__ = __str__


class Roadmap(Mapping):

    def __init__(
            self,
            problem : MotionPlanningProblem = None,
            samples : list = [],
    ):
        self.problem = problem

        self.vertices = {}
        self.edges = []
        self.add(samples)

    def __getitem__(self, q):
        if isinstance(q, torch.Tensor):
            q = to_tuple(q)
        elif isinstance(q, np.ndarray):
            q = to_tuple(q)

        return self.vertices[q]

    def __contains__(self, q):
        if isinstance(q, torch.Tensor):
            q = to_tuple(q)
        elif isinstance(q, np.ndarray):
            q = to_tuple(q)

        return q in self.vertices

    def __len__(self):
        return len(self.vertices)

    def __iter__(self):
        return iter(self.vertices)

    def add(self, samples):
        new_vertices = []
        for q in samples:
            key = to_tuple(q)
            if key not in self:
                self.vertices[key] = Vertex(q)
                new_vertices.append(self[key])

        return new_vertices

    def add_or_get(self, samples):
        vertices = []
        for q in samples:
            key = to_tuple(q)
            if key not in self:
                self.vertices[key] = Vertex(q)
                vertices.append(self[key])
            else:
                vertices.append(self[key])

        return vertices

    def connect(self, v1, v2, path=None):
        if v1 not in v2.edges:  # TODO - what about parallel edges?
            edge = Edge(v1, v2, path)
            self.edges.append(edge)
            return edge
        return None

    def clear(self):
        for v in self.vertices.values():
            v.clear()
        for e in self.edges:
            e.clear()

    def draw(self, ax):
        for v in self.vertices.values():
            v.draw(ax)
        for e in self.edges:
            e.draw(ax)

    def load_from_pkl(self, filename):
        with open(filename, 'rb') as f:
            roadmap_dict = pickle.load(f)
            self.vertices = roadmap_dict["vertices"]
            self.edges = roadmap_dict["edges"]

    def save_to_pkl(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                "vertices": self.vertices,
                "edges": self.edges,
            }, f)

    @staticmethod
    def merge(*roadmaps):
        new_roadmap = Roadmap()
        new_roadmap.vertices = merge_dicts(
            *[roadmap.vertices for roadmap in roadmaps])
        new_roadmap.edges = list(
            flatten(roadmap.edges for roadmap in roadmaps))
        return new_roadmap
