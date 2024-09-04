from dataclasses import dataclass
from typing import Mapping

class RobotPoses(Mapping):

    def __init__(self, d):
        self._link_poses_dict = d

        batch_sizes = [v.batch_size for v in d.values()]
        assert len(set(batch_sizes)) <= 1, "All batch sizes must be equal"
        self.batch_size = batch_sizes[0]

    def __getitem__(self, name):
        return self._link_poses_dict.__getitem__(name)

    def __iter__(self):
        return self._link_poses_dict.__iter__()

    def __len__(self):
        return self._link_poses_dict.__len__()

    @property
    def link_names(self):
        return list(self._link_poses_dict.keys())
