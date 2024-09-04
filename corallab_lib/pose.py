import numpy as np
import torch
from torch import Tensor
import einops

from dataclasses import dataclass, field
from jaxtyping import Float
from scipy.spatial.transform import Rotation as R


@dataclass
class Pose:
    position : Float[Tensor, "b dp"]
    quaternion : Float[Tensor, "b dq"]
    transform_mat : Float[Tensor, "b 4 4"] = field(init=False)
    batch_size : int = field(init=False)

    def __post_init__(self):
        assert self.position.ndim == self.quaternion.ndim == 2
        assert self.position.shape[0] == self.quaternion.shape[0]

        self.batch_size = self.position.shape[0]

        self.transform_mat = np.expand_dims(np.eye(4), axis=0).repeat(self.batch_size, axis=0)
        self.transform_mat[:, :3, 3] = self.position.cpu()
        self.transform_mat[:, :3, :3] = R.from_quat(self.quaternion.cpu()).as_matrix()

    def transform_points(self, points):
        assert points.ndim == 2

        breakpoint()

        return None
