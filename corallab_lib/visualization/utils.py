from torch_robotics.torch_utils.torch_utils import to_torch, to_numpy
from scipy.spatial.transform import Rotation as R

def plot_frame(ax, pose, arrow_length=0.1, arrow_alpha=1.0, arrow_linewidth=1.0, tensor_args=None):
    position = pose.position.cpu().numpy()
    m = pose.transform_mat[..., :3, :3]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    for color, column in zip(colors, range(3)):
        vec = m[:, column]
        from_p = position
        to_p = position + (vec * 0.1)
        ax.quiver(from_p[:, 0], from_p[:, 1], from_p[:, 2],
                  to_p[:, 0], to_p[:, 1], to_p[:, 2],
                  color=color,
                  length=arrow_length, normalize=True, alpha=arrow_alpha, linewidth=arrow_linewidth)
