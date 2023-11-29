import numpy as np
from scipy.spatial.transform import Rotation as R
from corallab_sim.using_pybullet.grasping.analytic.utils import antipodal_point_generator
from corallab_sim.using_pybullet.utils import draw_frame, draw_vec


def generate_downward_grasp(mesh):
    mesh_center = mesh.centroid
    generator = antipodal_point_generator(mesh)

    point_pair = next(generator)

    center_point = (point_pair.point_a + point_pair.point_b) / 2

    # draw_frame(point_pair.point_a)
    # draw_frame(center_point)
    # draw_frame(point_pair.point_b)

    # get orientation of gripper
    y_vec = point_pair.point_a - center_point
    y_vec /= np.linalg.norm(y_vec)

    # x_vec = np.random.randn(3)         # take a random vector
    # x_vec -= x_vec.dot(y_vec) * y_vec  # make it orthogonal to k
    # x_vec /= np.linalg.norm(x_vec)     # normalize it
    x_vec = np.array([0, 0, -1])

    z_vec = np.cross(x_vec, y_vec)

    rotation_matrix = np.stack([x_vec, y_vec, z_vec], axis=-1)
    rotation = R.from_matrix(rotation_matrix)
    rotation_quat = rotation.as_quat()

    return center_point, rotation_quat
