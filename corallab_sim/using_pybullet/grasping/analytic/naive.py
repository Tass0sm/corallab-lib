import numpy as np
from scipy.spatial.transform import Rotation as R
from corallab_sim.using_pybullet.grasping.analytic.utils import antipodal_point_generator
from corallab_sim.using_pybullet.utils import draw_frame


def generate_collision_free_grasp(mesh):
    mesh_center = mesh.centroid
    generator = antipodal_point_generator(mesh)
    found = False

    while not found:
        point_pair = next(generator)

        center_point = (point_pair.point_a + point_pair.point_b) / 2

        # get orientation of gripper
        y_vec = point_pair - center_point
        y_vec /= np.linalg.norm(y_vec)

        x_vec = np.random.randn(3)         # take a random vector
        x_vec -= x_vec.dot(y_vec) * y_vec  # make it orthogonal to k
        x_vec /= np.linalg.norm(x_vec)     # normalize it

        z_vec = np.cross(y_vec, x_vec)

        rotation_matrix = np.hstack([x_vec, y_vec, z_vec])
        rotation_quat = R.from_matrix(rotation_matrix).as_quat()

        draw_frame(mesh_center, rotation_quat)
        break
