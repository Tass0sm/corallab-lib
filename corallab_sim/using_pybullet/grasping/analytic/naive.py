import numpy as np
from scipy.spatial.transform import Rotation as R
from corallab_sim.using_pybullet.grasping.analytic.utils import opposite_vertical_face_generator
from corallab_sim.using_pybullet.utils import draw_frame, draw_vec


def generate_downward_grasp(mesh):
    mesh_center = mesh.centroid

    generator = opposite_vertical_face_generator(mesh)
    face_pair = next(generator)
    generator.close()

    # pos
    face_a_points = mesh.vertices[face_pair.face_a]
    face_b_points = mesh.vertices[face_pair.face_b]
    face_pair_points = np.concatenate((face_a_points, face_b_points), axis=0)
    face_pair_centroid = np.mean(face_pair_points, axis=0)

    # breakpoint()
    # center_point = (point_pair.point_a + point_pair.point_b) / 2
    # draw_frame(point_pair.point_a)
    # draw_frame(center_point)
    # draw_frame(point_pair.point_b)

    # orn
    y_vec = mesh.face_normals[face_pair.face_a_idx]
    # y_vec /= np.linalg.norm(y_vec)

    # x_vec = np.random.randn(3)         # take a random vector
    # x_vec -= x_vec.dot(y_vec) * y_vec  # make it orthogonal to k
    # x_vec /= np.linalg.norm(x_vec)     # normalize it
    x_vec = np.array([0, 0, -1])

    z_vec = np.cross(x_vec, y_vec)

    rotation_matrix = np.stack([x_vec, y_vec, z_vec], axis=-1)
    rotation = R.from_matrix(rotation_matrix)
    rotation_quat = rotation.as_quat()

    return face_pair_centroid, rotation_quat
