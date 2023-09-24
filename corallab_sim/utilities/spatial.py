import numpy as np

from scipy.spatial.transform import Rotation as R


def get_rotation(rotq=None, euler=None, rotvec=None, t_matrix=None):
    """ utility function to create transformation matrix from different input forms """
    if rotq is not None:
        m = R.from_quat(rotq)
    elif euler is not None:
        m = R.from_euler('xyz', euler)
    elif rotvec is not None:
        m = R.from_rotvec(rotvec)
    elif t_matrix is not None:
        m = R.from_matrix(t_matrix[:-1, :-1])

    return m


def get_transform(rotq=None, euler=None, rotvec=None, matrix=None, pos=(0, 0, 0)):
    """ utility function to create transformation matrix from different input forms """
    trans = np.eye(4)

    if rotq is not None:
        trans[:-1, :-1] = R.from_quat(rotq).as_matrix()
    elif euler is not None:
        trans[:-1, :-1] = R.from_euler('xyz', euler).as_matrix()
    elif rotvec is not None:
        trans[:-1, :-1] = R.from_rotvec(rotvec).as_matrix()
    elif matrix is not None:
        trans[:-1, :-1] = matrix

    trans[:-1, -1:] = np.array(pos).reshape(-1, 1)

    return trans


def invert_transform(transform):
    return np.linalg.inv(transform)


def transform_point(transform, point):
    homogenous_point = np.array([*point, 1])
    homogenous_point_prime = np.matmul(transform, homogenous_point)
    point_prime = homogenous_point_prime[0:3] / homogenous_point_prime[3]
    return point_prime


def change_basis():
    pass
