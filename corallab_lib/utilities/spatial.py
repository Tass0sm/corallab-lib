import numpy as np

from scipy.spatial.transform import Rotation as R


def get_rotation(rotq=None, euler=None, rotvec=None, matrix=None, t_matrix=None):
    """utility function to create transformation matrix from different input forms"""
    if rotq is not None:
        m = R.from_quat(rotq)
    elif euler is not None:
        m = R.from_euler("xyz", euler)
    elif rotvec is not None:
        m = R.from_rotvec(rotvec)
    elif matrix is not None:
        m = R.from_matrix(matrix)
    elif t_matrix is not None:
        m = R.from_matrix(t_matrix[..., :-1, :-1])

    return m


def get_transform(rotq=None, euler=None, rotvec=None, matrix=None, pos=np.array([[0, 0, 0]])):
    """utility function to create transformation matrix from different input forms"""
    if pos.ndim == 1:
        bs = 1
    else:
        bs = pos.shape[0]

    trans = np.zeros((bs, 4, 4))
    trans[:] = np.eye(4)

    if rotq is not None:
        trans[:, :-1, :-1] = R.from_quat(rotq).as_matrix()
    elif euler is not None:
        trans[:, :-1, :-1] = R.from_euler("xyz", euler).as_matrix()
    elif rotvec is not None:
        trans[:, :-1, :-1] = R.from_rotvec(rotvec).as_matrix()
    elif matrix is not None:
        trans[:, :-1, :-1] = matrix

    trans[:, :-1, -1:] = pos.reshape(bs, -1, 1)

    return trans


def decompose_transform(T):
    pos = T[:3, 3]
    rot_matrix = T[:3, :3]
    orn = get_rotation(matrix=rot_matrix).as_quat()
    return pos, orn


def invert_transform(transform):
    return np.linalg.inv(transform)


def transform_point(transform, point):
    homogenous_point = np.array([*point, 1])
    homogenous_point_prime = np.matmul(transform, homogenous_point)
    point_prime = homogenous_point_prime[0:3] / homogenous_point_prime[3]
    return point_prime

def transform_points(transform, points):
    homogenous_points = np.pad(points, ((0, 0), (0, 1)), mode="constant", constant_values=1).T
    homogenous_points_prime = np.matmul(transform, homogenous_points).T
    points_prime = homogenous_points_prime[:, :3] / homogenous_points_prime[:, 3:]
    return points_prime


def compute_transform(a_points, b_points):
    """
    - a_points: N x 3 matrix of points p1, p2, ... pn expressed in space A
    - b_points: N x 3 matrix of points p1, p2, ... pn expressed in space B

    returns T mapping points from space A to space B

    https://math.stackexchange.com/questions/1519134/how-to-find-the-best-fit-transformation-between-two-sets-of-3d-observations#1519503
    """
    a_cen = a_points.mean(axis=0)
    b_cen = b_points.mean(axis=0)

    P = a_points.T @ a_points - np.outer(a_cen, a_cen)
    Q = b_points.T @ a_points - np.outer(b_cen, a_cen)

    # # transformation which subtracts a_cen
    # a_reset_t = np.array([
    # T_a_to_b = Q @ np.linalg.pinv(P)

    return Q, np.linalg.pinv(P), a_cen, b_cen


def apply_transform(tr: tuple, src: np.ndarray) -> np.ndarray:
    Q, Pinv, s_cen, d_cen = tr
    return (Q @ Pinv @ (src - s_cen).T).T + d_cen


def change_basis(new_basis, old_basis_poses):
    """convert position (meters given in the world frame) and rot (a scipy
    spatial rotation) to a pose (x, y, z, rx, ry, rz) in the robot's base
    frame

    # I * [0, 0] = [1, 0. 1] * [?, ?]
    #              [0, 1, 1]
    #              [0, 0, 1]
    # I * [0, 0] = [1, 0. 1] * [?, ?]
    #              [0, 1, 1]
    #              [0, 0, 1]
    # I * [0, 0] = [1, 1] * [?, ?]

    For position:
    T_w * p = T_b * p'
    T_b^(-1) * T_w * p = p'
    T_b^(-1) * I * p = p'
    T_b^(-1) * p = p'

    For rotation:
    ...

    """
    new_basis = new_basis.cpu()

    # add batch dim
    if new_basis.ndim == 1:
        new_basis = new_basis.unsqueeze(0)

    T_b = get_transform(pos=new_basis[..., :3], rotq=new_basis[..., 3:]).squeeze()
    T_b_inv = invert_transform(T_b)
    R_b_inv = get_rotation(t_matrix=T_b_inv)

    # add batch dim
    if old_basis_poses.ndim == 1:
        old_basis_poses = old_basis_poses.unsqueeze(0)

    # change basis of points
    old_positions = old_basis_poses[:, :3].cpu().numpy()
    new_positions = transform_points(T_b_inv, old_positions)

    # change basis of orn
    old_rots = get_rotation(rotq=old_basis_poses[:, 3:].cpu().numpy())
    new_rot = old_rots * R_b_inv
    new_quats = new_rot.as_quat()

    # combine
    new_poses = np.hstack([new_positions, new_quats])
    return new_poses


# TODO: All planes
def random_translation(plane="xy", low=-0.1, high=0.1):
    if plane == "xy":
        random_xy = np.random.uniform(low=low, high=high, size=2)
        return np.array([*random_xy, 0])
    else:
        return np.random.uniform(low=low, high=high, size=3)


# TODO: All planes and general case
def random_rotation(plane="xy"):
    if plane == "xy":
        theta = np.random.rand() * 2 * np.pi
        return get_rotation(euler=[0, 0, theta])
    else:
        return get_rotation(euler=[0, 0, 0])


def random_transform():
    random_xy = np.random.rand(2) * 0.01
    # random_xy_rot = np.random.rand() * 2 * np.pi
    transform = get_transform(euler=[0, 0, 0], pos=(*random_xy, 0))
    return transform
