# import trimesh
import numpy as np
from collections import namedtuple


def opposite_vertical_face_generator(mesh):
    opposite_faces = []
    OppositeFaces = namedtuple('OppositeFaces',
                               ['face_a_idx', 'face_a',
                                'face_b_idx', 'face_b'])

    for i, face_a in enumerate(mesh.faces):
        for j, face_b in enumerate(mesh.faces):
            face_a_normal = mesh.face_normals[i]
            face_b_normal = mesh.face_normals[j]

            up_vec = np.array([0, 0, 1])
            d1 = np.dot(face_a_normal, up_vec)
            d2 = np.dot(face_a_normal, face_b_normal)

            # if np.isclose(d2, -1):
            #     print("OPPOSITE", face_a_normal, up_vec, d1)

            if np.isclose(d1, 0, atol=1e-4) and np.isclose(d2, -1):
                of = OppositeFaces(i, face_a, j, face_b)
                yield of


def find_opposite_faces(mesh):
    opposite_faces = []
    OppositeFaces = namedtuple('OppositeFaces',
                               ['face_a_idx', 'face_a',
                                'face_b_idx', 'face_b'])

    for i, face_a in enumerate(mesh.faces):
        for j, face_b in enumerate(mesh.faces):
            face_a_normal = mesh.face_normals[i]
            face_b_normal = mesh.face_normals[j]

            up_vec = np.array([0, 0, 1])
            d1 = np.dot(face_a_normal, up_vec)
            d2 = np.dot(face_a_normal, face_b_normal)
            if np.isclose(d1, 0) and np.isclose(d2, -1):
                of = OppositeFaces(i, face_a, j, face_b)
                opposite_faces.append(of)

    return opposite_faces


def sample_from_face(mesh, face):
    """https://mathworld.wolfram.com/TrianglePointPicking.html"""
    vs = mesh.vertices[face]
    a_vec = vs[1] - vs[0]
    b_vec = vs[2] - vs[0]

    # two random lengths in range [0.25, 0.75)
    a_len = np.random.rand()
    a_len = a_len * 0.5 + 0.25
    b_len = np.random.rand() * (1 - a_len)
    b_len = b_len * 0.5 + 0.25

    return vs[0] + a_len * a_vec + b_len * b_vec


def find_opposite_point(mesh, face_normal, point):
    ray_origin = np.array([point])
    ray_direction = np.array([face_normal * -1])

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origin,
        ray_directions=ray_direction)

    if len(locations) == 0:
        return None, None

    original_point_mask = np.all(locations == point, axis=1)

    if len(locations[~original_point_mask]) > 0:
        last_point = locations[~original_point_mask][-1]
        last_point_face_idx = index_tri[~original_point_mask][-1]
        return last_point, last_point_face_idx
    else:
        return None, None


def antipodal_point_generator(mesh):
    opposite_faces = find_opposite_faces(mesh)
    OppositePoints = namedtuple('OppositePoints',
                                ['face_a_idx', 'point_a',
                                 'face_b_idx', 'point_b'])

    np.random.shuffle(opposite_faces)
    for of in opposite_faces:
        face_normal = mesh.face_normals[of.face_a_idx]
        point = sample_from_face(mesh, of.face_a)
        op_point, op_point_face_idx = find_opposite_point(mesh, face_normal, point)

        if op_point_face_idx == of.face_b_idx:
            yield OppositePoints(of.face_a_idx, point,
                                 of.face_b_idx, op_point)

def are_flat(point_a, point_b):
    up_vec = np.array([0, 0, 1])
    return np.dot(point_a - point_b, up_vec) == 0

def mesh_center(mesh, obj_vertcs):
    vert_count = len(mesh.vertices)
    center = None
    assert vert_count in obj_vertcs

    # hack to identify mesh type by number of vertices
    irg_vertc_map = {
        40: "cylinder",
        74: "ccuboid",
        66: "roof",
        6: "pyramid",
        8: "[regular]",
    }

    if irg_vertc_map.get(vert_count) == "ccuboid":
        # cm to nearest
        cm = mesh.center_mass
        (center,), *_ = trimesh.proximity.closest_point(mesh, [cm])
    else:
        center = mesh.center_mass

    return np.array(center)
