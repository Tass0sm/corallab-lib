import pybullet as p
from corallab_sim.utilities.bullet import load_urdf
from importlib.resources import files

DYNAMICS_KWARGS= ["mass",
                  "lateralFriction",
                  "spinningFriction",
                  "rollingFriction"]


def add_plane(plane_height=0):
    plane_id = p.loadURDF('plane.urdf', basePosition=[0, 0, plane_height])
    return plane_id


def add_object(urdf_path, position=(0, 0, 0), **kwargs):
    block_id = load_urdf(p, str(urdf_path), position, (0, 0, 0, 1))

    dynamics_kwargs = {k: kwargs[k] for k in DYNAMICS_KWARGS if k in kwargs}
    if dynamics_kwargs:
        p.changeDynamics(block_id, -1, **dynamics_kwargs)

    return block_id


def add_block(**kwargs):
    block_urdf_path = files("corallab_sim.objects").joinpath("assets/cube.urdf")
    block_id = add_object(block_urdf_path, **kwargs)
    return block_id
