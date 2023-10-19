import pybullet as p
from corallab_sim.utilities.bullet import load_urdf
from importlib.resources import files


def add_block(position):
    block_urdf_path = files("corallab_sim.objects").joinpath("assets/cube.urdf")
    block_id = load_urdf(p, str(block_urdf_path), position, (0, 0, 0, 1))
    return block_id
