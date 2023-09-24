import pybullet as p
import pybullet_data


def setup_basic(plane_height=0, headless=False):
    """sets up the camera, adds gravity, and adds the plane"""
    physics_client = p.connect(p.DIRECT if headless else p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    target = (-0.07796166092157364, 0.005451506469398737, -0.06238798052072525)
    dist = 1.0
    yaw = 89.6000747680664
    pitch = -17.800016403198242
    p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

    plane_id = p.loadURDF('plane.urdf', basePosition=[0, 0, plane_height])

    return physics_client, plane_id
