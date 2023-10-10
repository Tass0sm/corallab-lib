import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R


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


def load_urdf(pybullet_client, file_path, *args, **kwargs):
    """Loads the given URDF filepath."""
    # Handles most general file open case.
    try:
        return pybullet_client.loadURDF(file_path, *args, **kwargs)
    except pybullet_client.error:
        pass


def draw_frame(position, quaternion):
    m = R.from_quat(quaternion).as_matrix()
    x_vec = m[:, 0]
    colors = [[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]

    for color, column in zip(colors, range(3)):
        vec = m[:, column]
        from_p = position
        to_p = position + (vec * 0.1)
        p.addUserDebugLine(from_p, to_p, color, lineWidth=3, lifeTime=0)


def draw_poses(poses):
    for position, quat, _ in poses:
        draw_frame(position, quat)
