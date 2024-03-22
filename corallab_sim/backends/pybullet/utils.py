import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R


def setup_basic(pc=p, add_plane=True, plane_height=0, headless=False):
    """sets up the camera, adds gravity, and adds the plane"""
    physics_client = pc.connect(pc.DIRECT if headless else pc.GUI)
    pc.setAdditionalSearchPath(pybullet_data.getDataPath())
    pc.setGravity(0, 0, -9.81)
    target = (-0.07796166092157364, 0.005451506469398737, -0.06238798052072525)
    dist = 1.0
    yaw = 89.6000747680664
    pitch = -17.800016403198242
    pc.resetDebugVisualizerCamera(dist, yaw, pitch, target)

    if add_plane:
        plane_id = pc.loadURDF("plane.urdf", basePosition=[0, 0, plane_height])
    else:
        plane_id = None

    return physics_client, plane_id


def load_urdf(pybullet_client, file_path, *args, **kwargs):
    """Loads the given URDF filepath."""
    # Handles most general file open case.
    try:
        return pybullet_client.loadURDF(file_path, *args, **kwargs)
    except pybullet_client.error:
        pass


def draw_vec(position, vec, color=[0, 1, 0], length=0.1):
    from_p = position
    to_p = position + (vec * length)
    p.addUserDebugLine(from_p, to_p, color, lineWidth=3, lifeTime=0)


def draw_frame(position, quaternion=[0, 0, 0, 1]):
    m = R.from_quat(quaternion).as_matrix()
    x_vec = m[:, 0]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    for color, column in zip(colors, range(3)):
        vec = m[:, column]
        from_p = position
        to_p = position + (vec * 0.1)
        p.addUserDebugLine(from_p, to_p, color, lineWidth=3, lifeTime=0)


def draw_poses(poses):
    for position, quat, _ in poses:
        draw_frame(position, quat)


def draw_text(position, text, *args, **kwargs):
    p.addUserDebugText(text.format(*args), position, **kwargs)
