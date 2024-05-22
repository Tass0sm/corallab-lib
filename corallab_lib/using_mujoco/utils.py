import mujoco


def setup_basic():
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
