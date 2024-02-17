import pybullet as p
from pybullet_utils.bullet_client import BulletClient
from corallab_sim.simulator import AbstractSimulator


class BulletSimulator(AbstractSimulator):
    def __init__(self, connection_mode=p.GUI, **kwargs):
        self.client = BulletClient(connection_mode=connection_mode,
                                   **kwargs)

    def step(self):
        self.client.stepSimulation()

    def close(self):
        self.client.disconnect()
