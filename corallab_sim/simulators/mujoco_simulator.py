import mujoco
from mujoco import MjModel, MjData
from mujoco.viewer import launch_passive
from corallab_sim.simulator import AbstractSimulator

DEFAULT_MODEL_XML_STRING="""
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

class MujocoSimulator(AbstractSimulator):
    def __init__(self):
        self.m = MjModel.from_xml_string(DEFAULT_MODEL_XML_STRING)
        self.d = MjData(self.m)
        self.viewer = launch_passive(self.m, self.d)

    def step(self):
        mujoco.mj_step(self.m, self.d)
        self.viewer.sync()

    def close(self):
        self.viewer.close()
