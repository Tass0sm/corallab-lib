import copy
from corallab_sim.using_pybullet.ompl import pb_ompl


class OMPLRobotMixin():

    def setup_ompl_interface(self, obstacles):
        self.pb_ompl_interface = pb_ompl.PbOMPL(self, obstacles)

    def set_state(self, state):
        '''
        Set robot state.
        To faciliate collision checking
        Args:
            state: list[Float], joint values of robot
        '''
        self.set_q(state)
        self.state = state

    def get_cur_state(self):
        return copy.deepcopy(self.state)
