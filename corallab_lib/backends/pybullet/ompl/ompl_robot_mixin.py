import copy

try:
    from corallab_lib.using_pybullet.ompl.pb_ompl import PbOMPL
except:
    PbOMPL = None


class OMPLRobotMixin():

    def setup_ompl_interface(self, obstacles):
        if PbOMPL is not None:
            self.pb_ompl_interface = PbOMPL(self, obstacles)
        else:
            raise Exception('No OMPL')

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
