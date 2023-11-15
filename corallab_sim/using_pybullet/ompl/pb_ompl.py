"""
Credit to: https://github.com/lyfkyle/pybullet_ompl/
"""

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

import pybullet as p
import time
import copy
from itertools import product
from . import utils

INTERPOLATE_NUM = 500
DEFAULT_PLANNING_TIME = 5.0


class PbStateSpace(ob.RealVectorStateSpace):
    def __init__(self, num_dim) -> None:
        super().__init__(num_dim)
        self.num_dim = num_dim
        self.state_sampler = None

    def allocStateSampler(self):
        '''
        This will be called by the internal OMPL planner
        '''
        # WARN: This will cause problems if the underlying planner is multi-threaded!!!
        if self.state_sampler:
            return self.state_sampler

        # when ompl planner calls this, we will return our sampler
        return self.allocDefaultStateSampler()

    def set_state_sampler(self, state_sampler):
        '''
        Optional, Set custom state sampler.
        '''
        self.state_sampler = state_sampler


class PbOMPL():
    def __init__(self, robot, obstacles = []) -> None:
        '''
        Args
            robot: A RobotBase instance.
            obstacles: list of obstacle ids. Optional.
        '''
        self.robot = robot
        self.robot_id = robot.id
        self.obstacles = obstacles
        print(self.obstacles)

        self.space = PbStateSpace(robot.arm_num_dofs)

        bounds = ob.RealVectorBounds(robot.arm_num_dofs)
        joint_bounds = zip(self.robot.arm_lower_limits, self.robot.arm_upper_limits)
        for i, (lower_limit, upper_limit) in enumerate(joint_bounds):
            bounds.setLow(i, lower_limit)
            bounds.setHigh(i, upper_limit)
        self.space.setBounds(bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.si = self.ss.getSpaceInformation()
        # self.si.setStateValidityCheckingResolution(0.005)
        # self.collision_fn = pb_utils.get_collision_fn(self.robot_id, self.robot.joint_idx, self.obstacles, [], True, set(),
        #                                                 custom_limits={}, max_distance=0, allow_collision_links=[])

        self.set_obstacles(obstacles)
        self.set_planner("RRT") # RRT by default

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

        # update collision detection
        self.setup_collision_detection(self.robot, self.obstacles)

    def add_obstacles(self, obstacle_id):
        self.obstacles.append(obstacle_id)

    def remove_obstacles(self, obstacle_id):
        self.obstacles.remove(obstacle_id)

    def is_state_valid(self, state):
        # satisfy bounds TODO
        # Should be unecessary if joint bounds is properly set

        self.robot.set_state(self.state_to_list(state))

        # check self-collision
        # for link1, link2 in self.check_link_pairs:
        #     if utils.pairwise_link_collision(self.robot_id, link1, self.robot_id, link2):
        #         # print(get_body_name(body), get_link_name(body, link1), get_link_name(body, link2))
        #         return False

        # check collision against environment
        for body1, body2 in self.check_body_pairs:
            if utils.pairwise_collision(body1, body2):
                # print('body collision', body1, body2)
                # print(get_body_name(body1), get_body_name(body2))
                return False

        return True

    def setup_collision_detection(self, robot, obstacles, self_collisions = True, allow_collision_links = []):
        self.check_link_pairs = utils.get_self_link_pairs(robot.id, robot.arm_controllable_joints) if self_collisions else []
        moving_links = frozenset(
            [item for item in utils.get_moving_links(robot.id, robot.arm_controllable_joints) if not item in allow_collision_links])
        moving_bodies = [(robot.id, moving_links)]
        self.check_body_pairs = list(product(moving_bodies, obstacles))

    def set_planner(self, planner_name):
        '''
        Note: Add your planner here!!
        '''
        if planner_name == "PRM":
            self.planner = og.PRM(self.ss.getSpaceInformation())
        elif planner_name == "RRT":
            self.planner = og.RRT(self.ss.getSpaceInformation())
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.ss.getSpaceInformation())
        elif planner_name == "EST":
            self.planner = og.EST(self.ss.getSpaceInformation())
        elif planner_name == "FMT":
            self.planner = og.FMT(self.ss.getSpaceInformation())
        elif planner_name == "BITstar":
            self.planner = og.BITstar(self.ss.getSpaceInformation())
        else:
            print("{} not recognized, please add it first".format(planner_name))
            return

        self.ss.setPlanner(self.planner)

    def plan_start_goal(self, start, goal, allowed_time = DEFAULT_PLANNING_TIME):
        '''
        plan a path to goal from the given robot start state
        '''
        print(f"start_planning between {start} and {goal}")
        print(self.planner.params())

        orig_robot_state = self.robot.get_cur_state()

        # set the start and goal states;
        s = ob.State(self.space)
        g = ob.State(self.space)
        for i in range(len(start)):
            s[i] = start[i]
            g[i] = goal[i]

        self.ss.setStartAndGoalStates(s, g)

        # attempt to solve the problem within allowed planning time
        solved = self.ss.solve(allowed_time)
        res = False
        sol_path_list = []
        if solved:
            print("Found solution: interpolating into {} segments".format(INTERPOLATE_NUM))
            # print the path to screen
            sol_path_geometric = self.ss.getSolutionPath()
            sol_path_geometric.interpolate(INTERPOLATE_NUM)
            sol_path_states = sol_path_geometric.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            # print(len(sol_path_list))
            # print(sol_path_list)
            for sol_path in sol_path_list:
                self.is_state_valid(sol_path)
            res = True
        else:
            print("No solution found")

        # reset robot state
        self.robot.set_state(orig_robot_state)
        return res, sol_path_list

    def plan(self, goal, allowed_time = DEFAULT_PLANNING_TIME):
        '''
        plan a path to gaol from current robot state
        '''
        start = self.robot.get_cur_state()
        return self.plan_start_goal(start, goal, allowed_time=allowed_time)

    def execute(self, path, dynamics=False):
        '''
        Execute a planned plan. Will visualize in pybullet.
        Args:
            path: list[state], a list of state
            dynamics: allow dynamic simulation. If dynamics is false, this API will use robot.set_state(),
                      meaning that the simulator will simply reset robot's state WITHOUT any dynamics simulation. Since the
                      path is collision free, this is somewhat acceptable.
        '''
        for q in path:
            if dynamics:
                self.robot.move_ee(q, "joint")
            else:
                self.robot.set_state(q)
            p.stepSimulation()
            time.sleep(0.01)

    # -------------
    # Configurations
    # ------------

    def set_state_sampler(self, state_sampler):
        self.space.set_state_sampler(state_sampler)

    # -------------
    # Util
    # ------------

    def state_to_list(self, state):
        return [state[i] for i in range(self.robot.arm_num_dofs)]

    # -------
    # Reset
    # -------

    def reset(self):
        self.ss.clear()
