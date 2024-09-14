import torch
import random
import numpy.random as rnd
from typing import Union

from . import expression as stl
from . import expression_extras as stl_extras
from .expression import Node


class STLGeneratorBase:
    """
    Base Generator for STL expressions over trajectories.
    """

    def __init__(
            self,
            problem,
            leaf_prob : float = 1.0,
            inner_node_probs : list = None,
            leaf_node_probs : list = None,
            threshold_mean : float = 0.0,
            threshold_sd : float = 1.0,
            unbound_prob : float = 0.1,
            right_unbound_prob : float = 0.2,
            time_bound_max_range : float = 64,
            adaptive_unbound_temporal_ops : bool = True,
            max_timespan : int = 100,
            seed : int = 0,
    ):
        """
        leaf_prob
            probability of generating a leaf (always zero for root)
        node_types = ["not", "and", "or", "always", "eventually", "until"]
            Inner node types
        inner_node_prob
            probability vector for the different types of internal nodes
        threshold_mean
        threshold_sd
            mean and std for the normal distribution of the thresholds of atoms
        unbound_prob
            probability of a temporal operator to have a time bound o the type [0,infty]
        time_bound_max_range
            maximum value of time span of a temporal operator (i.e. max value of t in [0,t])
        adaptive_unbound_temporal_ops
            if true, unbounded temporal operators are computed from current point to the end of the signal, otherwise
            they are evaluated only at time zero.
        max_timespan
            maximum time depth of a formula.
        """

        self.problem = problem
        self.rng = rnd.default_rng(seed=seed)

        # Address the mutability of default arguments
        self.leaf_prob = leaf_prob
        self.threshold_mean = threshold_mean
        self.threshold_sd = threshold_sd
        self.unbound_prob = unbound_prob
        self.right_unbound_prob = right_unbound_prob
        self.time_bound_max_range = time_bound_max_range
        self.adaptive_unbound_temporal_ops = adaptive_unbound_temporal_ops

        self.root_node_types = ["always", "eventually"]
        self.n_root_node_types = len(self.root_node_types)

        self.inner_node_types = ["and", "or", "always", "eventually"] #, "until"
        self.n_inner_node_types = len(self.inner_node_types)

        self.leaf_node_types = ["in_box", "not_in_box"]
        self.n_leaf_node_types = len(self.leaf_node_types)

        # if inner_node_probs is None:
        #     inner_node_probs = [0.2, 0.2, 0.2, 0.2, 0.2]

        # if leaf_node_probs is None:
        #     leaf_node_probs = [0.5, 0.5]

        # self.inner_node_probs = inner_node_probs
        # self.leaf_node_probs = leaf_node_probs

        self.max_timespan = max_timespan

    def bag_sample(self, bag_size, var):
        """
        Samples a bag of bag_size formulae

        Parameters
        ----------
        bag_size : INT
            number of formulae.
        nvars : INT
            number of vars in formulae.

        Returns
        -------
        a list of formulae.

        """
        formulae = []
        for _ in range(bag_size):
            phi = self.sample(var)
            formulae.append(phi)
        return formulae

    def sample(self, var):
        """
        Samples a random formula with distribution defined in class instance parameters

        Parameters
        ----------
        nvars : number of variables of input signals
            how many variables the formula is expected to consider.

        Returns
        -------
        TYPE
            A random formula.

        """
        return self._sample_root_node(var)

    def _sample_root_node(self, var):
        node : Union[None, Node] = None

        # choose node type
        nodetype_i = self.rng.choice(self.n_root_node_types)
        nodetype = self.root_node_types[nodetype_i]

        if nodetype == "always":
            node = self._sample_always(var)
        elif nodetype == "eventually":
            node = self._sample_eventually(var)
        elif nodetype == "until":
            node = self._sample_until(var)
        else:
            raise NotImplementedError()

        return node

    def _sample_internal_node(self, var):
        node : Union[None, Node] = None

        # choose node type
        nodetype_i = self.rng.choice(self.n_inner_node_types)
        nodetype = self.inner_node_types[node_type_i]

        if nodetype == "and":
            node = self._sample_and(var)
        elif nodetype == "or":
            node = self._sample_or(var)
        elif nodetype == "always":
            node = self._sample_always(var)
        elif nodetype == "eventually":
            node = self._sample_eventually(var)
        elif nodetype == "until":
            node = self._sample_until(var)
        else:
            raise NotImplementedError()

        return node

    def _sample_leaf_node(self, var):
        # Declare & dummy-assign "idiom"
        node : Union[None, Node] = None

        # choose node type
        nodetype_i = self.rng.choice(self.n_leaf_node_types)
        nodetype = self.leaf_node_types[nodetype_i]

        x_min, x_max = self._get_random_box()

        if nodetype == "in_box":
            node = stl_extras.InBox(var, x_min, x_max)
        elif nodetype == "not_in_box":
            node = stl_extras.NotInBox(var, x_min, x_max)
        else:
            raise NotImplementedError()

        return node

    def _sample_and(self, var):
        n1 = self._sample_node(var)
        n2 = self._sample_node(var)
        return stl.And(n1, n2)

    def _sample_or(self, var):
        n1 = self._sample_node(var)
        n2 = self._sample_node(var)
        return stl.Or(n1, n2)

    def _sample_always(self, var):
        n = self._sample_node(var)
        unbound, right_unbound, left_time_bound, right_time_bound = self._get_temporal_parameters()
        return stl.Always(
            n, unbound, right_unbound, left_time_bound, right_time_bound, self.adaptive_unbound_temporal_ops
        )

    def _sample_eventually(self, var):
        n = self._sample_node(var)
        unbound, right_unbound, left_time_bound, right_time_bound = self._get_temporal_parameters()
        return stl.Eventually(
            n, unbound, right_unbound, left_time_bound, right_time_bound, self.adaptive_unbound_temporal_ops
        )

    def _sample_until(self, var):
        n1 = self._sample_node(var)
        n2 = self._sample_node(var)
        unbound, right_unbound, left_time_bound, right_time_bound = self._get_temporal_parameters()
        return stl.Until(
            n1, n2, unbound, right_unbound, left_time_bound, right_time_bound
        )

    def _sample_node(self, var):
        node = None
        if self.rng.random() < self.leaf_prob:
            node = self._sample_leaf_node(var)
        else:
            node = self._sample_internal_node(var)

        return node

    def _get_temporal_parameters(self):
        if self.rng.random() < self.unbound_prob:
            return True, False, 0, 0
        elif self.rng.random() < self.right_unbound_prob:
            return False, True, self.rng.integers(0, self.time_bound_max_range), 1
        else:
            left_bound = self.rng.integers(0, self.time_bound_max_range)
            right_bound = self.rng.integers(left_bound, self.time_bound_max_range) + 1
            return False, False, left_bound, right_bound

    def _get_random_box(self):
        box_choices = [
            # Quad 1
            (torch.tensor([0.5, 0.5]),
             torch.tensor([1.0, 1.0])),
            # Quad 2
            (torch.tensor([-1.0, 0.5]),
             torch.tensor([-0.5, 1.0])),
            # Quad 3
            (torch.tensor([-1.0, -1.0]),
             torch.tensor([-0.5, -0.5])),
            # Quad 4
            (torch.tensor([0.5, -1.0]),
             torch.tensor([1.0, -0.5])),
        ]
        n_box_choices = len(box_choices)

        box_choice_i = self.rng.choice(n_box_choices)
        x_min, x_max = box_choices[box_choice_i]
        return x_min, x_max

    # def _get_atom(self, nvars):
    #     variable = self.rng.integers(0, nvars)
    #     lte = self.rng.random() > 0.5
    #     threshold = self.rng.normal(self.threshold_mean, self.threshold_sd)
    #     return variable, threshold, lte


class STLGenerator(STLGeneratorBase):

    def _get_random_box(self):
        box_choices = [
            # Quad 1
            (torch.tensor([0.5, 0.5]),
             torch.tensor([1.0, 1.0])),
            # Quad 2
            (torch.tensor([-1.0, 0.5]),
             torch.tensor([-0.5, 1.0])),
            # Quad 3
            (torch.tensor([-1.0, -1.0]),
             torch.tensor([-0.5, -0.5])),
            # Quad 4
            (torch.tensor([0.5, -1.0]),
             torch.tensor([1.0, -0.5])),
        ]
        n_box_choices = len(box_choices)

        box_choice_i = self.rng.choice(n_box_choices)
        x_min, x_max = box_choices[box_choice_i]
        return x_min, x_max

class SimpleSTLGenerator(STLGeneratorBase):

    def _get_random_box(self):
        box_choices = [
            # # Quad 1
            # (torch.tensor([0.5, 0.5]),
            #  torch.tensor([1.0, 1.0])),
            # Quad 2
            (torch.tensor([-1.0, 0.5]),
             torch.tensor([-0.5, 1.0])),
            # # Quad 3
            # (torch.tensor([-1.0, -1.0]),
            #  torch.tensor([-0.5, -0.5])),
            # # Quad 4
            # (torch.tensor([0.5, -1.0]),
            #  torch.tensor([1.0, -0.5])),
        ]
        n_box_choices = len(box_choices)

        box_choice_i = self.rng.choice(n_box_choices)
        x_min, x_max = box_choices[box_choice_i]
        return x_min, x_max



# class STLUniformGenerator:
#     """
#     Uses Uniform Distribution.
#     """

#     def __init__(
#             self,
#             leaf_prob=0.5,
#             inner_node_prob=None,
#             threshold_bounds=None,
#             unbound_prob=0.1,
#             time_bound_max_range=20,
#             adaptive_unbound_temporal_ops=True,
#             max_timespan=100,
#     ):
#         """
#         leaf_prob
#             probability of generating a leaf (always zero for root)
#         node_types = ["not", "and", "or", "always", "eventually", "until"]
#             Inner node types
#         inner_node_prob
#             probability vector for the different types of internal nodes
#         threshold_mean
#         threshold_sd
#             mean and std for the normal distribution of the thresholds of atoms
#         unbound_prob
#             probability of a temporal operator to have a time bound of the type [0,infty]
#         time_bound_max_range
#             maximum value of time span of a temporal operator (i.e. max value of t in [0,t])
#         adaptive_unbound_temporal_ops
#             if true, unbounded temporal operators are computed from current point to the end of the signal, otherwise
#             they are evaluated only at time zero.
#         max_timespan
#             maximum time depth of a formula.


#         """

#         # Address the mutability of default arguments
#         if inner_node_prob is None:
#             inner_node_prob = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]

#         if threshold_bounds is None:
#             threshold_bounds = [-3.0, 3.0]

#         self.leaf_prob = leaf_prob
#         self.inner_node_prob = inner_node_prob
#         self.threshold_bounds = threshold_bounds
#         self.unbound_prob = unbound_prob
#         self.time_bound_max_range = time_bound_max_range
#         self.adaptive_unbound_temporal_ops = adaptive_unbound_temporal_ops
#         self.node_types = ["not", "and", "or", "always", "eventually", "until"]
#         self.max_timespan = max_timespan

#     def sample(self, var):
#         """
#         Samples a random formula with distribution defined in class instance parameters

#         Parameters
#         ----------
#         var : number of variables of input signals
#             how many variables the formula is expected to consider.

#         Returns
#         -------
#         TYPE
#             A random formula.

#         """

#         breakpoint()

#         return self._sample_internal_node(var)

#     def bag_sample(self, bag_size, var):
#         """
#         Samples a bag of bag_size formulae

#         Parameters
#         ----------
#         bag_size : INT
#             number of formulae.
#         nvars : INT
#             number of vars in formulae.

#         Returns
#         -------
#         a list of formulae.

#         """
#         formulae = []
#         for _ in range(bag_size):
#             phi = self.sample(var)
#             formulae.append(phi)
#         return formulae

#     def _sample_internal_node(self, var):
#         # Declare & dummy-assing "idiom"
#         node: Union[None, Node]
#         node = None
#         # choose node type
#         nodetype = rnd.choice(self.node_types)

#         print(nodetype)

#         if nodetype == "not":
#             n = self._sample_node(var)
#             node = stl.Not(n)
#         elif nodetype == "and":
#             n1 = self._sample_node(var)
#             n2 = self._sample_node(var)
#             node = stl.And(n1, n2)
#         elif nodetype == "or":
#             n1 = self._sample_node(var)
#             n2 = self._sample_node(var)
#             node = stl.Or(n1, n2)
#         elif nodetype == "always":
#             n = self._sample_node(var)
#             unbound, time_bound = self._get_temporal_parameters()
#             node = stl.Globally(
#                 n, unbound, time_bound, self.adaptive_unbound_temporal_ops
#             )
#         elif nodetype == "eventually":
#             n = self._sample_node(var)
#             unbound, time_bound = self._get_temporal_parameters()
#             node = stl.Eventually(
#                 n, unbound, time_bound, self.adaptive_unbound_temporal_ops
#             )
#         elif nodetype == "until":
#             raise NotImplementedError(
#                 "Node for STL 'Until' operator not yet implemented!"
#             )

#         if (node is not None) and (node.time_depth() < self.max_timespan):
#             return node

#     def _sample_node(self, var):
#         if rnd.rand() < self.leaf_prob:
#             # sample a leaf
#             var, thr, lte = self._get_atom(var)
#             return stl.Atom(var, thr, lte)
#         else:
#             return self._sample_internal_node(var)

#     def _get_temporal_parameters(self):
#         if rnd.rand() < self.unbound_prob:
#             return True, 0
#         else:
#             return False, rnd.randint(self.time_bound_max_range) + 1

#     def _get_atom(self, nvars):
#         variable = rnd.randint(nvars)
#         lte = rnd.rand() > 0.5
#         threshold = rnd.uniform(self.threshold_bounds[0], self.threshold_bounds[1])
#         return variable, threshold, lte
