import numpy as np

import torch
import torch.nn as nn
from torch import Tensor

from .stlcg_utils import Maxish, Minish
from .expression import Var, Node, And, Or, LinearExp, NegLinearExp


class Conjunction(Node):
    """Conjunction node."""

    def __init__(self, subformulas) -> None:
        super().__init__()
        self.subformulas = nn.ModuleList(subformulas)
        self.operation = Minish()

    def __str__(self) -> str:
        s: str = (
            "( AND " + ", ".join(map(str, self.subformulas)) + " )"
        )
        return s

    def time_depth(self) -> int:
        return max([c.time_depth() for c in self.subformulas])

    def _boolean(self, env: dict) -> Tensor:
        zs = [c._boolean(env) for c in self.subformulas]
        size: int = min([z.size()[2] for z in zs])
        cropped_zs = torch.stack([z[:, :, :size] for z in zs])
        z: Tensor = cropped_zs.all(dim=-1)
        return z

    def _quantitative(self, env: dict, normalize: bool = False) -> Tensor:
        zs = [c._quantitative(env) for c in self.subformulas]
        size: int = min([z.size()[2] for z in zs])
        cropped_zs = torch.stack([z[:, :, :size] for z in zs])
        z: Tensor = cropped_zs.min(dim=-1)
        return z

    def _robustness_trace(self, env, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
        xx = torch.cat([
            And.separate_and(subformula, env, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs) for subformula in self.subformulas
        ], axis=-1)
        return self.operation(xx, scale, dim=-1, keepdim=False, agm=agm, distributed=distributed)                                         # [batch_size, time_dim, ...]


class Disjunction(Node):
    """Disjunction node."""

    def __init__(self, subformulas) -> None:
        super().__init__()
        self.subformulas = subformulas
        self.operation = Maxish()

    def __str__(self) -> str:
        s: str = (
            "( OR " + ", ".join(map(str, self.subformulas)) + " )"
        )
        return s

    def time_depth(self) -> int:
        return max([c.time_depth() for c in self.subformulas])

    def _boolean(self, env: dict) -> Tensor:
        zs = [c._boolean(env) for c in self.subformulas]
        size: int = min([z.size()[2] for z in zs])
        cropped_zs = torch.stack([z[:, :, :size] for z in zs])
        z: Tensor = cropped_zs.any(dim=-1)
        return z

    def _quantitative(self, env: dict, normalize: bool = False) -> Tensor:
        zs = [c._quantitative(env) for c in self.subformulas]
        size: int = min([z.size()[2] for z in zs])
        cropped_zs = torch.stack([z[:, :, :size] for z in zs])
        z: Tensor = cropped_zs.max(dim=-1)
        return z

    def _robustness_trace(self, env, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
        xx = torch.cat([
            Or.separate_or(subformula, env, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs) for subformula in self.subformulas
        ], axis=-1)
        return self.operation(xx, scale, dim=-1, keepdim=False, agm=agm, distributed=distributed)                                         # [batch_size, time_dim, ...]


def multiply_matrix_over_trajectory_batch(matrix, trajs):
    return torch.einsum("ij, bhj -> bhi", matrix, trajs)


class LinearExp(Atom):
    """Linear expression that states A @ x <= b, This can be used to test
    whether the point x is in a polytope. In other words, x is in Poly(H, b) if
    b - H @ x > 0, in which case A = H, and b = b.
    """

    def __init__(self, var: Var, A, b) -> None:
        super().__init__()
        self.var = var
        self.register_buffer('A', A)
        self.register_buffer('b', b)

    def __str__(self) -> str:
        return f"A @ {self.var.name} <= b"

    def _boolean(self, env: dict) -> Tensor:
        x = self.var.get_value(env) # batch x horizon x dim
        prods_over_horizon = multiply_matrix_over_trajectory_batch(self.A, x)
        x = prods_over_horizon - self.b # horizon x dim2 - (1, dim2)
        z = (x <= 0).all(dim=-1) # horizon
        return z

    def _quantitative(self, env: dict, normalize: bool = False) -> Tensor:
        x = self.var.get_value(env) # batch x horizon x dim
        prods_over_horizon = multiply_matrix_over_trajectory_batch(self.A, x)
        x = prods_over_horizon - self.b # horizon x dim2 - (1, dim2)
        z = (-x).min(dim=-1).values # horizon

        if normalize:
            z: Tensor = torch.tanh(z)

        return z

    def _robustness_trace(
            self,
            env: dict,
            pscale: float = 1.0,
            **kwargs
    ):
        return self._quantitative(env) * pscale


class NegLinearExp(Atom):
    """Linear expression that states any of the rows from (A @ x) are >= b, This
    can be used to test whether the point x is outside a polytope.
    """

    def __init__(self, var: Var, A, b) -> None:
        super().__init__()
        self.var = var
        self.register_buffer('A', A)
        self.register_buffer('b', b)

    def __str__(self) -> str:
        return f"A @ {self.var.name} >= b (or)"

    def _boolean(self, env: dict) -> Tensor:
        x = self.var.get_value(env) # horizon x dim
        prods_over_horizon = multiply_matrix_over_trajectory_batch(self.A, x)
        x = prods_over_horizon - self.b # horizon x dim2 - (1, dim2)
        z = (x >= 0).any(dim=-1) # horizon
        return z

    def _quantitative(self, env: dict, normalize: bool = False) -> Tensor:
        x = self.var.get_value(env) # batches x horizon x dim
        prods_over_horizon = multiply_matrix_over_trajectory_batch(self.A, x)
        x = prods_over_horizon - self.b # horizon x dim2 - (1, dim2)
        z = x.max(dim=-1).values # horizon

        if normalize:
            z: Tensor = torch.tanh(z)

        return z

    def _robustness_trace(
            self,
            env: dict,
            pscale: float = 1.0,
            **kwargs
    ):
        return self._quantitative(env) * pscale


class InBox(LinearExp):
    """Constraint which states signal x is <= x_max and >= x_min."""

    def __init__(
            self,
            var: Var,
            x_min,
            x_max,
    ):
        neg_eye = -torch.eye(var.dim)
        pos_eye = torch.eye(var.dim)
        A = torch.vstack((neg_eye, pos_eye))
        b = torch.cat((-x_min, x_max))
        super().__init__(var=var, A=A, b=b)

        self.x_min = x_min
        self.x_max = x_max

    def __str__(self) -> str:
        return f"{self.var.name} inside {self.x_min.tolist()}, {self.x_max.tolist()}"


class NotInBox(NegLinearExp):
    """Constraint which states signal x is outside x_max and x_min."""

    def __init__(
            self,
            var: Var,
            x_min,
            x_max,
    ):
        neg_eye = -torch.eye(var.dim)
        pos_eye = torch.eye(var.dim)
        A = torch.vstack((neg_eye, pos_eye))
        b = torch.cat((-x_min, x_max))
        super().__init__(var=var, A=A, b=b)

        self.x_min = x_min
        self.x_max = x_max

    def __str__(self) -> str:
        return f"{self.var.name} outside {self.x_min.tolist()}, {self.x_max.tolist()}"
