import numpy as np

# from . import gyms

from brax import envs
# from brax.io import model
# from brax.io import json
# from brax.io import html
# from brax.training.agents.ppo import train as ppo
# from brax.training.agents.sac import train as sac

from ..gym_interface import GymInterface


class BraxGym(GymInterface):

    def __init__(
            self,
            id: str,
            brax_backend="positional",
            seed=0,
            **kwargs
    ):
        assert brax_backend in ['generalized', 'positional', 'spring']

        self.env_impl = envs.get_environment(env_name=id,
                                             backend=brax_backend)

        # state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

        # HTML(html.render(env.sys, [state.pipeline_state]))

        # GymClass = getattr(gyms, id)
        # self.gym_impl = GymClass(
        #     **kwargs
        # )

    # @property
    # def action_space(self):
    #     return self.gym_impl.action_space

    # def seed(self, x):
    #     self.gym_impl.seed(x)

    # def reset(self, **kwargs):
    #     return self.gym_impl.reset(**kwargs)

    # def step(self, action, **kwargs):
    #     return self.gym_impl.step(action, **kwargs)

    # def sample(self):
    #     return self.gym_impl.sample()

    # def render(self):
    #     return self.gym_impl.render()

    # def close(self):
    #     self.gym_impl.close()
