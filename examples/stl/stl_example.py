import torch

from corallab_lib import stl

configuration_variable = stl.Var("q", dim=2)
in_box = stl.InBox(configuration_variable, torch.tensor([-1.0, 0.5]), torch.tensor([-0.5, 1.0]))
in_box_for_5 = stl.Always(in_box, right_time_bound=5)
stl_expression = stl.Eventually(in_box_for_5, left_time_bound=5, right_time_bound=15)

# If necessary, can move parameters to GPU
stl_expression = stl_expression.to("cuda")


traj = torch.tensor([[[0.0, 0.0],
                      [0.1, 0.1],
                      [0.2, 0.2],
                      [0.3, 0.3],
                      [0.4, 0.4],
                      [0.5, 0.5]]])


# in_box_for_5 = stl.Always(in_box, right_time_bound=5)

near_origin = stl.InBox(configuration_variable, torch.tensor([-0.01, -0.01]), torch.tensor([0.25, 0.25]))
far_from_origin = stl.InBox(configuration_variable, torch.tensor([0.25, 0.25]), torch.tensor([0.7, 0.7]))

near_until_far = stl.Until(near_origin, far_from_origin)

breakpoint()

near_until_far.robustness_trace({"q": traj})
# near_origin
