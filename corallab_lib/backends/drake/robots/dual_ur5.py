import os
import corallab_assets

from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder


class DualUR5:

    def __init__(self, **kwargs):
        self.urdf_path = str(corallab_assets.get_resource_path("dual_ur5/dual_ur5.urdf"))

        builder = DiagramBuilder()
        plant, _ = AddMultibodyPlantSceneGraph(builder, 0.0)
        (model_idx,) = Parser(plant).AddModels(self.urdf_path)

        world = plant.world_frame()
        base = plant.GetFrameByName("base_fixture_link")
        plant.WeldFrames(world, base)

        plant.Finalize()
        self.plant = plant
        self.model_idx = model_idx
