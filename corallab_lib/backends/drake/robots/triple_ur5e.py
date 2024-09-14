import os
import corallab_assets

from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder


class TripleUR5e:

    def __init__(self, **kwargs):
        self.urdf_path = str(corallab_assets.get_resource_path("triple_ur5/triple_ur5e.urdf"))

        builder = DiagramBuilder()
        plant, _ = AddMultibodyPlantSceneGraph(builder, 0.0)
        (model_idx,) = Parser(plant).AddModels(self.urdf_path)

        world = plant.world_frame()
        base = plant.GetFrameByName("shared_base_link")
        plant.WeldFrames(world, base)

        plant.Finalize()
        self.plant = plant
        self.model_idx = model_idx
