import os

import corallab_assets



class DualUR5:

    def __init__(self, **kwargs):
        self.urdf_path = str(corallab_assets.get_resource_path("dual_ur5/dual_ur5_test.urdf"))
