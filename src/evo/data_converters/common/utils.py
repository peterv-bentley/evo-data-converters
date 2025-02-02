from typing import Any

import numpy as np
from geoscience_object_models.components import BoundingBox_V1_0_1
from numpy.typing import NDArray


def vertices_bounding_box(vertices: NDArray[Any]) -> BoundingBox_V1_0_1:
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)

    return BoundingBox_V1_0_1(
        min_x=float(bbox_min[0]),
        max_x=float(bbox_max[0]),
        min_y=float(bbox_min[1]),
        max_y=float(bbox_max[1]),
        min_z=float(bbox_min[2]),
        max_z=float(bbox_max[2]),
    )
