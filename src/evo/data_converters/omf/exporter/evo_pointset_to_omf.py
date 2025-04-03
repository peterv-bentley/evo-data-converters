import asyncio
from typing import Optional
from uuid import UUID

import numpy as np
from evo_schemas.objects import Pointset_V1_1_0, Pointset_V1_2_0
from omf import PointSetElement, PointSetGeometry

from evo.objects.utils.data import ObjectDataClient

from .evo_attributes_to_omf import export_omf_attributes


def export_omf_pointset(
    object_id: UUID,
    version_id: Optional[str],
    pointset_go: Pointset_V1_1_0 | Pointset_V1_2_0,
    data_client: ObjectDataClient,
) -> PointSetElement:
    vertices_table = asyncio.run(
        data_client.download_table(object_id, version_id, pointset_go.locations.coordinates.as_dict())
    )
    vertices = np.asarray(vertices_table)
    vertex_attribute_data = export_omf_attributes(
        object_id, version_id, pointset_go.locations.attributes, "vertices", data_client
    )

    element_description = pointset_go.description if pointset_go.description else ""
    return PointSetElement(
        name=pointset_go.name,
        description=element_description,
        geometry=PointSetGeometry(vertices=vertices),
        data=vertex_attribute_data,
    )
