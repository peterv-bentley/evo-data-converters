import omf_python
import pyarrow as pa
from geoscience_object_models.components import Crs_V1_0_1_EpsgCode
from geoscience_object_models.elements import FloatArray3_V1_0_1
from geoscience_object_models.objects import Pointset_V1_2_0, Pointset_V1_2_0_Locations

import evo.logging
from evo.object.utils.data import ObjectDataClient

from ...common.utils import vertices_bounding_box
from .omf_attributes_to_evo import convert_omf_attributes

logger = evo.logging.getLogger("data_converters")


def convert_omf_pointset(
    pointset: omf_python.Element,
    project: omf_python.Project,
    reader: omf_python.Reader,
    data_client: ObjectDataClient,
    epsg_code: int,
) -> Pointset_V1_2_0:
    logger.debug(f'Converting omf_python Element: "{pointset.name}" to Pointset_V1_1_0.')

    coordinate_reference_system = Crs_V1_0_1_EpsgCode(epsg_code=epsg_code)

    geometry = pointset.geometry()

    # Convert vertices to absolute position in world space by adding the project and geometry origin
    vertices_array = reader.array_vertices(geometry.vertices) + project.origin + geometry.origin

    bounding_box_go = vertices_bounding_box(vertices_array)

    vertices_schema = pa.schema(
        [
            pa.field("x", pa.float64()),
            pa.field("y", pa.float64()),
            pa.field("z", pa.float64()),
        ]
    )

    coordinates_table = pa.Table.from_arrays(
        [pa.array(vertices_array[:, i], type=pa.float64()) for i in range(len(vertices_schema))],
        schema=vertices_schema,
    )
    coordinates_args = data_client.save_table(coordinates_table)
    coordinates_go = FloatArray3_V1_0_1.from_dict(coordinates_args)

    attributes_go = convert_omf_attributes(pointset, reader, data_client, omf_python.Location.Vertices)

    locations = Pointset_V1_2_0_Locations(
        coordinates=coordinates_go,
        attributes=attributes_go,
    )

    pointset_go = Pointset_V1_2_0(
        name=pointset.name,
        uuid=None,
        bounding_box=bounding_box_go,
        coordinate_reference_system=coordinate_reference_system,
        locations=locations,
    )

    logger.debug(f"Created: {pointset_go}")

    return pointset_go
