import omf_python
import pyarrow as pa
from geoscience_object_models.components import (
    Crs_V1_0_1_EpsgCode,
    Triangles_V1_2_0,
    Triangles_V1_2_0_Indices,
    Triangles_V1_2_0_Vertices,
)
from geoscience_object_models.objects import TriangleMesh_V2_1_0

import evo.logging
from evo.objects.utils.data import ObjectDataClient

from ...common.utils import vertices_bounding_box
from .omf_attributes_to_evo import convert_omf_attributes

logger = evo.logging.getLogger("data_converters")


def convert_omf_surface(
    surface: omf_python.Element,
    project: omf_python.Project,
    reader: omf_python.Reader,
    data_client: ObjectDataClient,
    epsg_code: int,
) -> TriangleMesh_V2_1_0:
    logger.debug(f'Converting omf_python Element: "{surface.name}" to TriangleMesh_V2_0_0.')

    coordinate_reference_system = Crs_V1_0_1_EpsgCode(epsg_code=epsg_code)

    geometry = surface.geometry()

    # Convert vertices to absolute position in world space by adding the project and geometry origin
    vertices_array = reader.array_vertices(geometry.vertices) + project.origin + geometry.origin
    indices_array = reader.array_triangles(geometry.triangles)

    bounding_box_go = vertices_bounding_box(vertices_array)

    vertices_schema = pa.schema(
        [
            pa.field("x", pa.float64()),
            pa.field("y", pa.float64()),
            pa.field("z", pa.float64()),
        ]
    )

    indices_schema = pa.schema([pa.field("x", pa.uint64()), pa.field("y", pa.uint64()), pa.field("z", pa.uint64())])

    vertices_table = pa.Table.from_arrays(
        [pa.array(vertices_array[:, i], type=pa.float64()) for i in range(len(vertices_schema))],
        schema=vertices_schema,
    )

    indices_table = pa.Table.from_arrays(
        [pa.array(indices_array[:, i], type=pa.uint64()) for i in range(len(indices_schema))],
        schema=indices_schema,
    )

    vertex_attributes_go = convert_omf_attributes(surface, reader, data_client, omf_python.Location.Vertices)
    triangle_attributes_go = convert_omf_attributes(surface, reader, data_client, omf_python.Location.Primitives)

    mesh_vertices_go = Triangles_V1_2_0_Vertices(
        **data_client.save_table(vertices_table), attributes=vertex_attributes_go
    )
    mesh_triangle_indices_go = Triangles_V1_2_0_Indices(
        **data_client.save_table(indices_table), attributes=triangle_attributes_go
    )

    triangle_mesh_go = TriangleMesh_V2_1_0(
        name=surface.name,
        uuid=None,
        bounding_box=bounding_box_go,
        coordinate_reference_system=coordinate_reference_system,
        triangles=Triangles_V1_2_0(vertices=mesh_vertices_go, indices=mesh_triangle_indices_go),
    )

    logger.debug(f"Created: {triangle_mesh_go}")

    return triangle_mesh_go
