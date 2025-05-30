import numpy as np
from evo_schemas.components import (
    Crs_V1_0_1_EpsgCode,
    Segments_V1_2_0,
    Segments_V1_2_0_Indices,
    Segments_V1_2_0_Vertices,
)
from evo_schemas.objects import LineSegments_V2_1_0, LineSegments_V2_1_0_Parts

import evo.logging
from evo.objects.utils.data import ObjectDataClient

from ..common import Polyline
from .utils import (
    get_name,
    vertices_array_to_go_and_bbox,
    indices_array_to_go,
    parts_to_go,
    obj_list_and_indices_to_arrays,
)

logger = evo.logging.getLogger("data_converters")


def _create_line_segments_obj(name, vertices_array, indices_array, parts, epsg_code, data_client):
    vertices_go, bounding_box_go = vertices_array_to_go_and_bbox(data_client, vertices_array, Segments_V1_2_0_Vertices)

    indices_go = indices_array_to_go(data_client, indices_array, Segments_V1_2_0_Indices)

    parts_go = parts_to_go(data_client, parts, LineSegments_V2_1_0_Parts)

    line_segments_go = LineSegments_V2_1_0(
        name=name,
        uuid=None,
        bounding_box=bounding_box_go,
        coordinate_reference_system=Crs_V1_0_1_EpsgCode(epsg_code=epsg_code),
        segments=Segments_V1_2_0(vertices=vertices_go, indices=indices_go),
        parts=parts_go,
    )
    logger.debug(f"Created: {line_segments_go}")
    return line_segments_go


def combine_duf_polylines(
    polylines: list[Polyline],
    data_client: ObjectDataClient,
    epsg_code: int,
) -> LineSegments_V2_1_0 | None:
    if not polylines:
        logger.warning("No polylines to combine.")
        return None

    name = get_name(polylines[0].Layer)
    logger.debug(f'Combining polylines from layer: "{name}" to LineSegments_V2_1_0.')

    indices_arrays = []
    for polyline in polylines:
        pl_num_vertices = polyline.VertexList.Count
        pl_indices_array = (
            np.row_stack(
                (
                    np.arange(pl_num_vertices - 1),
                    np.arange(1, pl_num_vertices),
                )
            )
            .astype("uint64")
            .T
        )
        indices_arrays.append(pl_indices_array)

    vertices_array, indices_array, parts = obj_list_and_indices_to_arrays(polylines, indices_arrays)

    return _create_line_segments_obj(name, vertices_array, indices_array, parts, epsg_code, data_client)


def convert_duf_polyline(
    polyline: Polyline,
    data_client: ObjectDataClient,
    epsg_code: int,
) -> LineSegments_V2_1_0:
    name = get_name(polyline)
    logger.debug(f'Converting polyline: "{name}" to LineSegments_V2_1_0.')

    num_vertices = polyline.VertexList.Count

    indices_array = (
        np.row_stack(
            (
                np.arange(num_vertices - 1),
                np.arange(1, num_vertices),
            )
        )
        .astype("uint64")
        .T
    )

    vertices_array, indices_array, parts = obj_list_and_indices_to_arrays([polyline], [indices_array])

    return _create_line_segments_obj(name, vertices_array, indices_array, parts, epsg_code, data_client)
