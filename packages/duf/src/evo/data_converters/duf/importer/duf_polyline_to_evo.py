import numpy as np
import pyarrow as pa
from evo_schemas.components import (
    Crs_V1_0_1_EpsgCode,
    Segments_V1_2_0,
    Segments_V1_2_0_Indices,
    Segments_V1_2_0_Vertices,
)
from evo_schemas.elements import IndexArray2_V1_0_1
from evo_schemas.objects import LineSegments_V2_1_0, LineSegments_V2_1_0_Parts

import evo.logging
from evo.objects.utils.data import ObjectDataClient

from evo.data_converters.common.utils import vertices_bounding_box
from ..common import Polyline
from .utils import get_name

logger = evo.logging.getLogger("data_converters")


def _create_line_segments_obj(parts, epsg_code, data_client, indices_array, name, vertices_array):
    bounding_box_go = vertices_bounding_box(vertices_array)

    vertices_schema = pa.schema(
        [
            pa.field("x", pa.float64()),
            pa.field("y", pa.float64()),
            pa.field("z", pa.float64()),
        ]
    )
    vertices_table = pa.Table.from_arrays(
        [pa.array(vertices_array[:, i], type=pa.float64()) for i in range(len(vertices_schema))],
        schema=vertices_schema,
    )
    vertices_go = Segments_V1_2_0_Vertices(**data_client.save_table(vertices_table))

    indices_schema = pa.schema([pa.field("n0", pa.uint64()), pa.field("n1", pa.uint64())])
    indices_table = pa.Table.from_arrays(
        [pa.array(indices_array[:, i], type=pa.uint64()) for i in range(len(indices_schema))],
        schema=indices_schema,
    )
    indices_go = Segments_V1_2_0_Indices(**data_client.save_table(indices_table))

    if parts:
        parts_schema = pa.schema([pa.field("offset", pa.uint64()), pa.field("count", pa.uint64())])
        parts_table = pa.Table.from_arrays(
            [pa.array(parts["offset"], type=pa.uint64()), pa.array(parts["count"], type=pa.uint64())],
            schema=parts_schema,
        )

        # TODO: if len(parts) > 2:
        #     # Part attributes present
        #     line_attributes_go = [convert_duf_attribute(key, values, data_client) for key, values in parts.items() if key not in {
        #         'offset', 'count'}]
        # else:
        line_attributes_go = None

        parts_go = LineSegments_V2_1_0_Parts(
            chunks=IndexArray2_V1_0_1(**data_client.save_table(parts_table)),
            attributes=line_attributes_go,
        )
    else:
        parts_go = None

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

    name = polylines[0].Layer.Name
    logger.debug(f'Combining polylines from layer: "{name}" to LineSegments_V2_1_0.')

    orig_num_vertices = sum(polyline.VertexList.Count for polyline in polylines)

    axes = ("X", "Y", "Z")
    vertices_array = np.fromiter(
        (getattr(vert, axis) for polyline in polylines for vert in polyline.VertexList for axis in axes),
        dtype=np.float64,
        count=orig_num_vertices * 3,
    ).reshape(orig_num_vertices, 3)

    vertices_array, orig_to_unique = np.unique(vertices_array, return_inverse=True, axis=0)  # Ensure unique vertices

    # Work out indices in the original vertices array
    parts = {"offset": [], "count": []}
    indices_arrays = []
    offset = 0
    vertex_offset = 0
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
            + vertex_offset
        )
        indices_arrays.append(pl_indices_array)
        count = pl_num_vertices - 1
        parts["offset"].append(offset)
        parts["count"].append(count)
        offset += count
        vertex_offset += pl_num_vertices
    indices_array = np.concatenate(indices_arrays, axis=0)

    if len(vertices_array) != orig_num_vertices:
        # Some duplicates were removed, remap to unique array
        indices_array = orig_to_unique[indices_array]

    logger.debug(f"Indices: {indices_array.shape}")
    logger.debug(f"Vertices: {vertices_array.shape}")

    if xprops := polylines[0].XProperties:
        # Same for all within a layer
        parts.update({xprop.Key: [polyline.XProperties[xprop.Key] for polyline in polylines] for xprop in xprops})
    logger.debug(f"Num polyline attributes: {len(parts) - 2}")

    return _create_line_segments_obj(parts, epsg_code, data_client, indices_array, name, vertices_array)


def convert_duf_polyline(
    polyline: Polyline,
    data_client: ObjectDataClient,
    epsg_code: int,
) -> LineSegments_V2_1_0:
    name = get_name(polyline)
    logger.debug(f'Converting polyline: "{name}" to LineSegments_V2_1_0.')

    num_vertices = polyline.VertexList.Count

    axes = ("X", "Y", "Z")
    vertices_array = np.fromiter(
        (getattr(vert, axis) for vert in polyline.VertexList for axis in axes),
        dtype=np.float64,
        count=num_vertices * 3,
    ).reshape(num_vertices, 3)

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

    logger.debug(f"Indices: {indices_array.shape}")
    logger.debug(f"Vertices: {vertices_array.shape}")

    attributes = (
        {xprop.Key: [xprop.Value.Value[0].Value] for xprop in polyline.XProperties} if polyline.XProperties else {}
    )
    logger.debug(f"Num polyline attributes: {len(attributes)}")

    # Use parts to store object-level attributes
    parts = {"offset": [0], "count": [num_vertices], **attributes} if attributes else None

    return _create_line_segments_obj(parts, epsg_code, data_client, indices_array, name, vertices_array)
