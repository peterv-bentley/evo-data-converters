import re
from collections import defaultdict

import numpy as np
import pyarrow as pa
from evo_schemas.elements import IndexArray2_V1_0_1

from evo.data_converters.common.utils import vertices_bounding_box
import evo.logging
from ..common.duf_wrapper import BaseEntity

logger = evo.logging.getLogger("data_converters")


def validify(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)[-255:]  # limit to 255 chars, keep the end


def get_name(obj: BaseEntity) -> str:
    if (label := getattr(obj, "Label", None)) is not None:
        return validify(label)
    obj_name = f"{type(obj).__name__}-{obj.Guid}"
    if (layer := getattr(obj, "Layer", None)) is not None:
        layer_name = layer.Name.split("\\")[-1]
        return validify(f"{layer_name}-{obj_name}".strip("-_"))
    else:
        return validify(obj_name)


def vertices_array_to_go_and_bbox(data_client, vertices_array, table_klass):
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
    return table_klass(**data_client.save_table(vertices_table)), bounding_box_go


def indices_array_to_go(data_client, indices_array, table_klass):
    width = indices_array.shape[1]
    indices_schema = pa.schema([pa.field(f"n{i}", pa.uint64()) for i in range(width)])
    indices_table = pa.Table.from_arrays(
        [pa.array(indices_array[:, i], type=pa.uint64()) for i in range(width)],
        schema=indices_schema,
    )
    return table_klass(**data_client.save_table(indices_table))


def parts_to_go(data_client, parts, parts_klass, chunks_klass=IndexArray2_V1_0_1):
    if parts:
        parts_schema = pa.schema([pa.field("offset", pa.uint64()), pa.field("count", pa.uint64())])
        parts_table = pa.Table.from_arrays(
            [pa.array(parts["offset"], type=pa.uint64()), pa.array(parts["count"], type=pa.uint64())],
            schema=parts_schema,
        )

        # TODO: if len(parts) > 2:
        #     # Part attributes present
        #     part_attributes_go = [convert_duf_attribute(key, values, data_client) for key, values in parts.items() if key not in {
        #         'offset', 'count'}]
        # else:
        part_attributes_go = None

        return parts_klass(
            chunks=chunks_klass(**data_client.save_table(parts_table)),
            attributes=part_attributes_go,
        )
    return None


def obj_list_and_indices_to_arrays(obj_list: list[BaseEntity], indices_arrays):
    orig_num_vertices = sum(obj.VertexList.Count for obj in obj_list)

    axes = ("X", "Y", "Z")
    vertices_array = np.fromiter(
        (getattr(vert, axis) for polyface in obj_list for vert in polyface.VertexList for axis in axes),
        dtype=np.float64,
        count=orig_num_vertices * 3,
    ).reshape(orig_num_vertices, 3)

    vertices_array, orig_to_unique = np.unique(vertices_array, return_inverse=True, axis=0)  # Ensure unique vertices
    if len(vertices_array) == orig_num_vertices:
        # No duplicates
        orig_to_unique = None

    # Work out indices in the combined original vertices array, and create parts with attributes
    parts = {"offset": [], "count": [], "attributes": defaultdict(list)}
    offset = 0
    vertex_offset = 0
    for obj, obj_indices_array in zip(obj_list, indices_arrays):
        obj_num_vertices = obj.VertexList.Count
        obj_count = len(obj_indices_array)

        obj_indices_array += vertex_offset  # Shift indices to the combined vertices array

        parts["offset"].append(offset)
        parts["count"].append(obj_count)

        offset += obj_count
        vertex_offset += obj_num_vertices

        if obj.XProperties:
            # Convert XProperties to attributes
            assert len(parts) == 2 or len(parts) == len(obj.XProperties) + 2, "Different number of attributes in object"
            for xprop in obj.XProperties:
                parts["attributes"][xprop.Key].append(xprop.Value.Value[0].Value)

    num_parts = len(obj_list)
    assert all(len(values) == num_parts for values in parts["attributes"].values()), (
        "Inconsistent attributes across objects"
    )

    indices_array = np.concatenate(indices_arrays, axis=0)

    if orig_to_unique is not None:
        # Some duplicates were removed, remap to unique array
        indices_array = orig_to_unique[indices_array]

    logger.debug(f"Num parts: {num_parts}")
    logger.debug(f"Indices: {indices_array.shape}")
    logger.debug(f"Vertices: {vertices_array.shape}")
    logger.debug(f"Num {type(obj_list[0]).__name__} attributes: {len(parts) - 2}")

    return vertices_array, indices_array, parts
