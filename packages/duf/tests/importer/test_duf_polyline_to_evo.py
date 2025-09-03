#  Copyright © 2025 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from collections import defaultdict
from os import path

import numpy as np
import pyarrow.parquet as pq
import pytest
from evo_schemas.components import (
    BoundingBox_V1_0_1,
    Crs_V1_0_1_EpsgCode,
)
from evo_schemas.objects import LineSegments_V2_1_0

import evo.data_converters.duf.common.deswik_types as dw
from evo.data_converters.duf.importer import convert_duf_polyline
from evo.data_converters.duf.importer.duf_polyline_to_evo import combine_duf_polylines


@pytest.fixture(scope="module")
def polyline_obj(simple_objects):
    return simple_objects.get_objects_of_type(dw.Polyline)[0][1]


def test_should_convert_duf_polyline_geometry(polyline_obj, data_client):
    epsg_code = 32650
    line_segments_go = convert_duf_polyline(polyline_obj, data_client, epsg_code)

    expected_line_segments_go = LineSegments_V2_1_0(
        name="0-dwPolyline-f83a4e34-0428-431c-aed7-c554febcbc4a",  # layer name - type - object guid
        uuid=None,
        coordinate_reference_system=Crs_V1_0_1_EpsgCode(epsg_code=epsg_code),
        bounding_box=line_segments_go.bounding_box,  # Tested later
        segments=line_segments_go.segments,  # Tested later
    )
    assert line_segments_go == expected_line_segments_go

    expected_bounding_box = BoundingBox_V1_0_1(
        min_x=2.0,
        max_x=5.0,
        min_y=2.0,
        max_y=5.0,
        min_z=2.0,
        max_z=5.0,
    )
    assert line_segments_go.bounding_box == expected_bounding_box

    vertices_parquet_file = path.join(str(data_client.cache_location), line_segments_go.segments.vertices.data)
    indices_parquet_file = path.join(str(data_client.cache_location), line_segments_go.segments.indices.data)

    vertices = pq.read_table(vertices_parquet_file)
    indices = pq.read_table(indices_parquet_file)

    expected_vertices = np.array([[2.0, 2.0, 2.0], [2.0, 4.0, 5.0], [5.0, 5.0, 5.0]])
    np.testing.assert_allclose(vertices.to_pandas(), expected_vertices)

    expected_indices = np.array(
        [
            [0, 1],
            [1, 2],
        ]
    )
    np.testing.assert_equal(indices.to_pandas(), expected_indices)


def test_combining_duf_polyline_geometry(multiple_objects, data_client):
    epsg_code = 32650

    # Two are from the same layer. We don't always get them in the same order, so group by layer and take the largest
    polyline_objs = [obj for _, obj in multiple_objects.get_objects_of_type(dw.Polyline)]
    by_layer = defaultdict(list)
    for pl in polyline_objs:
        by_layer[pl.Layer].append(pl)
    polyline_objs = max(by_layer.values(), key=len)
    assert len(polyline_objs) == 2
    polyline_objs = sorted(polyline_objs, key=lambda pl: pl.VertexList[0].X)  # Ensure consistent order

    line_segments_go = combine_duf_polylines(polyline_objs, data_client, epsg_code)

    expected_line_segments_go = LineSegments_V2_1_0(
        name="LINELAYER",
        uuid=None,
        coordinate_reference_system=Crs_V1_0_1_EpsgCode(epsg_code=epsg_code),
        parts=line_segments_go.parts,  # Tested later
        bounding_box=line_segments_go.bounding_box,  # Tested later
        segments=line_segments_go.segments,  # Tested later
    )
    assert line_segments_go == expected_line_segments_go

    expected_bounding_box = BoundingBox_V1_0_1(
        min_x=2.0,
        max_x=15.0,
        min_y=2.0,
        max_y=5.0,
        min_z=2.0,
        max_z=5.0,
    )
    assert line_segments_go.bounding_box == expected_bounding_box

    assert line_segments_go.parts is not None
    assert line_segments_go.parts.chunks.length == 2  # One per polyline
    assert line_segments_go.parts.chunks.width == 2  # Offset, count
    assert line_segments_go.parts.chunks.data_type == "uint64"

    chunks_parquet_file = path.join(str(data_client.cache_location), line_segments_go.parts.chunks.data)
    vertices_parquet_file = path.join(str(data_client.cache_location), line_segments_go.segments.vertices.data)
    indices_parquet_file = path.join(str(data_client.cache_location), line_segments_go.segments.indices.data)

    chunks = pq.read_table(chunks_parquet_file)
    vertices = pq.read_table(vertices_parquet_file)
    indices = pq.read_table(indices_parquet_file)

    # Two chunks, each with 2 segments
    expected_chunks = np.array(
        [
            [0, 2],
            [2, 2],
        ]
    )
    np.testing.assert_equal(chunks.to_pandas(), expected_chunks)

    # Two polylines, each with two segments, the second polyline has its vertices offset by 10 in X
    expected_vertices = np.array(
        [
            [2.0, 2.0, 2.0],
            [2.0, 4.0, 5.0],
            [5.0, 5.0, 5.0],
            [12.0, 2.0, 2.0],
            [12.0, 4.0, 5.0],
            [15.0, 5.0, 5.0],
        ]
    )
    np.testing.assert_allclose(vertices.to_pandas(), expected_vertices)

    # The second polyline has its indices offset by 3 (the number of indices in the first polyline)
    expected_indices = np.array(
        [
            [0, 1],
            [1, 2],
            [3, 4],
            [4, 5],
        ]
    )
    np.testing.assert_equal(indices.to_pandas(), expected_indices)
