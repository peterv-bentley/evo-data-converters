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

from os import path

import numpy as np
import pyarrow.parquet as pq
import pytest
from evo_schemas.components import (
    BoundingBox_V1_0_1,
    Crs_V1_0_1_EpsgCode,
)
from evo_schemas.objects import TriangleMesh_V2_1_0

import evo.data_converters.duf.common.deswik_types as dw
from evo.data_converters.duf.importer import convert_duf_polyface
from evo.data_converters.duf.importer.duf_polyface_to_evo import indices_from_polyface


@pytest.fixture(scope="module")
def polyface_obj(simple_objects):
    return simple_objects.get_objects_of_type(dw.Polyface)[0][1]


def test_should_convert_duf_polyface_geometry(polyface_obj, data_client):
    epsg_code = 32650
    triangle_mesh_go = convert_duf_polyface(polyface_obj, data_client, epsg_code)

    expected_triangle_mesh_go = TriangleMesh_V2_1_0(
        name="FACELAYER-dwPolyface-1c14ef99-e5e3-4388-bbe6-6120344712b1",  # layer name - type - object guid
        uuid=None,
        coordinate_reference_system=Crs_V1_0_1_EpsgCode(epsg_code=epsg_code),
        bounding_box=triangle_mesh_go.bounding_box,  # Tested later
        triangles=triangle_mesh_go.triangles,  # Tested later
    )
    assert triangle_mesh_go == expected_triangle_mesh_go

    expected_bounding_box = BoundingBox_V1_0_1(
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=10.0,
        min_z=0.0,
        max_z=0.0,
    )
    assert triangle_mesh_go.bounding_box == expected_bounding_box

    vertices_parquet_file = path.join(str(data_client.cache_location), triangle_mesh_go.triangles.vertices.data)
    indices_parquet_file = path.join(str(data_client.cache_location), triangle_mesh_go.triangles.indices.data)

    vertices = pq.read_table(vertices_parquet_file)
    indices = pq.read_table(indices_parquet_file)

    expected_vertices = np.array([[0.0, 0.0, 0.0], [0, 10.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]])
    np.testing.assert_allclose(vertices.to_pandas(), expected_vertices)

    expected_indices = np.array(
        [
            [1, 0, 2],
            [2, 3, 1],
        ]
    )
    np.testing.assert_equal(indices.to_pandas(), expected_indices)


class MockDwFaceList:
    def __init__(self, data):
        self.data = data

    @property
    def Count(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


def _run_indices_from_polyface_test(data, expected):
    pf = MockDwFaceList(data)
    result = indices_from_polyface(pf)
    assert np.array_equal(expected, result)


def test_indices_from_polyface_two_tris():
    two_tris = [1, 2, 3, 1, -1] + [3, 4, 5, 3, -1]
    expected = np.array([[0, 1, 2], [2, 3, 4]])
    _run_indices_from_polyface_test(two_tris, expected)


def test_indices_from_polyface_two_tris_two_quads():
    two_tris_two_quads = [1, 2, 3, 1, -1] + [4, 5, 6, 7, -1] + [2, 3, 4, 2, -1] + [7, 8, 9, 10, -1]
    expected = np.array(
        [[0, 1, 2], [3, 4, 5], [1, 2, 3], [6, 7, 8]]
        + [[5, 6, 3], [8, 9, 6]]  # The extra triangle from the quads gets added to the end
    )
    _run_indices_from_polyface_test(two_tris_two_quads, expected)


def test_indices_from_polyface_negative_indices():
    negative_indices = [-1, 2, -3, 1, -1] + [3, -4, 5, -3, -1]
    expected = np.array([[0, 1, 2], [2, 3, 4]])
    _run_indices_from_polyface_test(negative_indices, expected)


def test_indices_from_polyface_0_index():
    indices_with_0 = [0, 1, 2, 0, -1] + [2, 3, 4, 2, -1]
    expected = np.array([[1, 2, 3]])

    if __debug__:
        with pytest.raises(AssertionError):
            _run_indices_from_polyface_test(indices_with_0, expected)
    else:
        # Test this branch by running Python with `-O`
        _run_indices_from_polyface_test(indices_with_0, expected)


def test_indices_from_polyface_incomplete():
    incomplete = [1, 2, 3, 1, -1] + [4, 5]
    expected = np.array([[0, 1, 2]])

    if __debug__:
        with pytest.raises(AssertionError):
            _run_indices_from_polyface_test(incomplete, expected)
    else:
        # Test this branch by running Python with `-O`
        _run_indices_from_polyface_test(incomplete, expected)
