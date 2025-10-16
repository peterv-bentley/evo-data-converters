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


import numpy as np
import pyarrow as pa

from typing import Any, Dict, List

from evo_schemas.components import (
    CategoryAttribute_V1_1_0,
    ContinuousAttribute_V1_1_0,
    IntegerAttribute_V1_1_0,
    NanCategorical_V1_0_1,
    NanContinuous_V1_0_1,
    OneOfAttribute_V1_2_0,
)
from evo_schemas.elements import FloatArray1_V1_0_1, IntegerArray1_V1_0_1, LookupTable_V1_0_1


import evo.logging
from evo.objects.utils.data import ObjectDataClient

logger = evo.logging.getLogger("data_converters")


def _create_continuous_attribute(
    data_client: ObjectDataClient,
    # name: str,
    # array: vtk.vtkAbstractArray,
    # mask: npt.NDArray[np.bool_] | None,
    # grid_is_filtered: bool,
    attribute: Dict[str, Any],
) -> ContinuousAttribute_V1_1_0:
    # values = vtk_to_numpy(array)
    # # Convert to float64, as Geoscience Objects only support float64 for continuous attributes
    # table = _create_table(values, mask, grid_is_filtered, np.float64)

    return ContinuousAttribute_V1_1_0(
        name=attribute["name"],
        key=attribute["name"],
        nan_description=NanContinuous_V1_0_1(values=[]),
        values=FloatArray1_V1_0_1(**data_client.save_table(attribute["values"])),
    )


def _create_integer_attribute(
    data_client: ObjectDataClient,
    # name: str,
    # array: vtk.vtkAbstractArray,
    # mask: npt.NDArray[np.bool_] | None,
    # grid_is_filtered: bool,
    attribute: Dict[str, Any],
) -> IntegerAttribute_V1_1_0:
    # values = vtk_to_numpy(array)
    # # Convert to int32 or int64
    # dtype = np.int64 if values.dtype in [np.uint32, np.int64] else np.int32
    # table = _create_table(values, mask, grid_is_filtered, dtype)

    return IntegerAttribute_V1_1_0(
        name=attribute["name"],
        key=attribute["name"],
        nan_description=NanCategorical_V1_0_1(values=[]),
        values=IntegerArray1_V1_0_1(**data_client.save_table(attribute["values"])),
    )


_numpy_dtype_for_pyarrow_type = {
    pa.int32(): np.int32,
    pa.int64(): np.int64,
}


def _create_categorical_attribute(
    data_client: ObjectDataClient,
    # name: str,
    # array: vtk.vtkStringArray,
    # mask: npt.NDArray[np.bool_] | None,
    # grid_is_filtered: bool,
    attribute: Dict[str, Any],
) -> CategoryAttribute_V1_1_0:
    # values = [array.GetValue(i) for i in range(array.GetNumberOfValues())]
    # arrow_array = pa.array(values, mask=~mask if mask is not None else None)

    # # Encode the array as a dictionary encoded array
    # dict_array = arrow_array.dictionary_encode()

    # indices = dict_array.indices
    # if grid_is_filtered and mask is not None:
    #     indices = indices.filter(mask)

    # # Create a lookup table
    # indices_dtype = _numpy_dtype_for_pyarrow_type[indices.type]
    # lookup_table = pa.table(
    #     {"key": np.arange(len(dict_array.dictionary), dtype=indices_dtype), "value": dict_array.dictionary}
    # )

    # values_table = pa.table({"values": indices})

    return CategoryAttribute_V1_1_0(
        name=attribute["name"],
        key=attribute["name"],
        nan_description=NanCategorical_V1_0_1(values=[]),
        table=LookupTable_V1_0_1(**data_client.save_table(attribute["table"])),
        values=IntegerArray1_V1_0_1(**data_client.save_table(attribute["values"])),
    )


def convert_attributes(
    data_client: ObjectDataClient,
    attribute_data: List[Dict[str, Any]],
) -> OneOfAttribute_V1_2_0:
    """
    Convert attributes that were extracted from VTK into Geoscience Objects attributes.

    :param attribute_data: VTK attributes
    :param data_client: Data client used to save the attribute values
    """
    attributes = []

    for item in attribute_data:
        if item["type"] == "continuous":
            attribute = _create_continuous_attribute(data_client, item)
        elif item["type"] == "integer":
            attribute = _create_integer_attribute(data_client, item)
        elif item["type"] == "category":
            attribute = _create_categorical_attribute(data_client, item)
        else:
            logger.warning(
                f"Unsupported data type {item['type']} for attribute {item['name']}, skipping this attribute"
            )
            continue
        attributes.append(attribute)
        print("evo convert attributes: ", attributes)
    return attributes
