from os import path

import numpy as np
import pandas as pd
import pyarrow as pa
from evo_schemas.components import (
    BaseAttribute_V1_0_0,
    ContinuousAttribute_V1_1_0,
    IntegerAttribute_V1_1_0,
    DateTimeAttribute_V1_1_0,
    CategoryAttribute_V1_1_0,
)
from pyarrow import parquet as pq


def extract_attr_values(attr: BaseAttribute_V1_0_0, data_client):
    match attr:
        case ContinuousAttribute_V1_1_0() | IntegerAttribute_V1_1_0():
            values_parquet_file = path.join(str(data_client.cache_location), attr.values.data)
            values = pq.read_table(values_parquet_file).to_pandas().iloc[:, 0]
            if nans := attr.nan_description.values:
                values = values.astype(float)
                values[values.isin(nans)] = np.nan
            return values
        case DateTimeAttribute_V1_1_0():
            values_parquet_file = path.join(str(data_client.cache_location), attr.values.data)
            values = pq.read_table(values_parquet_file).to_pandas().iloc[:, 0]
            if nans := attr.nan_description.values:
                ts_nans = pa.array([nans], pa.timestamp("uz", "utc")).to_pandas()
                values[values.isin(ts_nans)] = pd.NaT
            return values
        case CategoryAttribute_V1_1_0():
            values_parquet_file = path.join(str(data_client.cache_location), attr.values.data)
            values = pq.read_table(values_parquet_file)
            lookup_parquet_file = path.join(str(data_client.cache_location), attr.table.data)
            lookup = pq.read_table(lookup_parquet_file).to_pandas()
            lookup.set_index(lookup.columns[0], inplace=True)
            values = lookup.loc[values.column(0).to_pylist(), lookup.columns[0]]
            if nans := attr.nan_description.values:
                values[values.isin(nans)] = None
            return values
        case _:
            raise TypeError(f"Unexpected attribute type: {type(attr)}")


def extract_single_attr_value(attr: BaseAttribute_V1_0_0, data_client):
    return extract_attr_values(attr, data_client).iloc[0]
