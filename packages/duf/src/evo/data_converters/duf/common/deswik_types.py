# ruff: noqa: E402

# This links the C# libraries and sets up the Python runtime to import Deswik's C# libraries
from evo.data_converters.duf.common import setup_deswik_lib_bindings  # noqa: F401

from Deswik.Core.Structures import Vector3_dp, Vector4_dp
from Deswik.Duf import (
    EntityMetadata,
    CompressionMethod,
    DufImplementation,
    FilterCriteria,
    ItemHeader,
    NotDufFileException,
    PerformanceTweaking,
    SaveByIndexSet,
    SaveEntityItem,
    SaveSet,
    SaveByEnumerableSet,
)
from Deswik.Entities import BaseEntity, PropValue, XProperty, XProperties
from Deswik.Entities.Base import DufList, SerializationBehaviour
from Deswik.Entities.Cad import (
    Activator,
    Category,
    Document,
    Figure,
    Layer,
    Polyface,
    Polyline,
    Upgrader,
    dwPolyline,
    LineType,
    Color,
    dwPoint,
)
from Deswik.Serialization import GuidReferences
from System import Boolean, Double, Guid, Int32, NullReferenceException, String, UInt32
from System.Collections.Generic import List
from System.Reflection import BindingFlags

import clr
clr.AddReference("SimpleDuf")

from SimpleDuf import Duf

__all__ = [
    # Deswik
    "Activator",
    "BaseEntity",
    "BindingFlags",
    "Category",
    "Color",
    "CompressionMethod",
    "Document",
    "Double",
    "Duf",
    "DufList",
    "DufImplementation",
    "dwPoint",
    "dwPolyline",
    "EntityMetadata",
    "Figure",
    "FilterCriteria",
    "GuidReferences",
    "ItemHeader",
    "Layer",
    "LineType",
    "NotDufFileException",
    "PerformanceTweaking",
    "Polyface",
    "Polyline",
    "PropValue",
    "SaveByEnumerableSet",
    "SaveEntityItem",
    "SaveByIndexSet",
    "SaveSet",
    "SerializationBehaviour",
    "Upgrader",
    "Vector3_dp",
    "Vector4_dp",
    "XProperty",
    "XProperties",
    # System
    "Boolean",
    "Guid",
    "NullReferenceException",
    "Int32",
    "String",
    "UInt32",
    "List",
]
