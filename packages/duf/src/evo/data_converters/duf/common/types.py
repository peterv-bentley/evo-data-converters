from dataclasses import dataclass
from typing import Protocol

import numpy
from numpy.typing import NDArray


@dataclass
class EvoAttributes:
    name: str
    values: NDArray
    type: str
    description: str
    nan_description: str | None


class IndexedEvoAttributes:
    def __init__(self, index: int, attributes: EvoAttributes):
        self._index = index
        self._attributes = attributes


class AttributedEvoData(Protocol):
    name: str
    attributes: list[EvoAttributes]


@dataclass
class FetchedTriangleMesh:
    name: str
    vertices: NDArray[numpy.float64]
    parts: list[NDArray[numpy.int32]]  # Lists of triangles
    attributes: list[EvoAttributes]


@dataclass
class FetchedLines:
    name: str
    paths: list[NDArray[numpy.float64]]  # Lists of arrays of 3D points
    attributes: list[EvoAttributes]
