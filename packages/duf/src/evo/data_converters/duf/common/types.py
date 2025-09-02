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


class AttributedEvoData(Protocol):
    name: str
    attributes: list[EvoAttributes]


@dataclass
class FetchedTriangleMesh:
    name: str
    vertices: NDArray[numpy.float64]
    parts: list[NDArray[numpy.int32]]  # Lists of triangles
    attributes: list[EvoAttributes]
