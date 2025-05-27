from pathlib import Path

import pytest

from evo.data_converters.duf import DufCollectorContext


@pytest.fixture(scope="session")
def simple_objects_path():
    return str((Path(__file__).parent.parent / "data" / "simple_objects.duf").resolve())


@pytest.fixture(scope="session")
def simple_objects(simple_objects_path):
    with DufCollectorContext(simple_objects_path) as context:
        yield context.collector
