import os

import pytest

from evo.data_converters.common import (
    create_evo_object_service_and_data_client,
    EvoWorkspaceMetadata,
)


@pytest.fixture(scope="session")
def evo_metadata(tmp_path_factory):
    cache_root_dir = tmp_path_factory.mktemp("temp", numbered=False)
    return EvoWorkspaceMetadata(
        workspace_id="9c86938d-a40f-491a-a3e2-e823ca53c9ae",
        cache_root=cache_root_dir.name,
    )


@pytest.fixture(scope="session")
def data_client(evo_metadata):
    _, data_client = create_evo_object_service_and_data_client(evo_metadata)
    return data_client


@pytest.fixture(autouse=True, scope="function")
def cleanup():
    test_dir = os.path.dirname(__file__)
    files_to_clean = [
        # TODO Getting permission error when trying to clean these
        # os.path.join(test_dir, 'importer', 'temp'),
        # os.path.join(test_dir, 'exporter', 'temp'),
        os.path.join(test_dir, "exporter", "test_out.duf"),
    ]

    yield
    for filepath in files_to_clean:
        if os.path.exists(filepath):
            os.remove(filepath)
