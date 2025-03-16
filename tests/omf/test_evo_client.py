import tempfile
from unittest import TestCase

from evo.data_converters.common import EvoWorkspaceMetadata, create_evo_object_service_and_data_client
from evo.objects import ObjectServiceClient
from evo.objects.utils.data import ObjectDataClient


class TestEvoClient(TestCase):
    def setUp(self) -> None:
        self.cache_root_dir = tempfile.TemporaryDirectory()

    def test_should_create_objects_with_minimal_metadata(self) -> None:
        metadata = EvoWorkspaceMetadata(cache_root=self.cache_root_dir.name)
        object_service_client, data_client = create_evo_object_service_and_data_client(metadata)

        self.assertIsInstance(object_service_client, ObjectServiceClient)
        self.assertIsInstance(data_client, ObjectDataClient)

    def test_should_create_objects_with_detailed_metadata(self) -> None:
        metadata = EvoWorkspaceMetadata(
            hub_url="https://example.com",
            org_id="8ac3f041-b186-41f9-84ba-43d60f8683be",  # randomly generated
            workspace_id="2cf1697f-2771-485e-848d-e6674d2ac63f",  # randomly generated
            cache_root=self.cache_root_dir.name,
        )
        object_service_client, data_client = create_evo_object_service_and_data_client(metadata)

        self.assertIsInstance(object_service_client, ObjectServiceClient)
        self.assertIsInstance(data_client, ObjectDataClient)
