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

from pathlib import Path
from typing import TYPE_CHECKING, Optional
import json

from pygef import read_cpt

from evo_schemas.objects import DownholeCollection_V1_3_0 as DownholeCollection

import evo.logging
from evo.data_converters.common import (
    EvoWorkspaceMetadata,
    create_evo_object_service_and_data_client,
    publish_geoscience_objects,
)
from evo.objects.data import ObjectMetadata

from .parse_gef_files import parse_gef_files
from .gef_to_downhole_collection import create_downhole_collection

logger = evo.logging.getLogger("data_converters")

if TYPE_CHECKING:
    from evo.notebooks import ServiceManagerWidget


def convert_gef(
    filepaths: list[str | Path],
    evo_workspace_metadata: Optional[EvoWorkspaceMetadata] = None,
    service_manager_widget: Optional["ServiceManagerWidget"] = None,
    tags: Optional[dict[str, str]] = None,
    upload_path: str = "",
) -> DownholeCollection | ObjectMetadata | None:
    """Converts a collection of GEF-CPT files into a Downhole Collection Geoscience Object.

    :param filepaths: List of Paths to the GEF files.
    :param evo_workspace_metadata: (Optional) Evo workspace metadata.
    :param service_manager_widget: (Optional) Service Manager Widget for use in jupyter notebooks.
    :param tags: (Optional) Dict of tags to add to the Geoscience Object.
    :param upload_path: (Optional) Path object will be published under.

    One of evo_workspace_metadata or service_manager_widget is required.

    Converted object will be published if either of the following is true:
    - evo_workspace_metadata.hub_url is present, or
    - service_manager_widget was passed to this function.

    :return: Geoscience Object or ObjectMetadata if published.

    :raise MissingConnectionDetailsError: If no connections details could be derived.
    :raise ConflictingConnectionDetailsError: If both evo_workspace_metadata and service_manager_widget present.
    """
    publish_object = True

    object_service_client, data_client = create_evo_object_service_and_data_client(
        evo_workspace_metadata=evo_workspace_metadata, service_manager_widget=service_manager_widget
    )
    if evo_workspace_metadata and not evo_workspace_metadata.hub_url:
        logger.debug("Publishing will be skipped due to missing hub_url.")
        publish_object = False

    gef_cpt_data = parse_gef_files(filepaths)
    geoscience_object = create_downhole_collection(gef_cpt_data)

    gef_data = read_cpt(filepaths[0])
    gef_object = {
        "schema": "/objects/downhole-collection/1.3.1/downhole-collection.schema.json",
        "name": "my test object",  # TODO: how will this work?
        "uuid": "07d26297-b50e-4491-aeb4-7f9ce37c47f4",  # TODO: possibly you're not supposed to allocate a UUID yourself
        "description": "my CPT data",  # TODO: how will this work?
        "extensions": {"project_id": gef_data.project_id},  # TODO: can we put some data from the header in this?
        "tags": tags,
    }

    json_string = json.dumps(gef_object, indent=4)
    print(json_string)

    if geoscience_object:
        if geoscience_object.tags is None:
            geoscience_object.tags = {}
        geoscience_object.tags["Source"] = "GEF-CPT files (via Evo Data Converters)"
        geoscience_object.tags["Stage"] = "Experimental"
        geoscience_object.tags["InputType"] = "GEF-CPT"

        # Add custom tags
        if tags:
            geoscience_object.tags.update(tags)

    object_metadata = None
    if publish_object:
        logger.debug("Publishing Geoscience Object")
        object_metadata = publish_geoscience_objects(
            [geoscience_object], object_service_client, data_client, upload_path
        )

    return object_metadata if object_metadata else geoscience_object
