from dataclasses import dataclass
from typing import Optional

import omf


@dataclass
class OMFMetadata:
    name: Optional[str] = ""
    revision: Optional[str] = ""
    description: Optional[str] = ""

    def to_project(self, elements: list[omf.base.ProjectElement] = []) -> omf.Project:
        """Create an OMF project from this metadata

        :param elements: List of ProjectElement objects to include in the project.
        """
        project = omf.Project(name=self.name, description=self.description, revision=self.revision)

        project.elements = elements
        assert project.validate()

        return project
