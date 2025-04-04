"""
Utility and common functions for RESQML
"""

import pathlib

from resqpy.grid import Grid
from resqpy.surface import Surface


def get_metadata(object: Surface | Grid) -> dict[str, dict[str, str | dict[str, str]]]:
    """Generate metadata about the source file, and the RESQML object"""
    name = object.citation_title or ""
    uuid = str(object.uuid or "")
    originator = object.originator or ""
    return {
        "resqml": {
            "epc_filename": pathlib.Path(object.model.epc_file).name,
            "name": name,
            "uuid": uuid,
            "originator": originator,
        },
    }
