from collections import defaultdict
from pathlib import PurePosixPath

from evo_schemas.components import BaseSpatialDataProperties_V1_0_1


def generate_paths(object_models: list[BaseSpatialDataProperties_V1_0_1], path_prefix: str = "") -> list[str]:
    """
    Generates a list of paths where each object will be uploaded to.

    The path for each object follows the pattern of: "<path_prefix>/<object_name>{_<n>}.json"

    For example: "myproject/mysite/myobject_2.json"
    """
    count: defaultdict[str, int] = defaultdict(int)
    paths: list[str] = []

    for obj in object_models:
        obj_path = obj.name + ".json"

        if (n := count[obj.name]) > 0:
            if n == 1:
                # must rename the existing path
                paths[paths.index(obj_path)] = obj.name + "_1.json"

            obj_path = f"{obj.name}_{n + 1}.json"

        paths.append(obj_path)
        count[obj.name] += 1

    if path_prefix:
        # prepend in-place
        for i, obj_path in enumerate(paths):
            paths[i] = str(PurePosixPath(path_prefix, obj_path)).lstrip("/")

    return paths
