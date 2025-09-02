import clr
import os
import platform
import sys

import evo.logging


if not platform.system() == "Windows":
    raise RuntimeError("This script is only supported on Windows.")

if not os.path.exists(r"C:\Program Files\Deswik"):
    raise RuntimeError("Deswik.Suite is not installed. Please install Deswik.Suite to run this script.")

installs = [pth for pth in os.listdir(r"C:\Program Files\Deswik") if "Suite" in pth]
if not installs:
    raise RuntimeError("Deswik.Suite is not installed. Please install Deswik.Suite to run this script.")


# Sort by version
def by_version(path):
    version = path.split(" ")[-1]
    year, month = version.split(".")
    return int(year), int(month)


newest = os.path.join(r"C:\Program Files\Deswik", sorted(installs, key=by_version)[-1])
logger = evo.logging.getLogger("data_converters")
logger.debug("Looking for Deswik DLLs in: %s", newest)

sys.path.append(newest)
clr.AddReference("Deswik.Duf")
clr.AddReference("Deswik.Entities")
clr.AddReference("Deswik.Entities.Cad")
clr.AddReference("Deswik.Serialization")
clr.AddReference("Deswik.Core.Structures")
