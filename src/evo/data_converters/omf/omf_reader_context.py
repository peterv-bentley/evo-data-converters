from tempfile import NamedTemporaryFile, _TemporaryFileWrapper
from typing import Optional

import omf_python

import evo.logging

logger = evo.logging.getLogger("data_converters")


class OmfReaderContext:
    """OMF Reader Context

    Reads an OMF v1 or v2 file and creates an omf_python.Reader object which can be accessed via the reader() method.

    If an OMF v1 file is provided, it is automatically converted to a temporary v2 file.
    The temporary file is automatically deleted when this object is garbage collected.
    """

    def __init__(self, filepath: str):
        self._temp_file: Optional[_TemporaryFileWrapper] = None
        self._reader = self._load_omf_reader(filepath)

    def reader(self) -> omf_python.Reader:
        return self._reader

    def temp_file(self) -> Optional[_TemporaryFileWrapper]:
        return self._temp_file

    def _load_omf_reader(self, filepath: str) -> omf_python.Reader:
        """Attempts to load an omf_python.Reader object for the given OMF file.

        :param filepath: Path to the OMF file.

        :raise omf_python.OmfFileIoException: If the file does not exist.
        :raise omf_python.OmfLimitExceededException: If the json_bytes limit is reached.
        """
        if omf_python.detect_omf1(filepath):
            logger.debug(f"{filepath} detected as OMF v1, converting to a temporary v2 file.")
            self._temp_file = NamedTemporaryFile(mode="w+b", suffix=".omf")
            converter = omf_python.Omf1Converter()
            converter.convert(filepath, self._temp_file.name)

            logger.debug(f"Converted {filepath} to OMFv2 using temporary file {self._temp_file.name}")
            filepath = self._temp_file.name

        logger.debug(f"Loading omf_python.Reader with {filepath}")
        return omf_python.Reader(filepath)
