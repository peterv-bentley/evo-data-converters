from os import path
from unittest import TestCase

from omf2 import OmfFileIoException

from evo.data_converters.omf import is_omf


class TestIsOmf(TestCase):
    def test_should_detect_omf1_file_as_omf(self) -> None:
        omf1_file = path.join(path.dirname(__file__), "data/omf1.omf")
        self.assertTrue(is_omf(omf1_file))

    def test_should_detect_omf2_file_as_omf(self) -> None:
        omf2_file = path.join(path.dirname(__file__), "data/omf2.omf")
        self.assertTrue(is_omf(omf2_file))

    def test_should_not_detect_non_omf_file_as_omf(self) -> None:
        non_omf_file = path.join(path.dirname(__file__), "data/empty_zip_file.omf")
        self.assertFalse(is_omf(non_omf_file))

    def test_should_raise_expected_exception_when_file_not_found(self) -> None:
        invalid_file_path = "invalid path"
        with self.assertRaises(OmfFileIoException) as context:
            is_omf(invalid_file_path)

        self.assertEqual(str(context.exception), "File IO error: No such file or directory (os error 2)")
