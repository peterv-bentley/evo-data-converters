from os import path
from unittest import TestCase

from evo.data_converters.resqml import is_resqml


class TestIsResqml(TestCase):
    def test_should_not_detect_non_resqml_file_as_resqml(self) -> None:
        non_resqml_file = path.join(path.dirname(__file__), "data/invalid.epc")
        self.assertFalse(is_resqml(non_resqml_file))

    def test_should_not_detect_unzipped_resqml_file_as_resqml(self) -> None:
        non_resqml_file = path.join(path.dirname(__file__), "data/not_zipped.epc")
        self.assertFalse(is_resqml(non_resqml_file))

    def test_should_raise_FileNotFoundError_for_non_existent_file(self) -> None:
        invalid_file_path = "invalid path"
        with self.assertRaises(FileNotFoundError) as _:
            is_resqml(invalid_file_path)

    def test_should_not_detect_omf_file_as_resqml(self) -> None:
        non_resqml_file = path.join(path.dirname(__file__), "data/omf2.epc")
        self.assertFalse(is_resqml(non_resqml_file))

    def test_should_detect_valid_epc_file_as_resqml(self) -> None:
        resqml_file = path.join(path.dirname(__file__), "data/surface.epc")
        self.assertTrue(is_resqml(resqml_file))
