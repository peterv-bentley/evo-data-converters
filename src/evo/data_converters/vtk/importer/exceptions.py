class VTKImportError(Exception):
    """Exception that is raised if there is an error during the reading of a VTK file."""


class VTKConversionError(Exception):
    """Exception that is raised if there is an error during the conversion of a VTK data object."""


class GhostValueError(VTKConversionError):
    """Exception that is raised if ghost cells or points are detected in the VTK data object.

    This includes if points are blanked out in the VTK data object.
    """


class UnsupportedCellTypeError(VTKConversionError):
    """Exception that is raised if an unsupported cell type is detected in the VTK unstructured grid."""
