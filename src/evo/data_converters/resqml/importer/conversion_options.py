from dataclasses import dataclass


@dataclass
class ResqmlConversionOptions:
    """Options to control the conversion of RESQML files"""

    """Only the active cells in grids are to be exported (default True)"""
    active_cells_only: bool = True

    """The grid.corner_points array can get very large.
       Grids will only be converted if the estimated size of grid.corner_points
       is less than the threshold

       Default 8 GiB"""
    memory_threshold: int = 8 * 1024 * 1024 * 1024
