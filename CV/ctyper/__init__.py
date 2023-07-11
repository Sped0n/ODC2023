from typing import TypeAlias
import numpy as np

# custom type aliases
Coordinate: TypeAlias = tuple[int, int]
Distance: TypeAlias = int
Vector: TypeAlias = tuple[int, int]
Image: TypeAlias = np.ndarray
Box: TypeAlias = np.ndarray
Array: TypeAlias = np.ndarray
PixelCoordinate: TypeAlias = tuple[int, int]
Treasure: TypeAlias = tuple[int, int, int]


# custom exepctions
class NoAnchorFound(Exception):
    pass


class NullArea(Exception):
    pass


class QuantityMismatch(Exception):
    pass


class TreasureNull(Exception):
    pass
