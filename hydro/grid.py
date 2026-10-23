"""Regular lon/lat grid helpers (EPSG:4326, north-up, row 0 = northernmost)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_007.2


@dataclass(frozen=True)
class Grid:
    west: float      # longitude of the western edge of column 0
    north: float     # latitude of the northern edge of row 0
    res: float       # cell size in degrees
    nrows: int
    ncols: int

    @property
    def south(self) -> float:
        return self.north - self.nrows * self.res

    @property
    def east(self) -> float:
        return self.west + self.ncols * self.res

    @property
    def lon(self) -> np.ndarray:
        return self.west + (np.arange(self.ncols) + 0.5) * self.res

    @property
    def lat(self) -> np.ndarray:
        return self.north - (np.arange(self.nrows) + 0.5) * self.res

    def bounds(self) -> dict:
        return {"west": self.west, "east": self.east, "south": self.south, "north": self.north}

    def rowcol(self, lon: float, lat: float) -> tuple[int, int]:
        col = int(np.floor((lon - self.west) / self.res))
        row = int(np.floor((self.north - lat) / self.res))
        if not (0 <= row < self.nrows and 0 <= col < self.ncols):
            raise ValueError(f"point ({lon}, {lat}) is outside the grid")
        return row, col

    def cell_area_m2(self) -> np.ndarray:
        """Area of each cell (nrows, ncols) on a sphere, in m^2."""
        lat_n = np.deg2rad(self.north - np.arange(self.nrows) * self.res)
        lat_s = np.deg2rad(self.north - (np.arange(self.nrows) + 1) * self.res)
        band = EARTH_RADIUS_M ** 2 * np.deg2rad(self.res) * (np.sin(lat_n) - np.sin(lat_s))
        return np.repeat(band[:, None], self.ncols, axis=1)

    def lat2d(self) -> np.ndarray:
        return np.repeat(self.lat[:, None], self.ncols, axis=1)

    def lon2d(self) -> np.ndarray:
        return np.repeat(self.lon[None, :], self.nrows, axis=0)

    def to_dict(self) -> dict:
        return {"west": self.west, "north": self.north, "res": self.res,
                "nrows": self.nrows, "ncols": self.ncols}
