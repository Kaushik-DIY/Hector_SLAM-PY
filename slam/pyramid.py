from __future__ import annotations
from dataclasses import dataclass
from typing import List
from .gridmap import GridMap


@dataclass
class MapPyramid:
    levels: List[GridMap]  # coarse -> fine


    @staticmethod
    def create(base_res: float, size_m: float, num_levels: int, l0: float, l_min: float, l_max: float) -> "MapPyramid":
        """
        Hector uses multi-resolution maps.
        We'll create coarse->fine resolutions:
          res(level i) = base_res * 2^(num_levels-1-i)
        so the last is base_res (finest).
        """
        grids = []
        for i in range(num_levels):
            res = base_res * (2 ** (num_levels - 1 - i))
            grids.append(GridMap(res=res, size_m=size_m, l0=l0, l_min=l_min, l_max=l_max))
        return MapPyramid(levels=grids)
    
    def finest(self) -> GridMap:
        return self.levels[-1]
    
