from dataclasses import dataclass
from typing import Literal


@dataclass
class PainterConfig:
    nside: int = 8192
    search_radius: float = 2  # multiplier on R200c: search pixels within search_radius * R200c of each halo centre
    weight_bin_width: float = 2e-5  # for plotting purposes
    halo_geometry: Literal["triaxial", "spherical"] = "triaxial"
