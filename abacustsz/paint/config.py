from dataclasses import dataclass
from typing import Literal


@dataclass
class PainterConfig:
    nside: int = 8192
    # painting extent in units of R200c; must match the interpolator truncation
    search_radius_mode: Literal["closure", "legacy"] = "legacy"
    search_radius_low_mass: float = 10.0
    search_radius_high_mass: float = 4.0
    search_radius_mass_threshold_msun: float = 1e14
    weight_bin_width: float = 2e-5  # radians
    halo_geometry: Literal["triaxial", "spherical"] = "triaxial"
    # None -> SLURM_CPUS_PER_TASK, else 8
    pixel_search_workers: int | None = None
    n_halo_batches: int = 1
    # "density" needs interpolators built with TSZ_LOS_GEOMETRY=sphere
    ownership: Literal["none", "density"] = "none"
    ownership_c_roche: float = 1.0
    # ghost buffer in units of the largest R_p in the shell
    ownership_ghost_buffer: float = 2.2
    ownership_pair_chunk: int = 20_000_000
