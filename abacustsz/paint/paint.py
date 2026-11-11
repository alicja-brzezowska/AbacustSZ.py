import os
import re
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import asdf
import healpy as hp
import numba
import numpy as np
from loguru import logger

from abacustsz.converters import convert_rad_to_cart
from abacustsz.cosmology.angular_size import get_angular_size_from_comoving
from abacustsz.logging import memory_usage, time_calls, trace_calls
from abacustsz.paint.abacus_loader import SimulationData, load_abacus_for_painting
from abacustsz.paint.config import PainterConfig
from abacustsz.paint.pixel_search import find_pixels_in_halos
from abacustsz.paint.weights import compute_weights
from abacustsz.paths import INTERPOLATORS_PATH
from abacustsz.y_profile.closure_radius import (
    closure_radius_r200c,
    describe_table,
    truncation_tag,
)
from abacustsz.y_profile.interpolator import BattagliaLogInterpolator
from abacustsz.y_profile.y_profile import create_battaglia_profile

MODEL = create_battaglia_profile()
PYTHON_PATH = INTERPOLATORS_PATH / "y_values_python.pkl"
JAX_PATH = INTERPOLATORS_PATH / "y_values_jax.pkl"
JULIA_PATH = INTERPOLATORS_PATH / "battaglia_interpolation.jld2"


def load_interpolator(path: Path = JAX_PATH):
    return BattagliaLogInterpolator.from_pickle(path)


def get_real_space_from_eigenvals(
    eigenvalues: np.ndarray,
    r98: np.ndarray,
) -> np.ndarray:
    """Convert inertia eigenvalues to real-space semi-axis lengths a>b>c.
    Uses volume-preserving normalization abc = r200_{c}^3.
    """
    eps = np.finfo(np.float32).tiny
    eigenvalues = np.maximum(np.asarray(eigenvalues, dtype=np.float32), eps)
    r98 = np.asarray(r98, dtype=np.float32)

    ratio_ba = np.sqrt(eigenvalues[:, 2] / eigenvalues[:, 1])
    ratio_ca = np.sqrt(eigenvalues[:, 2] / eigenvalues[:, 0])

    a = r98 / np.cbrt(ratio_ba * ratio_ca)
    b = ratio_ba * a
    c = ratio_ca * a
    return np.stack([a, b, c], axis=1)


def search_radius_multiplier(config: PainterConfig, m_halos_msun: np.ndarray) -> np.ndarray:
    if config.search_radius_mode == "closure":
        mult = closure_radius_r200c(np.log10(m_halos_msun))
    elif config.search_radius_mode == "legacy":
        mult = np.where(
            m_halos_msun < config.search_radius_mass_threshold_msun,
            config.search_radius_low_mass,
            config.search_radius_high_mass,
        )
    else:
        raise ValueError(f"Unknown search_radius_mode {config.search_radius_mode!r}")
    return np.asarray(mult, dtype=np.float32)


def search_radius_header(config: PainterConfig) -> dict:
    if config.search_radius_mode == "closure":
        return {
            "search_radius_mode": "closure",
            "search_radius_table": truncation_tag("closure"),
        }
    return {
        "search_radius_mode": "legacy",
        "search_radius_low_mass": config.search_radius_low_mass,
        "search_radius_high_mass": config.search_radius_high_mass,
        "search_radius_mass_threshold_msun": config.search_radius_mass_threshold_msun,
    }


@dataclass
class PaintContext:
    pix_in_halos: np.ndarray
    zeta: np.ndarray
    halo_starts: np.ndarray
    halo_counts: np.ndarray
    weights: np.ndarray
    log_M_values: np.ndarray
    log_distances: np.ndarray
    z_values: np.ndarray
    data: SimulationData
    config: PainterConfig
    r_search: np.ndarray  # per-halo search radii (radians), same length as data.m_halos


@memory_usage
@time_calls
@trace_calls
def build_paint_context(
    config: PainterConfig,
    data: SimulationData,
    use_weights: bool = True,
) -> PaintContext:
    logger.info(
        "CPU threads: "
        f"SLURM_CPUS_PER_TASK={os.getenv('SLURM_CPUS_PER_TASK')}, " # for CSD3 purposes
        f"OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS')}, "
        f"NUMBA_NUM_THREADS={os.getenv('NUMBA_NUM_THREADS')}, "
        f"MKL_NUM_THREADS={os.getenv('MKL_NUM_THREADS')}, "
        f"OPENBLAS_NUM_THREADS={os.getenv('OPENBLAS_NUM_THREADS')}, "
        f"NUMEXPR_NUM_THREADS={os.getenv('NUMEXPR_NUM_THREADS')}"
    )
    logger.info(f"Numba threads in use: {numba.get_num_threads()}")
    logger.info(
        f"Starting paint setup: {len(data.m_halos)} halos, nside={config.nside}"
    )
    logger.info(
        f"  particle_counts: {data.particle_counts.nbytes / 1e6:.1f}MB, dtype={data.particle_counts.dtype}"
    )

    halo_xyz = convert_rad_to_cart(data.theta, data.phi)

    if config.halo_geometry == "triaxial":
        semi_axes_comoving = get_real_space_from_eigenvals(
            data.eigenvalues,
            data.radii_halos,
        )
    else:
        spherical_r_comoving = np.asarray(data.radii_halos, dtype=np.float32)
        semi_axes_comoving = np.repeat(spherical_r_comoving[:, None], 3, axis=1)

    search_radius = search_radius_multiplier(config, data.m_halos)
    logger.info(
        f"search radius mode={config.search_radius_mode}: multiplier on R200c "
        f"min={search_radius.min():.2f}, median={np.median(search_radius):.2f}, "
        f"max={search_radius.max():.2f}"
    )

    semi_axes_angular = get_angular_size_from_comoving(
        MODEL, semi_axes_comoving, data.redshift
    )
    semi_axes_angular *= search_radius[:, None]

    n_workers = config.pixel_search_workers or int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    pix_in_halos, zeta, halo_starts, halo_counts, halo_indices, projected_r_eff_2d = (
        find_pixels_in_halos(
            config.nside,
            halo_xyz,
            semi_axes_angular,
            data.eigenvectors,
            n_workers=n_workers,
            geometry=config.halo_geometry,
        )
    )

    r_search = projected_r_eff_2d
    r_90_equiv = r_search / search_radius

    logger.info(
        f"r_search stats: min={r_search.min():.3e}, median={np.median(r_search):.3e}, max={r_search.max():.3e}"
    )
    logger.info(f"pixel-halo pairs: {len(pix_in_halos):,}")
    logger.info(f"bytes: {zeta.nbytes / 1e9:.2f} GB")

    if use_weights:
        weights = compute_weights(
            config,
            pixel_indices=pix_in_halos,
            distances=zeta,
            halo_starts=halo_starts,
            halo_counts=halo_counts,
            r_90=r_90_equiv,
            search_radius=search_radius,
            particle_counts=data.particle_counts,
            method="vectorized",
        )
    else:
        weights = np.ones(len(pix_in_halos), dtype=np.float32)

    log_distances = np.clip(np.log(zeta + 1e-40), -11.5, None)

    if len(log_distances) > 0:
        logger.info(
            f"""log_distances stats: min={log_distances.min():.3e}, \n
        median={np.median(log_distances):.3e}, max={log_distances.max():.3e}"""
        )

    log_M_values = np.log10(data.m_halos)[halo_indices]

    z_values = np.full_like(zeta, data.redshift, dtype=np.float32)

    return PaintContext(
        pix_in_halos=pix_in_halos,
        zeta=zeta,
        halo_starts=halo_starts,
        halo_counts=halo_counts,
        weights=weights,
        log_M_values=log_M_values,
        log_distances=log_distances,
        z_values=z_values,
        data=data,
        config=config,
        r_search=r_search,
    )


def _slice_simulation_data(data: SimulationData, halo_idx: np.ndarray) -> SimulationData:
    return SimulationData(
        theta=data.theta[halo_idx],
        phi=data.phi[halo_idx],
        m_halos=data.m_halos[halo_idx],
        particle_counts=data.particle_counts,
        redshift=data.redshift,
        radii_halos=data.radii_halos[halo_idx],
        eigenvalues=data.eigenvalues[halo_idx],
        eigenvectors=data.eigenvectors[halo_idx],
        chi=data.chi[halo_idx] if data.chi is not None else None,
    )


def paint_one_interpolator(
    ctx: PaintContext,
    interpolator: BattagliaLogInterpolator,
) -> np.ndarray:
    y_values = interpolator.eval_for_logs(
        ctx.log_distances, ctx.z_values, ctx.log_M_values
    ) 
    y_values = np.asarray(y_values, dtype=np.float64) * ctx.weights
    y_map = np.bincount(
        ctx.pix_in_halos, weights=y_values, minlength=hp.nside2npix(ctx.config.nside)
    )
    return y_map.astype(np.float32)


def _make_short_filename(interp_stem: str) -> str:
    """3-sig-fig filename from alpha/beta_mul/gamma stem."""
    m = re.match(
        r"alpha=(?P<alpha>[^_]+)_beta_mul=(?P<beta_mul>[^_]+)_gamma=(?P<gamma>.+)$",
        interp_stem,
    )
    if m:
        alpha = float(m.group("alpha"))
        beta_mul = float(m.group("beta_mul"))
        gamma = float(m.group("gamma"))
        return f"alpha={alpha:.3g}_beta_mul={beta_mul:.3g}_gamma={gamma:.3g}"
    return interp_stem


def display_map_statistics(y_map: np.ndarray):
    logger.info("\nMap statistics:")
    logger.info(f"  Min: {y_map.min():.3e}")
    logger.info(f"  Max: {y_map.max():.3e}")
    logger.info(f"  Mean: {y_map.mean():.3e}")
    logger.info(f"  Non-zero pixels: {np.sum(y_map > 0)}/{len(y_map)}")


def paint_abacus(
    config: PainterConfig,
    halo_catalog_index: list,  # list[HaloCatalogInfo] from load_halo_catalog_index()
    healcounts_file_1: Path,
    output_file: str,
    use_weights: bool = True,
    logm_min: float = 11.5,
    run_label: str = "",
    interp_dir: Path | None = None,
):
    """
    Paint the y-compton map using Abacus halo catalogs and heal-counts.
    """
    data = load_abacus_for_painting(
        halo_catalog_index=halo_catalog_index,
        healcounts_file_1=healcounts_file_1,
        nside=config.nside,
        logm_min=logm_min,
    )
    paint_y_maps(
        config,
        data,
        healcounts_file_1=healcounts_file_1,
        output_file=output_file,
        use_weights=use_weights,
        run_label=run_label,
        interp_dir=interp_dir,
    )


def paint_y_maps(
    config: PainterConfig,
    data: SimulationData,
    healcounts_file_1: Path | None = None,
    output_file: str | None = None,
    use_weights: bool = True,
    run_label: str = "",
    interp_dir: Path | None = None,
):
    if output_file:
        output_path = Path(output_file)
        step_name = (
            Path(healcounts_file_1).stem.split("_")[-1]
            if healcounts_file_1 is not None
            else "unknown_step"
        )
        prefix = f"{run_label}_" if run_label else ""
        step_dir = output_path / f"{prefix}{step_name}"
        step_dir.mkdir(parents=True, exist_ok=True)

    search_dir = Path(interp_dir) if interp_dir is not None else INTERPOLATORS_PATH
    interp_files = sorted(
        f
        for f in glob(str(search_dir / "*.pkl"))
        if re.search(r"alpha=", Path(f).name)
    )

    def _output_path_for(int_filename: str) -> Path | None:
        if not output_file:
            return None
        interp_stem = Path(int_filename).stem
        short_name = _make_short_filename(interp_stem)
        return step_dir / f"{short_name}.asdf"

    def _write_map(int_filename: str, out: Path | None, ymap: np.ndarray):
        ymap_max = float(ymap.max())
        if ymap_max > 1.0:
            logger.warning(f"UNPHYSICAL ymap max={ymap_max:.3e} — possible interpolator extrapolation!")
        else:
            logger.info(f"ymap max={ymap_max:.3e}")

        if out is not None:
            af = asdf.AsdfFile(
                {
                    "header": {
                        "nside": config.nside,
                        "nest": True,
                        "redshift": data.redshift,
                        **search_radius_header(config),
                        "bin_width": config.weight_bin_width,
                        "halo_geometry": config.halo_geometry,
                        "healpix_file": str(healcounts_file_1)
                        if healcounts_file_1 is not None
                        else None,
                        "interpolator_filename": int_filename,
                    },
                    "data": {
                        "y_map": ymap,
                    },
                }
            )
            af.set_array_compression(af["data"]["y_map"], "blsc")
            af.write_to(out)
            logger.info(f"Saved to {out}")

    n_batches = max(1, config.n_halo_batches)

    if n_batches <= 1:
        ctx = build_paint_context(config, data, use_weights=use_weights)

        for int_filename in interp_files:
            logger.info(f"Processing interpolator: {Path(int_filename).name}")
            out = _output_path_for(int_filename)
            if out is not None and out.exists() and not out.is_symlink():
                logger.info(f"Output file exists, skipping: {out}")
                continue
            if out is not None and out.is_symlink():
                out.unlink()

            interpolator = BattagliaLogInterpolator.from_pickle(Path(int_filename))
            ymap = paint_one_interpolator(ctx, interpolator)
            del interpolator

            _write_map(int_filename, out, ymap)
            del ymap
        return

    targets = []
    for int_filename in interp_files:
        out = _output_path_for(int_filename)
        if out is not None and out.exists() and not out.is_symlink():
            logger.info(f"Output file exists, skipping: {out}")
            continue
        if out is not None and out.is_symlink():
            out.unlink()
        targets.append((int_filename, out))

    if not targets:
        return

    interpolators = {
        int_filename: BattagliaLogInterpolator.from_pickle(Path(int_filename))
        for int_filename, _ in targets
    }
    npix = hp.nside2npix(config.nside)
    y_maps = {int_filename: np.zeros(npix, dtype=np.float32) for int_filename, _ in targets}

    n_halos = len(data.m_halos)
    batches = [b for b in np.array_split(np.arange(n_halos), n_batches) if len(b) > 0]

    for i, halo_idx in enumerate(batches):
        logger.info(f"Halo batch {i + 1}/{len(batches)}: {len(halo_idx):,} halos")
        batch_data = _slice_simulation_data(data, halo_idx)
        ctx = build_paint_context(config, batch_data, use_weights=use_weights)

        for int_filename, _ in targets:
            logger.info(f"  Processing interpolator: {Path(int_filename).name}")
            contribution = paint_one_interpolator(ctx, interpolators[int_filename])
            y_maps[int_filename] += contribution
            del contribution

        del ctx

    for int_filename, out in targets:
        _write_map(int_filename, out, y_maps[int_filename])
        del y_maps[int_filename]
