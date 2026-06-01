import argparse
import gc
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
from astropy.table import Table
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ACT_data_match import APERTURES_CAP, APERTURES_RR, stack_profiles

from abacustsz.cosmology.mass_conversion import H_ABACUS, MAX_M200M_H, Z_REF
from abacustsz.logging import setup_logging
from abacustsz.paint.abacus_loader import obtain_healcount_edges


NPROC = 8
FILTER_TYPE = "ring_ring_cum"  # "cap" or "ring_ring_cum"
APPLY_BEAM = True

YMAP_ROOT = Path("/home/ab2927/rds/hpc-work/tSZPaint_data")
OUT_ROOT = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/stacked_profiles")
YMAP_DIR_RAW = YMAP_ROOT / "Step0689-0695"
YMAP_DIR_CONV = YMAP_ROOT / "preconvolved_Step0689-0695"

HEALCOUNTS_DIR = Path(
    "/home/ab2927/rds/hpc-work/backlight_cp999/lightcone_healpix/total/heal-counts"
)
SUMMED_STEPS = [
    "Step0571-0576",
    "Step0582-0587",
    "Step0588-0593",
    "Step0594-0598",
    "Step0599-0604",
    "Step0605-0610",
    "Step0611-0616",
    "Step0617-0622",
    "Step0623-0627",
    "Step0628-0634",
    "Step0635-0640",
    "Step0641-0646",
    "Step0647-0652",
    "Step0653-0658",
    "Step0659-0664",
    "Step0665-0670",
    "Step0671-0676",
    "Step0677-0682",
    "Step0683-0688",
    "Step0689-0695",
]

FIDUCIAL_LRG_SNAPSHOTS = [
    YMAP_ROOT / "hod_mocks/lightcone_halos/z0.503/galaxies_rsd/LRGs.dat",
    YMAP_ROOT / "hod_mocks/lightcone_halos/z0.542/galaxies_rsd/LRGs.dat",
    YMAP_ROOT / "hod_mocks/lightcone_halos/z0.582/galaxies_rsd/LRGs.dat",
    YMAP_ROOT / "hod_mocks/lightcone_halos/z0.625/galaxies_rsd/LRGs.dat",
    YMAP_ROOT / "hod_mocks/lightcone_halos/z0.671/galaxies_rsd/LRGs.dat",
    YMAP_ROOT / "hod_mocks/lightcone_halos/z0.718/galaxies_rsd/LRGs.dat",
    YMAP_ROOT / "hod_mocks/lightcone_halos/z0.769/galaxies_rsd/LRGs.dat",
]

PARAM_NAMES = ["alpha", "beta0", "gamma", "log10P0"]

PARAM_RE = re.compile(
    r"alpha=(?P<alpha>[+-]?\d+\.?\d*(?:e[+-]?\d+)?)"
    r"_beta0=(?P<beta0>[+-]?\d+\.?\d*(?:e[+-]?\d+)?)"
    r"_gamma=(?P<gamma>[+-]?\d+\.?\d*(?:e[+-]?\d+)?)"
    r"_log10P0=(?P<log10P0>[+-]?\d+\.?\d*(?:e[+-]?\d+)?)"
)

_THETA_LRG = None
_PHI_LRG = None
_APPLY_BEAM = True
_N_WORKERS = None
_FILTER_TYPE = "cap"
_YMAP_DIR = None


def parse_params(filename: str) -> dict:
    m = PARAM_RE.search(filename)
    if m is None:
        raise ValueError(f"Could not parse params from: {filename}")
    return {k: float(v) for k, v in m.groupdict().items()}


def get_chi_range() -> tuple[float, float]:
    chi_mins, chi_maxs = [], []
    for step in SUMMED_STEPS:
        hc_file = HEALCOUNTS_DIR / f"LightCone0_total_heal-counts_{step}.asdf"
        lo, hi = obtain_healcount_edges(hc_file)
        chi_mins.append(lo)
        chi_maxs.append(hi)
    return min(chi_mins), max(chi_maxs)


def load_and_filter_lrgs(snapshots: list, chi_min: float, chi_max: float):
    tables = [Table.read(str(f), format="ascii.ecsv") for f in snapshots]
    x = np.concatenate([np.asarray(t["x"]) for t in tables])
    y = np.concatenate([np.asarray(t["y"]) for t in tables])
    z = np.concatenate([np.asarray(t["z"]) for t in tables])
    m = np.concatenate([np.asarray(t["mass"]) for t in tables])
    chi = np.sqrt(x**2 + y**2 + z**2)
    mask = (chi >= chi_min) & (chi <= chi_max) & (m <= MAX_M200M_H)
    x, y, z = x[mask], y[mask], z[mask]
    logger.info(
        f"LRGs: {len(m):,} total -> {mask.sum():,} kept  "
        f"(chi=[{chi_min:.1f},{chi_max:.1f}] Mpc/h, M200m <= {MAX_M200M_H / H_ABACUS:.2e} M_sun)"
    )
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x) % (2 * np.pi)
    return theta, phi


def _worker_init(omp_threads: int):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(omp_threads)


def _process_one(fname: str):
    """Worker: stack one y-map. Uses fork-inherited globals."""
    params = parse_params(fname)
    t0 = perf_counter()
    y_stacked, y_err = stack_profiles(
        str(_YMAP_DIR / fname),
        _THETA_LRG,
        _PHI_LRG,
        apply_beam=_APPLY_BEAM,
        n_workers=_N_WORKERS,
        filter_type=_FILTER_TYPE,
    )
    elapsed = perf_counter() - t0
    gc.collect()
    return fname, [params[k] for k in PARAM_NAMES], y_stacked, y_err, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, default=None, help="Override NPROC map workers.")
    parser.add_argument(
        "--ymap-start", type=int, default=None,
        help="Start index into sorted y-map list (for batched runs).",
    )
    parser.add_argument(
        "--ymap-end", type=int, default=None,
        help="End index into sorted y-map list (exclusive, for batched runs).",
    )
    parser.add_argument(
        "--preconvolved", action="store_true",
        help="Use preconvolved maps (skip beam convolution).",
    )
    parser.add_argument(
        "--output-name", type=str, default=None,
        help="Override output filename (without .npz extension).",
    )
    args = parser.parse_args()

    nproc = args.nproc or NPROC
    ymap_start = args.ymap_start
    ymap_end = args.ymap_end
    preconvolved = args.preconvolved
    batched = ymap_start is not None or ymap_end is not None

    ymap_dir = YMAP_DIR_CONV if preconvolved else YMAP_DIR_RAW
    apply_beam = False if preconvolved else APPLY_BEAM
    output_name = f"{args.output_name}.npz" if args.output_name else f"fiducial_{FILTER_TYPE}.npz"

    setup_logging(f"{FILTER_TYPE}_stack")

    chi_min, chi_max = get_chi_range()
    logger.info(f"y-map chi range: [{chi_min:.1f}, {chi_max:.1f}] Mpc/h")

    global _THETA_LRG, _PHI_LRG, _APPLY_BEAM, _N_WORKERS, _FILTER_TYPE, _YMAP_DIR
    _THETA_LRG, _PHI_LRG = load_and_filter_lrgs(FIDUCIAL_LRG_SNAPSHOTS, chi_min, chi_max)

    MAX_LRG = 3_000_000
    if len(_THETA_LRG) > MAX_LRG:
        rng = np.random.default_rng(seed=42)
        sub = rng.choice(len(_THETA_LRG), MAX_LRG, replace=False)
        _THETA_LRG = _THETA_LRG[sub]
        _PHI_LRG = _PHI_LRG[sub]
        logger.info(f"Subsampled to {MAX_LRG:,} galaxies")

    _APPLY_BEAM = apply_beam
    _N_WORKERS = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())) // nproc)
    _FILTER_TYPE = FILTER_TYPE
    _YMAP_DIR = ymap_dir

    ext = ".npy" if not apply_beam else ".asdf"
    all_fnames = sorted(f.name for f in ymap_dir.glob(f"*{ext}"))
    if batched:
        all_fnames = all_fnames[ymap_start:ymap_end]
        logger.info(f"Batched run: y-maps [{ymap_start}:{ymap_end}] -> {len(all_fnames)} maps")

    out_file = OUT_ROOT / output_name
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processing {len(all_fnames)} maps ({nproc} workers) -> {out_file}")

    results = {}
    omp_per_worker = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", nproc)) // nproc)
    with ProcessPoolExecutor(
        max_workers=nproc, initializer=_worker_init, initargs=(omp_per_worker,)
    ) as ex:
        futures = {ex.submit(_process_one, fname): fname for fname in all_fnames}
        for i, fut in enumerate(as_completed(futures), 1):
            fname = futures[fut]
            try:
                fname, params, y_stacked, y_err, elapsed = fut.result()
                results[fname] = (params, y_stacked, y_err)
                logger.info(f"[{i}/{len(all_fnames)}] {fname}  ({elapsed:.1f}s)")
            except Exception as e:
                logger.error(f"[{i}/{len(all_fnames)}] FAILED {fname}: {e}")

    all_params, profiles, errors = [], [], []
    for fname in sorted(results):
        params, y_stacked, y_err = results[fname]
        all_params.append(params)
        profiles.append(y_stacked)
        errors.append(y_err)

    np.savez(
        out_file,
        params=np.array(all_params),
        param_names=PARAM_NAMES,
        profiles=np.array(profiles),
        errors=np.array(errors),
        apertures_arcmin=APERTURES_CAP if FILTER_TYPE == "cap" else APERTURES_RR,
    )
    logger.info(f"Saved {len(profiles)} profiles -> {out_file}")


if __name__ == "__main__":
    main()
