"""Stack beam-convolved y-maps on the mock LRGs with the ring-ring filter.

The output .npz (one stacked profile per Latin-hypercube point) is the training
set of the emulator.
"""

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import asdf
import numpy as np
from astropy.table import Table

from abacustsz.cosmology.model import H_ABACUS, MAX_M200C_MSUN, m200c_to_m200m, m200m_to_mdeltamean
from abacustsz.paint.abacus_loader import obtain_healcount_edges
from abacustsz.paths import HALO_CATALOGS_PATH, HEALCOUNTS_TOTAL_PATH, OUTPUT_PATH
from config import LRG_REDSHIFTS, STEPS, healcounts_file
from filters import APERTURES_ARCMIN, load_ymap, ring_ring_filter

PARAM_NAMES = ["alpha", "beta0", "gamma", "log10P0"]
NUMBER = r"([+-]?\d+\.?\d*(?:e[+-]?\d+)?)"
PARAM_RE = re.compile("_".join(f"{name}={NUMBER}" for name in PARAM_NAMES))

_theta = _phi = None
_threads = 1


def max_raw_mass(z):
    """The painting mass cut (M200c <= MAX_M200C_MSUN) in the catalogue's raw SO mass [Msun/h]."""
    with asdf.open(HALO_CATALOGS_PATH / f"z{z}" / "lightcone_halo_info_000.asdf", lazy_load=True) as af:
        so_density = float(af["header"]["SODensityL1"])
    m200m = m200c_to_m200m(MAX_M200C_MSUN * H_ABACUS, float(z))
    return m200m_to_mdeltamean(m200m, so_density, float(z))


def load_lrgs(lrg_dir, lrg_file, max_lrgs, seed=42):
    """LRG sky positions, keeping galaxies inside painted shells and below the painting mass cut."""
    x, y, z, keep = [], [], [], []
    shells = [obtain_healcount_edges(HEALCOUNTS_TOTAL_PATH / healcounts_file(s)) for s in STEPS]
    for zsnap in LRG_REDSHIFTS:
        t = Table.read(lrg_dir / f"z{zsnap}" / "galaxies_rsd" / lrg_file, format="ascii.ecsv")
        xi, yi, zi = (np.asarray(t[c]) for c in ("x", "y", "z"))
        chi = np.sqrt(xi**2 + yi**2 + zi**2)
        painted = np.any([(chi >= lo) & (chi <= hi) for lo, hi in shells], axis=0)
        keep.append(painted & (np.asarray(t["mass"]) <= max_raw_mass(zsnap)))
        x.append(xi)
        y.append(yi)
        z.append(zi)
    keep = np.concatenate(keep)
    x, y, z = (np.concatenate(a)[keep] for a in (x, y, z))
    print(f"{keep.sum():,} of {len(keep):,} LRGs kept")

    if len(x) > max_lrgs:
        sub = np.random.default_rng(seed).choice(len(x), max_lrgs, replace=False)
        x, y, z = x[sub], y[sub], z[sub]
    r = np.sqrt(x**2 + y**2 + z**2)
    return np.arccos(z / r), np.arctan2(y, x) % (2 * np.pi)


def init_worker(theta, phi, threads):
    global _theta, _phi, _threads
    _theta, _phi, _threads = theta, phi, threads


def stack_one(path):
    values = ring_ring_filter(load_ymap(path), _theta, _phi, n_workers=_threads)
    params = [float(v) for v in PARAM_RE.search(path.name).groups()]
    return path.name, params, values.mean(0), values.std(0) / np.sqrt(len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("maps_dir", type=Path, help="beam-convolved .npy maps (preconvolve_maps.py)")
    parser.add_argument("--lrg-dir", type=Path, default=OUTPUT_PATH / "hod_mocks" / "lightcone_halos")
    parser.add_argument("--lrg-file", default="LRGs.dat")
    parser.add_argument("--max-lrgs", type=int, default=3_000_000)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    theta, phi = load_lrgs(args.lrg_dir, args.lrg_file, args.max_lrgs)
    threads = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())) // args.nproc)
    maps = sorted(args.maps_dir.glob("*.npy"))
    print(f"Stacking {len(theta):,} LRGs on {len(maps)} maps")

    with ProcessPoolExecutor(args.nproc, initializer=init_worker, initargs=(theta, phi, threads)) as ex:
        results = sorted(ex.map(stack_one, maps))
    np.savez(
        args.out,
        params=np.array([r[1] for r in results]),
        param_names=PARAM_NAMES,
        profiles=np.array([r[2] for r in results]),
        errors=np.array([r[3] for r in results]),
        apertures_arcmin=APERTURES_ARCMIN,
    )
    print(f"Saved {len(results)} profiles -> {args.out}")


if __name__ == "__main__":
    main()
