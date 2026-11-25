"""Nested-sampling fit of the semi-analytic 1h + A_2h 2h model to the ACT x DESI LRG profile.

The model is evaluated directly at every likelihood call (direct_model.py), so the covariance
is the ACT one only. The host-halo (M200c, z) distribution and the central/satellite split come
from the LRG mocks (AbacusHOD writes the Ncent centrals first, then the satellites).
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import asdf
import corner
import dynesty
import matplotlib

matplotlib.use("Agg")
import numpy as np
from astropy.table import Table
from dynesty.utils import resample_equal
from scipy.stats import chi2 as chi2_dist

from abacustsz.cosmology.model import H_ABACUS, MAX_M200C_MSUN, m200m_to_m200c, mdeltamean_to_m200m
from abacustsz.paint.abacus_loader import obtain_healcount_edges
from abacustsz.paths import HALO_CATALOGS_PATH, HEALCOUNTS_TOTAL_PATH, OUTPUT_PATH

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "forward_model"))
sys.path.insert(0, str(HERE.parent / "inference"))
from config import LRG_REDSHIFTS, STEPS, healcounts_file  # noqa: E402
from direct_model import DirectModel  # noqa: E402
from halo_model import mean_host_bias  # noqa: E402
from inference import gaussian_loglike, load_act  # noqa: E402

NAMES = ["alpha", "beta0", "gamma", "log10P0", "A2h"]
BOUNDS = np.array([[0.1, 2.0], [0.1, 6.0], [-1.0, -0.1], [-2.0, 2.0], [0.0, 12.0]])
CHI_STEPS = STEPS[7:18]  # LRGs between Step0617-0622 and Step0677-0682
N_M_BINS, N_Z_BINS, Z_RANGE = 150, 15, (0.4, 0.8)

_model = _y = _cov = _gamma = None


def load_lrg_hosts(lrg_dir, lrg_file):
    """log10 M200c [Msun] of each LRG's host, its snapshot redshift, and whether it is a satellite."""
    edges = [obtain_healcount_edges(HEALCOUNTS_TOTAL_PATH / healcounts_file(s)) for s in CHI_STEPS]
    chi_lo, chi_hi = min(e[0] for e in edges), max(e[1] for e in edges)
    logm, zs, sat = [], [], []
    for z in LRG_REDSHIFTS:
        t = Table.read(lrg_dir / f"z{z}" / "galaxies_rsd" / lrg_file, format="ascii.ecsv")
        chi = np.sqrt(np.asarray(t["x"]) ** 2 + np.asarray(t["y"]) ** 2 + np.asarray(t["z"]) ** 2)
        keep = (chi >= chi_lo) & (chi <= chi_hi)
        with asdf.open(HALO_CATALOGS_PATH / f"z{z}" / "lightcone_halo_info_000.asdf", lazy_load=True) as af:
            so_density = float(af["header"]["SODensityL1"])
        m200m = mdeltamean_to_m200m(np.asarray(t["mass"])[keep], so_density, float(z))
        mass = m200m_to_m200c(m200m, float(z))[0] / H_ABACUS
        cut = mass <= MAX_M200C_MSUN
        logm.append(np.log10(mass[cut]))
        zs.append(np.full(cut.sum(), float(z)))
        sat.append((np.arange(len(t)) >= int(t.meta["Ncent"]))[keep][cut])
    logm, zs, sat = np.concatenate(logm), np.concatenate(zs), np.concatenate(sat)
    print(f"{len(logm):,} LRGs, satellite fraction {sat.mean():.3f}")
    return logm, zs, sat


def histogram(logm, zs, weights=None):
    counts, m_edges, z_edges = np.histogram2d(
        logm, zs, bins=[N_M_BINS, N_Z_BINS], range=[[logm.min(), logm.max()], Z_RANGE], weights=weights
    )
    return counts, 0.5 * (m_edges[1:] + m_edges[:-1]), 0.5 * (z_edges[1:] + z_edges[:-1])


def full_theta(theta):
    return theta if _gamma is None else np.insert(theta, 2, _gamma)


def predict(theta):
    alpha, beta0, gamma, log10P0, a2h = full_theta(theta)
    y1h, y2h = _model(alpha, beta0, gamma, log10P0)
    return y1h + a2h * y2h


def log_likelihood(theta):
    return gaussian_loglike(predict(theta), _y, _cov)


def init_worker(model_args, y, cov, gamma):
    global _model, _y, _cov, _gamma
    _model, _y, _cov, _gamma = DirectModel(*model_args), y, cov, gamma


class Prior:
    def __init__(self, bounds):
        self.lo, self.width = bounds[:, 0], bounds[:, 1] - bounds[:, 0]

    def __call__(self, u):
        return self.lo + u * self.width


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freq", type=int, choices=[90, 150], default=150)
    parser.add_argument("--fix-gamma", type=float, help="e.g. -0.3")
    parser.add_argument("--a2h-max", type=float, default=12.0)
    parser.add_argument("--no-satellites", action="store_true", help="put satellites at their host's centre")
    parser.add_argument("--lrg-dir", type=Path, default=OUTPUT_PATH / "hod_mocks" / "lightcone_halos")
    parser.add_argument("--lrg-file", default="LRGs_thinned.dat")
    parser.add_argument("--nlive", type=int, default=1500)
    parser.add_argument("--dlogz", type=float, default=0.1)
    parser.add_argument("--nproc", type=int, default=mp.cpu_count())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="continue from <out>/dynesty.save")
    parser.add_argument("--out", type=Path, default=Path("results_direct"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    logm, zs, sat = load_lrg_hosts(args.lrg_dir, args.lrg_file)
    counts, m_centers, z_centers = histogram(logm, zs)
    counts_sat = None if args.no_satellites else histogram(logm, zs, weights=sat.astype(float))[0]
    model_args = (args.freq, counts, m_centers, z_centers, mean_host_bias(counts, m_centers, z_centers), counts_sat)

    y, cov = load_act(args.freq)
    bounds = BOUNDS.copy()
    bounds[-1, 1] = args.a2h_max
    names = NAMES
    if args.fix_gamma is not None:
        bounds, names = np.delete(bounds, 2, axis=0), [n for n in NAMES if n != "gamma"]
    init_worker(model_args, y, cov, args.fix_gamma)

    checkpoint = args.out / "dynesty.save"
    with mp.get_context("spawn").Pool(args.nproc, init_worker, (model_args, y, cov, args.fix_gamma)) as pool:
        if args.resume:
            sampler = dynesty.NestedSampler.restore(str(checkpoint), pool=pool)
            sampler.run_nested(resume=True, checkpoint_file=str(checkpoint))
        else:
            sampler = dynesty.NestedSampler(
                log_likelihood, Prior(bounds), len(names), nlive=args.nlive, bound="multi",
                sample="rslice", slices=10, rstate=np.random.default_rng(args.seed),
                pool=pool, queue_size=args.nproc,
            )
            sampler.run_nested(dlogz=args.dlogz, checkpoint_file=str(checkpoint), checkpoint_every=300)
    res = sampler.results

    samples = resample_equal(res.samples, np.exp(res.logwt - res.logz[-1]))
    theta_map = res.samples[np.argmax(res.logl)]
    alpha, beta0, gamma, log10P0, a2h = full_theta(theta_map)
    y1h, y2h = _model(alpha, beta0, gamma, log10P0)
    r = y - (y1h + a2h * y2h)
    chi2 = float(r @ np.linalg.solve(cov, r))
    dof = len(y) - len(names)

    for i, name in enumerate(names):
        lo, med, hi = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{name:8s} = {med:.3f} +{hi - med:.3f} -{med - lo:.3f}   (MAP {theta_map[i]:.3f})")
    print(f"chi2 = {chi2:.2f} for {dof} dof, p = {chi2_dist.sf(chi2, dof):.3f}")
    print(f"log Z = {res.logz[-1]:.3f} +- {res.logzerr[-1]:.3f}")

    np.savez(
        args.out / f"posterior_f{args.freq:03d}.npz",
        samples=samples, param_names=names, prior_bounds=bounds, theta_map=theta_map,
        gamma_fixed=np.nan if args.fix_gamma is None else args.fix_gamma, satellites=not args.no_satellites,
        y_act=y, cov_act=cov, y1h_map=y1h, y2h_map=a2h * y2h, chi2_map=chi2,
        logz=res.logz[-1], logzerr=res.logzerr[-1],
    )
    corner.corner(samples, labels=names, truths=theta_map, show_titles=True).savefig(
        args.out / f"corner_f{args.freq:03d}.png", dpi=150
    )


if __name__ == "__main__":
    main()
