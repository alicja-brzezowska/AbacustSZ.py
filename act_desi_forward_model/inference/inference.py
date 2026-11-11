"""Nested-sampling fit of the gNFW pressure profile to the ACT x DESI LRG tSZ profile.

Likelihood: Gaussian with covariance C_ACT + diag(sigma_GP^2).
Prior: uniform over the range of the Latin hypercube.
"""

import argparse
from pathlib import Path

import corner
import dynesty
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from dynesty.utils import resample_equal
from scipy.stats import chi2 as chi2_dist

from emulator import GPEmulator, load_training_set

# ACT DR5 f090/f150 maps stacked on DESI photometric LRGs (redshift bin 2) with the
# semi-cumulative ring-ring filter; Liu et al. 2025, PRD 112, 083561, arXiv:2502.08850.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ACT_PROFILES = DATA_DIR / "fig15.csv"
ACT_COVARIANCE = DATA_DIR / "fig15_cov.npz"


def load_act(freq):
    column = f"DESI_pz2_act_dr5_f{freq}_ringring2"
    df = pd.read_csv(ACT_PROFILES).dropna()
    y, err = df[column].values, df[f"{column}_err"].values
    shape = np.load(ACT_COVARIANCE)[column]
    sigma = np.sqrt(np.diag(shape))
    corr = np.clip(shape / np.outer(sigma, sigma), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return y, err[:, None] * corr * err[None, :]


def gaussian_loglike(mu, y, cov):
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return -1e30
    r = np.linalg.solve(L, mu - y)
    return -0.5 * (r @ r + 2.0 * np.log(np.diag(L)).sum() + len(y) * np.log(2.0 * np.pi))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_set", help="stacked LHC profiles (.npz from run_stacking.py)")
    parser.add_argument("--emulator", help="trained emulator .pkl (trained here if omitted)")
    parser.add_argument("--exclude-rows", type=int, nargs="*", default=[])
    parser.add_argument("--freq", type=int, choices=[90, 150], default=150)
    parser.add_argument("--nlive", type=int, default=4000)
    parser.add_argument("--dlogz", type=float, default=0.01)
    parser.add_argument("--slices", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_params, profiles, params, apertures, names = load_training_set(args.training_set, args.exclude_rows)
    if args.emulator:
        em = GPEmulator.load(args.emulator)
    else:
        em = GPEmulator(params, profiles, p0_index=names.index("log10P0"))
        em.save(out / "emulator.pkl")

    y, cov = load_act(args.freq)
    bounds = np.column_stack([all_params.min(0), all_params.max(0)])

    def prior_transform(u):
        return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])

    def log_likelihood(theta):
        mu, sig = em.predict(theta)
        return gaussian_loglike(mu[0], y, cov + np.diag(sig[0] ** 2))

    sampler = dynesty.NestedSampler(
        log_likelihood, prior_transform, len(names),
        nlive=args.nlive, bound="multi", sample="rslice", slices=args.slices,
        rstate=np.random.default_rng(args.seed),
    )
    sampler.run_nested(dlogz=args.dlogz)
    res = sampler.results

    samples = resample_equal(res.samples, np.exp(res.logwt - res.logz[-1]))
    theta_map = res.samples[np.argmax(res.logl)]
    mu, sig = em.predict(theta_map)
    r = y - mu[0]
    chi2 = float(r @ np.linalg.solve(cov + np.diag(sig[0] ** 2), r))
    dof = len(y) - len(names)

    for i, name in enumerate(names):
        lo, med, hi = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{name:8s} = {med:.3f} +{hi - med:.3f} -{med - lo:.3f}   (MAP {theta_map[i]:.3f})")
    print(f"chi2 = {chi2:.2f} for {dof} dof, p = {chi2_dist.sf(chi2, dof):.3f}")
    print(f"log Z = {res.logz[-1]:.3f} +- {res.logzerr[-1]:.3f}")

    np.savez(
        out / f"posterior_f{args.freq:03d}.npz",
        samples=samples, param_names=names, prior_bounds=bounds, theta_map=theta_map,
        apertures=apertures, y_act=y, cov_act=cov, logz=res.logz[-1], logzerr=res.logzerr[-1],
    )
    corner.corner(samples, labels=names, truths=theta_map, show_titles=True).savefig(
        out / f"corner_f{args.freq:03d}.png", dpi=150
    )


if __name__ == "__main__":
    main()
