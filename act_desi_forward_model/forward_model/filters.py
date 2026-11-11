"""ACT beam and the semi-cumulative ring-ring aperture filter.

The filter follows Liu et al. (2025, arXiv:2502.08850): for each aperture
theta_d, mean(y in [theta_0, theta_d]) - mean(y in [theta_d, theta_out]) with
theta_0 = 1 arcmin and theta_out = sqrt(2 theta_d^2 - theta_0^2).
"""

import os
from concurrent.futures import ThreadPoolExecutor

import asdf
import healpy as hp
import numpy as np

APERTURES_ARCMIN = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
THETA_0_ARCMIN = 1.0
BEAM_FWHM_ARCMIN = {"f090": 2.1, "f150": 1.3}  # ACT DR5 (Hilton et al. 2021)
ELL_MAX = 24000


def load_ymap(path):
    """y-map in NESTED ordering, from a painted .asdf or a beam-convolved .npy."""
    if str(path).endswith(".npy"):
        return np.load(path)
    with asdf.open(path) as f:
        return np.array(f["data"]["y_map"])


def convolve_beam(ymap, fwhm_arcmin):
    nside = hp.get_nside(ymap)
    ell = np.arange(ELL_MAX + 1)
    sigma = np.deg2rad(fwhm_arcmin / 60.0) / np.sqrt(8.0 * np.log(2.0))
    beam = np.exp(-0.5 * ell * (ell + 1) * sigma**2)
    alm = hp.map2alm(hp.reorder(ymap, n2r=True), lmax=ELL_MAX, iter=3)
    return hp.reorder(hp.alm2map(hp.almxfl(alm, beam), nside), r2n=True)


def ring_ring_filter(ymap, theta, phi, apertures_arcmin=APERTURES_ARCMIN, n_workers=None, chunk=50_000):
    """Filtered value for every galaxy and aperture, shape (N_gal, N_ap)."""
    nside = hp.get_nside(ymap)
    theta_0 = np.deg2rad(THETA_0_ARCMIN / 60.0)
    theta_d = np.deg2rad(apertures_arcmin / 60.0)
    theta_out = np.sqrt(2 * theta_d**2 - theta_0**2)

    def one_galaxy(i):
        vec = hp.ang2vec(theta[i], phi[i])
        ipix = hp.query_disc(nside, vec, theta_out[-1])
        row = np.zeros(len(theta_d))
        if len(ipix) == 0:
            return row
        ang = np.arccos(np.clip(np.array(hp.pix2vec(nside, ipix)).T @ vec, -1.0, 1.0))
        order = np.argsort(ang)
        ang = ang[order]
        cumsum = np.cumsum(ymap[hp.ring2nest(nside, ipix)[order]].astype(np.float64))

        i0 = np.searchsorted(ang, theta_0, side="right")
        start = cumsum[i0 - 1] if i0 > 0 else 0.0
        for j in range(len(theta_d)):
            i_d = np.searchsorted(ang, theta_d[j], side="right")
            i_out = np.searchsorted(ang, theta_out[j], side="right")
            if i_d == i0 or i_out == i_d:
                continue
            inner = (cumsum[i_d - 1] - start) / (i_d - i0)
            outer = (cumsum[i_out - 1] - cumsum[i_d - 1]) / (i_out - i_d)
            row[j] = inner - outer
        return row

    values = np.zeros((len(theta), len(theta_d)))
    with ThreadPoolExecutor(n_workers or os.cpu_count()) as ex:
        for lo in range(0, len(theta), chunk):
            hi = min(lo + chunk, len(theta))
            values[lo:hi] = list(ex.map(one_galaxy, range(lo, hi)))
    return values
