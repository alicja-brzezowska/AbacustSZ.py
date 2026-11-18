"""Halo-model ingredients of the semi-analytic tSZ profile.

gNFW pressure profile of Battaglia et al. (2012) with the same (M, z) scalings as the
painted maps (abacustsz.y_profile), truncated in a sphere at R_trunc(M) R200c
(abacustsz.y_profile.closure_radius). Halo mass function, halo bias and linear power
spectrum from CCL (Tinker et al. 2010; BBKS transfer function). Masses are M200c in Msun,
distances physical Mpc, angles radians.
"""

import sys
from pathlib import Path

import numpy as np
import pyccl as ccl
from scipy.special import i0e
from wcosmo import comoving_distance, efunc
from wcosmo.utils import disable_units

import abacustsz.constants as const
from abacustsz.cosmology.model import (
    H_ABACUS, N_S_ABACUS, OMEGA_B_ABACUS, OMEGA_C_ABACUS, OMEGA_M_ABACUS, SIGMA8_ABACUS,
)
from abacustsz.y_profile.closure_radius import truncation_radius_r200c
from abacustsz.y_profile.y_profile_jax import get_gnfw_params

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forward_model"))
from filters import APERTURES_ARCMIN, BEAM_FWHM_ARCMIN, THETA_0_ARCMIN  # noqa: E402

disable_units()

DELTA = 200
F_B = OMEGA_B_ABACUS / OMEGA_M_ABACUS
H0 = 100.0 * H_ABACUS
H0_CGS = H0 * 1e3 / 3.085677581491367e22
P_E_OVER_P_TH = 0.5176  # electron / total thermal pressure (Battaglia et al. 2012)
ARCMIN = np.deg2rad(1.0 / 60.0)
G_SI, G_CGS, M_SUN = float(const.G_SI.value), float(const.G_CGS), float(const.M_SUN)
P_E_FACTOR = float(const.P_E_FACTOR)
G_CM3_TO_KG_M3, G_CM3_TO_MSUN_MPC3 = float(const.G_CM3_TO_KG_M3), float(const.G_CM3_TO_MSUN_MPC3)

COSMO = ccl.Cosmology(
    Omega_c=OMEGA_C_ABACUS, Omega_b=OMEGA_B_ABACUS, h=H_ABACUS,
    sigma8=SIGMA8_ABACUS, n_s=N_S_ABACUS, transfer_function="bbks",
)
_MASS_FUNC = ccl.halos.MassFuncTinker10()
_HALO_BIAS = ccl.halos.HaloBiasTinker10()


def halo_mass_function(mass, z):
    return _MASS_FUNC(COSMO, mass, 1.0 / (1.0 + z))


def halo_bias(mass, z):
    return _HALO_BIAS(COSMO, mass, 1.0 / (1.0 + z))


def linear_power(k, z):
    return ccl.linear_matter_power(COSMO, k, 1.0 / (1.0 + z))


def angular_diameter_distance(z):
    return np.array([float(comoving_distance(zz, H0, OMEGA_M_ABACUS)) for zz in np.atleast_1d(z)]) / (
        1.0 + np.atleast_1d(z)
    )


def critical_density(z):
    """rho_crit(z) in g/cm^3."""
    z = np.atleast_1d(z)
    rho0 = 3.0 * H0_CGS**2 / (8.0 * np.pi * G_CGS)
    return rho0 * np.array([float(efunc(zz, OMEGA_M_ABACUS)) for zz in z]) ** 2


def r200c(mass, z):
    rho = critical_density(z) * G_CM3_TO_MSUN_MPC3
    return (3.0 * mass / (4.0 * np.pi * DELTA * rho)) ** (1.0 / 3.0)


def gnfw_scalings(mass, z):
    """x_c, beta / beta0 and P0 / 10^log10_P0 at (M, z); none depends on the free parameters."""
    xc, _, beta_scale, _, p0_scale = get_gnfw_params(1.0, 1.0, 0.0, mass, z, 0.0)
    return np.asarray(xc), np.asarray(beta_scale), np.asarray(p0_scale)


def p200_no_radius(mass, z):
    """P200 R200 = G M Delta rho_crit f_b / 2 (SI); P200 itself is this / R200."""
    rho = critical_density(z) * G_CM3_TO_KG_M3
    return G_SI * mass * M_SUN * DELTA * rho * F_B / 2.0


def truncation_radius(mass):
    """Truncation radius in units of R200c."""
    return np.asarray(truncation_radius_r200c(np.log10(mass)), dtype=float)


def mean_host_bias(counts, m_centers, z_centers):
    i, j = np.nonzero(counts)
    b = np.array([halo_bias(10.0**m_centers[a], z_centers[c]) for a, c in zip(i, j)])
    return float((counts[i, j] * b).sum() / counts[i, j].sum())


def trapz_weights(x):
    w = np.empty(len(x))
    w[0], w[-1] = (x[1] - x[0]) / 2, (x[-1] - x[-2]) / 2
    w[1:-1] = (x[2:] - x[:-2]) / 2
    return w


def sinc(x):
    return np.where(x > 1e-8, np.sin(x) / np.where(x > 1e-8, x, 1.0), 1.0)


class Apertures:
    """Beam convolution and ring-ring filter as matrices on a shared theta grid.

    The grid starts at 0.01 arcmin, runs 8 beam sigmas past the outermost annulus and
    contains theta_0, theta_d and theta_out as nodes, so the annulus edges are exact.
    """

    def __init__(self, freq, apertures_arcmin=APERTURES_ARCMIN, n_theta=300, pad_sigma=8.0):
        self.apertures_arcmin = np.asarray(apertures_arcmin, dtype=float)
        sigma = BEAM_FWHM_ARCMIN[f"f{freq:03d}"] * ARCMIN / np.sqrt(8.0 * np.log(2.0))
        theta_0 = THETA_0_ARCMIN * ARCMIN
        theta_d = self.apertures_arcmin * ARCMIN
        theta_out = np.sqrt(2.0 * theta_d**2 - theta_0**2)
        base = np.geomspace(0.01 * ARCMIN, theta_out[-1] + pad_sigma * sigma, n_theta)
        theta = np.unique(np.concatenate([base, [theta_0], theta_d, theta_out]))

        # azimuthally averaged 2D Gaussian beam: K @ y(theta) = (B * y)(theta)
        ti, tj = theta[:, None], theta[None, :]
        beam = np.exp(-((ti - tj) ** 2) / (2 * sigma**2)) * i0e(ti * tj / sigma**2) * tj / sigma**2
        beam *= trapz_weights(theta)[None, :]

        def annulus_mean(lo, hi):
            idx = np.nonzero((theta >= lo) & (theta <= hi))[0]
            w = np.zeros(len(theta))
            tw = trapz_weights(theta[idx]) * theta[idx]
            w[idx] = tw / tw.sum()
            return w

        ring_ring = np.array([annulus_mean(theta_0, d) - annulus_mean(d, o) for d, o in zip(theta_d, theta_out)])
        ring_area = np.pi * (self.apertures_arcmin**2 - THETA_0_ARCMIN**2)

        self.theta = theta
        self.filter = ring_area[:, None] * (ring_ring @ beam)  # (n_ap, n_theta): y(theta) -> Y [arcmin^2]
