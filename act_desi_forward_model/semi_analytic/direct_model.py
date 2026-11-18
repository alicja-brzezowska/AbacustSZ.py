"""Semi-analytic stacked tSZ profile of the LRGs, evaluated directly (no emulator).

    Y(theta_d) = Y_1h + A_2h Y_2h

Everything that does not depend on (alpha, beta0, gamma, log10_P0) is precomputed, so a call
costs ~0.1 s.

1-halo: each LRG sees the projected gNFW profile of its host. The LOS integral uses fixed
Gauss-Legendre nodes in u with l = s sinh(u), s = theta / theta_200, which keeps the integrand
smooth through the gamma < 0 cusp. LRGs are binned in (M200c, z) and each bin is weighted by
its number of LRGs.

Satellites: a satellite sits at a random point of an NFW distribution (truncated at R200c)
around its host's centre. Projection commutes with convolution, so its profile is y_cen(theta)
convolved in 2D with the projected NFW. Per (M, z) bin that is a fixed linear map of y_cen,
folded into the bin's aperture weights.

2-halo: P_2h(k) = b_LRG P_lin(k) int dM n(M) b(M) u_P(k|M), Fourier-transformed to xi(r) and
projected. The chain is linear in the pressure profile P(r|M, z), so it is precomputed as one
tensor H[(z, M), aperture, r].
"""

import numpy as np
import jax
import jax.numpy as jnp

from abacustsz.cosmology.model import C_MODEL, H_ABACUS, MAX_M200M_H, concentration_200c
from halo_model import (
    P_E_FACTOR, P_E_OVER_P_TH, Apertures, angular_diameter_distance, gnfw_scalings, halo_bias,
    halo_mass_function, linear_power, p200_no_radius, r200c, sinc, trapz_weights, truncation_radius,
)

jax.config.update("jax_enable_x64", True)

N_LOS = 64       # Gauss-Legendre nodes along the line of sight (1-halo)
N_SAT_R = 48     # nodes in the satellite offset, theta' / theta_200 in [0, 1]
N_SAT_PHI = 48   # nodes in the angle between theta and theta'

LOGM_MIN_2H = 11.5
LOGM_MAX_2H = np.log10(MAX_M200M_H / H_ABACUS)
K_2H = np.geomspace(1e-3, 10.0, 150)      # 1/Mpc
R_2H = np.geomspace(0.1, 700.0, 300)      # Mpc
CHI_2H = np.geomspace(1e-4, 600.0, 600)   # LOS, Mpc


@jax.jit
def _gnfw_shape(log_xbar, beta_scale, alpha, beta0, gamma):
    return jnp.exp(gamma * log_xbar - beta0 * beta_scale * jnp.log1p(jnp.exp(alpha * log_xbar)))


@jax.jit
def _y1h(log_xbar, quad_w, beta_scale, amp, G, alpha, beta0, gamma, log10_P0):
    f = _gnfw_shape(log_xbar, beta_scale[:, None, None], alpha, beta0, gamma)
    y = jnp.maximum(0.0, 10.0**log10_P0 * amp[:, None] * jnp.sum(quad_w * f, axis=-1))
    return jnp.einsum("bt,bta->a", y, G)


@jax.jit
def _y2h(H, log_xbar, beta_scale, p200, alpha, beta0, gamma, log10_P0):
    P = 10.0**log10_P0 * p200[:, None] * _gnfw_shape(log_xbar, beta_scale[:, None], alpha, beta0, gamma)
    return jnp.einsum("bar,br->a", H, P)


def projected_nfw(c, n_r=N_SAT_R, n_los=64):
    """Nodes s = R / R200c and weights (summing to 1) of the projected, R200c-truncated NFW."""
    t, w = np.polynomial.legendre.leggauss(n_r)
    s, ws = 0.5 * (t + 1.0), 0.5 * w
    tl, wl = np.polynomial.legendre.leggauss(n_los)
    umax = np.arcsinh(np.sqrt(1.0 - s**2) / s)
    r = s[:, None] * np.cosh(0.5 * umax[:, None] * (tl + 1.0))
    sigma = np.sum(umax[:, None] * wl * r / (c * r * (1.0 + c * r) ** 2), axis=1)
    a = 2.0 * np.pi * s * sigma * ws
    return s, a / a.sum()


def satellite_matrix(theta_out, theta_in, theta_200, c, n_r=N_SAT_R, n_phi=N_SAT_PHI):
    """C with C @ y_cen(theta_in) = y_sat(theta_out); y_cen interpolated linearly in log theta."""
    s, a = projected_nfw(c, n_r)
    tp, wp = np.polynomial.legendre.leggauss(n_phi)
    cos_phi, w_phi = np.cos(0.5 * np.pi * (tp + 1.0)), 0.5 * wp
    t, tq = theta_out[:, None, None], (s * theta_200)[None, :, None]
    d = np.sqrt(np.maximum(t**2 + tq**2 - 2.0 * t * tq * cos_phi, 0.0))
    ld, lg = np.log(np.maximum(d, theta_in[0])), np.log(theta_in)
    i = np.clip(np.searchsorted(lg, ld, side="right") - 1, 0, len(lg) - 2)
    f = np.clip((ld - lg[i]) / (lg[i + 1] - lg[i]), 0.0, 1.0)
    w = np.broadcast_to(a[None, :, None] * w_phi, ld.shape)
    row = np.broadcast_to(np.arange(len(theta_out))[:, None, None], ld.shape) * len(theta_in)
    n = len(theta_out) * len(theta_in)
    C = np.bincount((row + i).ravel(), (w * (1 - f)).ravel(), n) + np.bincount((row + i + 1).ravel(), (w * f).ravel(), n)
    return C.reshape(len(theta_out), len(theta_in))


class OneHalo:
    """Y_1h from the (M, z) histograms of all LRGs (counts) and of the satellites (counts_sat)."""

    def __init__(self, ap, counts, m_centers, z_centers, counts_sat=None):
        im, iz = np.nonzero(counts)
        w = counts[im, iz] / counts[im, iz].sum()
        w_sat = np.zeros_like(w) if counts_sat is None else counts_sat[im, iz] / counts[im, iz].sum()
        mass, z = 10.0 ** m_centers[im], z_centers[iz]
        theta_200 = np.arctan(r200c(mass, z) / angular_diameter_distance(z))
        xc, beta_scale, p0_scale = gnfw_scalings(mass, z)
        rt = truncation_radius(mass)

        # satellites need y_cen out to theta_200 beyond the grid edge
        theta = ap.theta
        sat = np.nonzero(w_sat > 0)[0]
        if len(sat):
            dlog = np.log(theta[-1] / theta[-2])
            n_ext = int(np.ceil(np.log(1.02 * (theta[-1] + theta_200[sat].max()) / theta[-1]) / dlog))
            theta = np.concatenate([theta, theta[-1] * np.exp(dlog * np.arange(1, n_ext + 1))])
        n = len(ap.theta)
        filt = np.zeros((len(theta), len(ap.apertures_arcmin)))
        filt[:n] = ap.filter.T

        G = w[:, None, None] * filt[None]
        for b in sat:
            c = float(concentration_200c(mass[b] * H_ABACUS, z[b], c_model=C_MODEL))
            C = satellite_matrix(ap.theta, theta, theta_200[b], c)
            C[:, :n] -= np.eye(n)
            G[b] += w_sat[b] * (C.T @ ap.filter.T)

        # sphere of radius rt: l in [0, sqrt(rt^2 - s^2)], l = s sinh(u)
        s = theta[None, :] / theta_200[:, None]
        umax = np.arcsinh(np.sqrt(np.maximum(0.0, rt[:, None] ** 2 - s**2)) / s)
        t, wl = np.polynomial.legendre.leggauss(N_LOS)
        s_cosh = s[..., None] * np.cosh(0.5 * umax[..., None] * (t + 1.0))
        quad_w = 0.5 * umax[..., None] * wl * s_cosh
        amp = P_E_OVER_P_TH * p200_no_radius(mass, z) * 2.0 * P_E_FACTOR * p0_scale

        self.f_sat = float(w_sat.sum())
        self._args = [jnp.asarray(x) for x in (np.log(s_cosh / xc[:, None, None]), quad_w, beta_scale, amp, G)]

    def __call__(self, alpha, beta0, gamma, log10_P0):
        return np.asarray(_y1h(*self._args, alpha, beta0, gamma, log10_P0))


class TwoHalo:
    """Y_2h for LRGs with mean host bias b_LRG and redshift distribution n_z."""

    def __init__(self, ap, n_z, z_centers, b_lrg, nm=80, nr=512):
        k, r = K_2H, R_2H
        xi_of_P = trapz_weights(k)[None, :] * k**2 * sinc(k[None, :] * r[:, None]) / (2.0 * np.pi**2)
        w_chi = 2.0 * trapz_weights(CHI_2H)

        logm = np.linspace(LOGM_MIN_2H, LOGM_MAX_2H, nm)
        masses = 10.0**logm
        dlnm = np.log(10.0) * (logm[1] - logm[0])
        rt = truncation_radius(masses)

        H, log_xbar, beta_scale, p200 = [], [], [], []
        for j, z in enumerate(z_centers):
            if n_z[j] <= 0:
                continue
            dA = angular_diameter_distance(z)[0]
            proj = np.zeros((len(ap.theta), len(r)))
            for i, th in enumerate(ap.theta):
                rr = np.sqrt(CHI_2H**2 + (dA * th) ** 2)
                idx = np.clip(np.searchsorted(r, rr, side="right") - 1, 0, len(r) - 2)
                f = np.clip((rr - r[idx]) / (r[idx + 1] - r[idx]), 0.0, 1.0)
                np.add.at(proj[i], idx, w_chi * (1 - f))
                np.add.at(proj[i], idx + 1, w_chi * f)
            T = (n_z[j] / n_z.sum()) * P_E_OVER_P_TH * P_E_FACTOR * (ap.filter @ proj @ xi_of_P)
            T = T * (b_lrg * linear_power(k, z))[None, :]

            nb = np.array([halo_mass_function(m, z) * halo_bias(m, z) for m in masses]) * dlnm
            zz = np.full(nm, z)
            R200 = r200c(masses, zz)
            xc, bsc, p0_scale = gnfw_scalings(masses, zz)
            for m in range(nm):
                r_m = np.geomspace(1e-5, rt[m] * R200[m], nr)
                u = 4.0 * np.pi * r_m**2 * trapz_weights(r_m) * sinc(k[:, None] * r_m[None, :])
                H.append(nb[m] * (T @ u))
                log_xbar.append(np.log(r_m / (R200[m] * xc[m])))
                beta_scale.append(bsc[m])
                p200.append(p200_no_radius(masses[m], z)[0] / R200[m] * p0_scale[m])

        self._args = [jnp.asarray(np.array(x)) for x in (H, log_xbar, beta_scale, p200)]

    def __call__(self, alpha, beta0, gamma, log10_P0):
        return np.asarray(_y2h(*self._args, alpha, beta0, gamma, log10_P0))


class DirectModel:
    """(Y_1h, Y_2h) at the ring-ring apertures for gNFW parameters (alpha, beta0, gamma, log10_P0)."""

    def __init__(self, freq, counts, m_centers, z_centers, b_lrg, counts_sat=None):
        ap = Apertures(freq)
        self.one_halo = OneHalo(ap, counts, m_centers, z_centers, counts_sat)
        self.two_halo = TwoHalo(ap, counts.sum(axis=0), z_centers, b_lrg)

    def __call__(self, alpha, beta0, gamma, log10_P0):
        p = tuple(float(v) for v in (alpha, beta0, gamma, log10_P0))
        return self.one_halo(*p), self.two_halo(*p)
