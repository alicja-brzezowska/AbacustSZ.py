"""Ownership kernel, chord table and partner lists against brute force."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TSZ_LOS_GEOMETRY", "sphere")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import healpy as hp
import numpy as np
from scipy.integrate import quad

from abacustsz.paint.ownership import (
    _kept_kernel,
    build_cumulative_table,
    build_ownership_geometry,
    compute_kept_fractions,
    gnfw_shape,
)
from abacustsz.y_profile.closure_radius import truncation_radius_r200c

ALPHA, BETA0, GAMMA, Z = 1.457, 1.383, -0.143, 0.634
rng = np.random.default_rng(0)


def gnfw(r, logm):
    xc, a, b, g = gnfw_shape(ALPHA, BETA0, GAMMA, 10.0**logm, Z)
    xb = np.maximum(r, 1e-12) / xc
    return xb**g * (1.0 + xb**a) ** (-b)


def r200_of(logm):
    """Any smooth R200c(M) [Mpc/h] is fine for these tests (only ratios matter)."""
    return 0.20 * 10 ** ((logm - 12.0) / 3.0)


# ---------------------------------------------------------------------------
# 1. table vs quadrature
# ---------------------------------------------------------------------------
table = build_cumulative_table(ALPHA, BETA0, GAMMA, Z, 11.5, 15.0)
assert np.all(table.C[:, :, 0] == 0.0)
assert np.allclose(table.C[:, :, -1], 0.5)
assert np.all(np.diff(table.C, axis=2) >= -1e-12)
for logm, u, f in [(12.0, 0.1, 0.3), (13.5, 0.6, 0.8), (14.5, 0.0, 0.05), (14.5, 0.016, 0.05), (11.6, 0.9, 0.5)]:
    # compare exactly on the table's (u, logm) grid points, interpolating in f only
    iu = np.argmin(abs(table.u - u))
    im = np.argmin(abs(table.logm - logm))
    u, logm = float(table.u[iu]), float(table.logm[im])
    rt = float(truncation_radius_r200c(logm))
    xperp = u * rt
    L = rt * np.sqrt(1 - u * u)
    full = quad(lambda l: gnfw(np.sqrt(xperp**2 + l**2), logm), 0, L, limit=200)[0]
    part = quad(lambda l: gnfw(np.sqrt(xperp**2 + l**2), logm), 0, f * L, limit=200)[0]
    want = 0.5 * part / full
    got = np.interp(f, table.f, table.C[iu, im])
    assert abs(got - want) < 2e-3, (logm, u, f, got, want)
print("1. cumulative table: ok")


# ---------------------------------------------------------------------------
# helpers shared by 2-5
# ---------------------------------------------------------------------------
def make_geometry(xyz, logm, c_roche=1.0, n_shell=None):
    n_shell = len(xyz) if n_shell is None else n_shell
    r200 = r200_of(logm)
    rp = r200 * truncation_radius_r200c(logm)
    return build_ownership_geometry(
        xyz[:n_shell], r200[:n_shell], rp[:n_shell], logm[:n_shell],
        xyz[n_shell:], r200[n_shell:], rp[n_shell:], logm[n_shell:],
        c_roche=c_roche, workers=1,
    )


def kernel(geom, nvec, hidx):
    nvec = np.atleast_2d(nvec).astype(np.float64)
    out = np.empty(len(nvec), dtype=np.float32)
    _kept_kernel(
        np.ascontiguousarray(nvec[:, 0]), np.ascontiguousarray(nvec[:, 1]), np.ascontiguousarray(nvec[:, 2]),
        np.asarray(hidx, dtype=np.int64), geom.xyz, geom.r200, geom.r_paint, geom.logm,
        geom.partner_ptr, geom.partner_idx, geom.c_roche, table.u, table.logm, table.C, out,
    )
    return out.astype(np.float64)


def brute_force(geom, n, i, n_l=200_001):
    """Pointwise ownership along the ray, trapezoid in l with the exact gNFW."""
    ci = geom.xyz[i]
    ti = ci @ n
    bi2 = max(ci @ ci - ti * ti, 0.0)
    Li2 = geom.r_paint[i] ** 2 - bi2
    if Li2 <= 0:
        return 1.0
    Li = np.sqrt(Li2)
    l = np.linspace(-Li, Li, n_l)
    P = gnfw(np.sqrt(bi2 + l * l) / geom.r200[i], geom.logm[i])
    si2 = (bi2 + l * l) / geom.r200[i] ** 2
    owned = np.ones_like(l, dtype=bool)
    for j in range(len(geom.xyz)):
        if j == i:
            continue
        cj = geom.xyz[j]
        tj = cj @ n
        bj2 = max(cj @ cj - tj * tj, 0.0)
        lj = tj - ti
        inside = (bj2 + (l - lj) ** 2) < geom.r_paint[j] ** 2
        if not inside.any():
            continue
        if geom.logm[i] > geom.logm[j]:
            kappa = geom.c_roche
        elif geom.logm[i] < geom.logm[j]:
            kappa = 1.0 / geom.c_roche
        else:
            kappa = 1.0
        sj2 = (bj2 + (l - lj) ** 2) / geom.r200[j] ** 2
        owned &= ~inside | (si2 < kappa * sj2)
    return np.trapezoid(P * owned, l) / np.trapezoid(P, l)


def random_ray_through(geom, i, frac=0.95):
    """Unit vector hitting halo i's sphere at impact parameter <= frac * R_p."""
    c = geom.xyz[i]
    d = np.linalg.norm(c)
    n0 = c / d
    t1 = np.cross(n0, [0, 0, 1.0]); t1 /= np.linalg.norm(t1)
    t2 = np.cross(n0, t1)
    b = frac * geom.r_paint[i] * np.sqrt(rng.random())
    phi = rng.uniform(0, 2 * np.pi)
    p = c + b * (np.cos(phi) * t1 + np.sin(phi) * t2)
    return p / np.linalg.norm(p)


# ---------------------------------------------------------------------------
# 2. kernel vs brute force, random configurations
# ---------------------------------------------------------------------------
worst = 0.0
n_checked = 0
for c_roche in (1.0, 2.0):
    for cfg in range(12):
        n_h = rng.integers(2, 7)
        logm = rng.uniform(11.6, 14.6, n_h)
        logm[rng.integers(n_h)] = logm[0]  # force an exact mass tie now and then
        centre = np.array([0, 0, 1600.0]) + rng.normal(0, 3, 3)
        xyz = centre + rng.normal(0, 1.2, (n_h, 3))
        geom = make_geometry(xyz, logm, c_roche)
        for i in range(n_h):
            for _ in range(8):
                n = random_ray_through(geom, i)
                got = kernel(geom, n, [i])[0]
                want = brute_force(geom, n, i)
                worst = max(worst, abs(got - want))
                n_checked += 1
                assert abs(got - want) < 1.5e-2, (c_roche, cfg, i, got, want)
print(f"2. kernel vs brute force: ok ({n_checked} rays, worst |diff| = {worst:.2e})")


# ---------------------------------------------------------------------------
# 3. analytic cases
# ---------------------------------------------------------------------------
# (a) equal pair lined up along the LOS: symmetric kept fractions on the central ray
logm = np.array([13.0, 13.0])
rp = r200_of(13.0) * float(truncation_radius_r200c(13.0))
d = 0.6 * rp
xyz = np.array([[0, 0, 1600.0], [0, 0, 1600.0 + d]])
geom = make_geometry(xyz, logm)
k = kernel(geom, [[0, 0, 1.0], [0, 0, 1.0]], [0, 1])
assert abs(k[0] - k[1]) < 1e-6 and 0.5 < k[0] < 1.0, k
# boundary at l = d/2 from A: kept_A = 0.5 + C(d/(2 L_A))
want = 0.5 + np.interp(0.5 * d / rp, table.f, table.C[0, np.argmin(abs(table.logm - 13.0))])
assert abs(k[0] - want) < 3e-3, (k[0], want)

# (b) equal pair side by side at the same chi: the ray through A's centre is all A's
xyz = np.array([[0, 0, 1600.0], [d, 0, 1600.0]])
geom = make_geometry(xyz, logm)
nA = xyz[0] / np.linalg.norm(xyz[0])
k = kernel(geom, [nA, nA], [0, 1])
assert abs(k[0] - 1.0) < 1e-6 and abs(k[1]) < 1e-6, k

# (c) no partner -> 1 ; ray missing the sphere -> 1
xyz = np.array([[0, 0, 1600.0], [0, 0, 1700.0]])
geom = make_geometry(xyz, logm)
assert geom.n_pairs == 0 and kernel(geom, [[0, 0, 1.0]], [0])[0] == 1.0
xyz = np.array([[0, 0, 1600.0], [0, 0, 1600.0 + d]])
geom = make_geometry(xyz, logm)
far = np.array([3 * rp, 0, 1600.0]); far /= np.linalg.norm(far)
assert kernel(geom, [far], [0])[0] == 1.0

# (d) c_roche = 2, unequal pair: the heavier halo keeps more of the gap than with c = 1
logm = np.array([13.5, 12.5])
xyz = np.array([[0, 0, 1600.0], [0, 0, 1600.0 + 1.0]])
k1 = kernel(make_geometry(xyz, logm, 1.0), [[0, 0, 1.0]] * 2, [0, 1])
k2 = kernel(make_geometry(xyz, logm, 2.0), [[0, 0, 1.0]] * 2, [0, 1])
assert k2[0] >= k1[0] - 1e-9 and k2[1] <= k1[1] + 1e-9, (k1, k2)

# (e) exact duplicate (same position, same mass): painted once, by the lower index
logm = np.array([13.0, 13.0, 12.4])
xyz = np.array([[30.0, 20.0, 1600.0], [30.0, 20.0, 1600.0], [30.4, 20.0, 1600.3]])
geom = make_geometry(xyz, logm)
n = random_ray_through(geom, 0, 0.5)
k = kernel(geom, [n, n], [0, 1])
alone = kernel(make_geometry(xyz[[0, 2]], logm[[0, 2]]), [n], [0])[0]
assert abs(k[0] - alone) < 1e-6 and k[1] == 0.0, (k, alone)
print("3. analytic cases: ok")


# ---------------------------------------------------------------------------
# 4. partner lists vs brute force, with ghosts
# ---------------------------------------------------------------------------
n_s, n_g = 300, 80
logm = rng.uniform(11.6, 14.5, n_s + n_g)
xyz = np.array([0, 0, 1600.0]) + rng.uniform(-8, 8, (n_s + n_g, 3))
geom = make_geometry(xyz, logm, n_shell=n_s)
D = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)
ov = D < geom.r_paint[:, None] + geom.r_paint[None]
np.fill_diagonal(ov, False)
for i in range(n_s):
    got = np.sort(geom.partner_idx[geom.partner_ptr[i] : geom.partner_ptr[i + 1]])
    assert np.array_equal(got, np.flatnonzero(ov[i])), i
assert geom.n_ghost == n_g
print(f"4. partner lists: ok ({geom.n_pairs} pairs, {n_g} ghosts)")


# ---------------------------------------------------------------------------
# 5. wrapper: pair -> halo mapping with zero-count halos, against the kernel
# ---------------------------------------------------------------------------
nside = 512
n_h = 6
logm = rng.uniform(12.5, 14.0, n_h)
xyz = np.array([0, 0, 1600.0]) + rng.normal(0, 1.0, (n_h, 3))
geom = make_geometry(xyz, logm)
pix_lists = []
for i in range(n_h):
    if i == 2:  # a halo that contributes no pixels
        pix_lists.append(np.zeros(0, np.int64))
        continue
    ang = np.arctan(geom.r_paint[i] / np.linalg.norm(xyz[i]))
    pix_lists.append(hp.query_disc(nside, xyz[i] / np.linalg.norm(xyz[i]), ang, nest=True, inclusive=True))
counts = np.array([len(p) for p in pix_lists])
starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
pix = np.concatenate(pix_lists)
hidx = np.repeat(np.arange(n_h), counts)
got = compute_kept_fractions(pix, starts, counts, np.arange(n_h), geom, table, nside, chunk=97)
x, y, z = hp.pix2vec(nside, pix, nest=True)
want = kernel(geom, np.stack([x, y, z], 1), hidx)
assert np.allclose(got, want, atol=1e-6), np.abs(got - want).max()
assert (got < 1).any(), "expected some reduced pairs in this crowded configuration"
# batch with a permuted global index: results must follow the mapping, not the order
perm = rng.permutation(n_h)
inv = np.argsort(perm)
got_b = compute_kept_fractions(pix, starts, counts, perm, geom, table, nside, chunk=50)
want_b = kernel(geom, np.stack([x, y, z], 1), perm[hidx])
assert np.allclose(got_b, want_b, atol=1e-6)
print(f"5. wrapper: ok ({len(pix)} pairs, {np.mean(got < 1):.0%} reduced)")

print("\nall ownership tests passed")
