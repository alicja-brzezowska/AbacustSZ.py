"""Ownership of overlapping painted halo volumes.

Each point is painted only by the halo closest to it in units of R200c. A
pixel-halo pair keeps the fraction of its line-of-sight pressure that the halo owns.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from time import perf_counter

import healpy as hp
import numpy as np
from loguru import logger
from numba import njit, prange
from scipy.spatial import cKDTree

from abacustsz.y_profile.closure_radius import truncation_radius_r200c

# ---------------------------------------------------------------------------
# Partner lists (once per shell)
# ---------------------------------------------------------------------------


@dataclass
class OwnershipGeometry:
    """Halo positions and radii (shell halos, then ghosts) and overlap partners per shell halo."""

    xyz: np.ndarray  # (N_all, 3) comoving Mpc/h, observer at origin
    r200: np.ndarray  # (N_all,) comoving Mpc/h
    r_paint: np.ndarray  # (N_all,) comoving Mpc/h
    logm: np.ndarray  # (N_all,) log10 M200c [Msun]
    n_shell: int
    partner_ptr: np.ndarray  # (n_shell + 1,) int64
    partner_idx: np.ndarray  # int64, indices into the N_all arrays
    c_roche: float = 1.0
    n_ghost: int = field(init=False)

    def __post_init__(self):
        self.n_ghost = len(self.xyz) - self.n_shell
        assert self.partner_ptr.shape == (self.n_shell + 1,)

    @property
    def n_pairs(self) -> int:
        return int(len(self.partner_idx))

    @property
    def halos_with_partners(self) -> int:
        return int(np.count_nonzero(np.diff(self.partner_ptr)))


def build_ownership_geometry(
    xyz_shell: np.ndarray,
    r200_shell: np.ndarray,
    r_paint_shell: np.ndarray,
    logm_shell: np.ndarray,
    xyz_ghost: np.ndarray | None,
    r200_ghost: np.ndarray | None,
    r_paint_ghost: np.ndarray | None,
    logm_ghost: np.ndarray | None,
    c_roche: float = 1.0,
    workers: int = 8,
    chunk: int = 200_000,
) -> OwnershipGeometry:
    """All pairs (shell halo i, any halo j) with |x_i - x_j| < R_p,i + R_p,j."""
    t0 = perf_counter()
    if xyz_ghost is None or len(xyz_ghost) == 0:
        xyz_ghost = np.zeros((0, 3))
        r200_ghost = np.zeros(0)
        r_paint_ghost = np.zeros(0)
        logm_ghost = np.zeros(0)
    xyz = np.ascontiguousarray(np.concatenate([xyz_shell, xyz_ghost]), dtype=np.float64)
    r200 = np.concatenate([r200_shell, r200_ghost]).astype(np.float64)
    r_paint = np.concatenate([r_paint_shell, r_paint_ghost]).astype(np.float64)
    logm = np.concatenate([logm_shell, logm_ghost]).astype(np.float64)
    n_shell = len(xyz_shell)
    r_max = float(r_paint.max())

    tree = cKDTree(xyz, balanced_tree=False, compact_nodes=False)
    counts = np.zeros(n_shell, dtype=np.int64)
    idx_chunks = []
    for start in range(0, n_shell, chunk):
        sl = np.arange(start, min(start + chunk, n_shell))
        lists = tree.query_ball_point(xyz[sl], r_paint[sl] + r_max, workers=workers, return_sorted=False)
        nq = np.fromiter(map(len, lists), dtype=np.int64, count=len(lists))
        j = np.fromiter(itertools.chain.from_iterable(lists), dtype=np.int64, count=int(nq.sum()))
        i = np.repeat(sl, nq)
        del lists
        d = np.linalg.norm(xyz[i] - xyz[j], axis=1)
        keep = (j != i) & (d < r_paint[i] + r_paint[j])
        i, j = i[keep], j[keep]
        order = np.argsort(i, kind="stable")  # CSR needs partners grouped by i
        i, j = i[order], j[order]
        counts[sl] = np.bincount(i - start, minlength=len(sl))
        idx_chunks.append(j)
    partner_idx = np.concatenate(idx_chunks) if idx_chunks else np.zeros(0, np.int64)
    partner_ptr = np.zeros(n_shell + 1, dtype=np.int64)
    partner_ptr[1:] = np.cumsum(counts)
    geom = OwnershipGeometry(xyz, r200, r_paint, logm, n_shell, partner_ptr, partner_idx, float(c_roche))
    logger.info(
        f"ownership geometry: {n_shell:,} shell halos + {geom.n_ghost:,} ghosts, "
        f"{geom.n_pairs:,} overlap pairs, {geom.halos_with_partners / n_shell:.1%} of halos "
        f"have >=1 partner, c_roche={c_roche:g} ({perf_counter() - t0:.0f}s)"
    )
    return geom


# ---------------------------------------------------------------------------
# Cumulative LOS table (once per gNFW shape and shell redshift)
# ---------------------------------------------------------------------------


@dataclass
class CumulativeLOSTable:
    """C[u, m, f]: fraction of the chord pressure between l = 0 and l = f L, at impact parameter u = b / R_p."""

    u: np.ndarray
    logm: np.ndarray
    f: np.ndarray
    C: np.ndarray  # (nu, nm, nf) float64, C-contiguous
    alpha: float
    beta0: float
    gamma: float
    redshift: float


def gnfw_shape(alpha: float, beta0: float, gamma: float, mass_msun, redshift: float):
    """(xc, alpha, beta, gamma) with the mass/redshift scalings of the pressure profile."""
    z1 = redshift + 1.0
    m = np.asarray(mass_msun, dtype=np.float64) / 1e14
    xc = 0.497 * m ** (-0.00865) * z1**0.731
    beta = beta0 * m**0.0393 * z1**0.415
    return xc, alpha, beta, gamma


def build_cumulative_table(
    alpha: float,
    beta0: float,
    gamma: float,
    redshift: float,
    logm_min: float,
    logm_max: float,
    n_u: int = 65,
    n_m: int = 49,
    n_f: int = 33,
    n_l: int = 2049,
) -> CumulativeLOSTable:
    """Cumulative gNFW chord integrals on a (u, log M, f) grid."""
    t0 = perf_counter()
    u = np.linspace(0.0, 1.0, n_u)
    logm = np.linspace(logm_min - 0.05, logm_max + 0.05, n_m)
    f = np.linspace(0.0, 1.0, n_f)
    v = np.linspace(0.0, 1.0, n_l)
    C = np.zeros((n_u, n_m, n_f))
    for im, lm in enumerate(logm):
        xc, a, b, g = gnfw_shape(alpha, beta0, gamma, 10.0**lm, redshift)
        rt = float(truncation_radius_r200c(lm))
        uu = u[:-1, None]  # u = 1 handled by copying the last computed column
        xperp = uu * rt
        L = rt * np.sqrt(1.0 - uu**2)
        l = L * v[None, :] ** 2
        r = np.sqrt(xperp**2 + l**2)
        xb = np.maximum(r, 1e-12) / xc
        P = xb**g * (1.0 + xb**a) ** (-b)
        integrand = P * 2.0 * L * v[None, :]  # dl = 2 L v dv
        cum = np.concatenate(
            [np.zeros((n_u - 1, 1)), np.cumsum(0.5 * (integrand[:, 1:] + integrand[:, :-1]) * np.diff(v)[None, :], axis=1)],
            axis=1,
        )
        cum *= 0.5 / cum[:, -1:]  # full chord (both halves) normalised to 1
        f_of_v = v**2
        for iu in range(n_u - 1):
            C[iu, im] = np.interp(f, f_of_v, cum[iu])
        C[-1, im] = C[-2, im]
    logger.info(
        f"cumulative LOS table: alpha={alpha:g} beta0={beta0:g} gamma={gamma:g} z={redshift:.3f}, "
        f"grid {n_u}x{n_m}x{n_f} ({perf_counter() - t0:.1f}s)"
    )
    return CumulativeLOSTable(u, logm, f, np.ascontiguousarray(C), alpha, beta0, gamma, redshift)


_STEM_RE = re.compile(r"alpha=(?P<alpha>[^_]+)_beta0=(?P<beta0>[^_]+)_gamma=(?P<gamma>[^_]+)")


def parse_gnfw_stem(stem: str) -> tuple[float, float, float]:
    """(alpha, beta0, gamma) from an interpolator stem."""
    m = _STEM_RE.search(stem)
    if m is None:
        raise ValueError(
            f"cannot read alpha/beta0/gamma from interpolator stem {stem!r}; "
            "ownership needs the alpha=..._beta0=..._gamma=... naming"
        )
    return float(m["alpha"]), float(m["beta0"]), float(m["gamma"])


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@njit(cache=True)
def _profile_in_f(tu, tm, tC, u, lm, out):
    """C(f) at (u, log M), bilinear."""
    nu = tu.shape[0]
    nm = tm.shape[0]
    nf = tC.shape[2]
    pu = (u - tu[0]) / (tu[1] - tu[0])
    pm = (lm - tm[0]) / (tm[1] - tm[0])
    if pu < 0.0:
        pu = 0.0
    if pu > nu - 1:
        pu = nu - 1.0
    if pm < 0.0:
        pm = 0.0
    if pm > nm - 1:
        pm = nm - 1.0
    iu = int(pu)
    im = int(pm)
    if iu >= nu - 1:
        iu = nu - 2
    if im >= nm - 1:
        im = nm - 2
    wu = pu - iu
    wm = pm - im
    for k in range(nf):
        out[k] = (
            (1.0 - wu) * (1.0 - wm) * tC[iu, im, k]
            + wu * (1.0 - wm) * tC[iu + 1, im, k]
            + (1.0 - wu) * wm * tC[iu, im + 1, k]
            + wu * wm * tC[iu + 1, im + 1, k]
        )


@njit(cache=True)
def _C_of_f(prof, f):
    """C(f) for f in [-1, 1] (odd in f)."""
    nf = prof.shape[0]
    s = 1.0
    a = f
    if f < 0.0:
        s = -1.0
        a = -f
    if a >= 1.0:
        return s * prof[nf - 1]
    pos = a * (nf - 1)
    k = int(pos)
    w = pos - k
    return s * (prof[k] * (1.0 - w) + prof[k + 1] * w)


@njit(parallel=True, cache=True)
def _kept_kernel(nx, ny, nz, hidx, xyz, r200, rp, logm, pptr, pidx, c_roche, tu, tm, tC, out):
    """Kept fraction per (pixel ray, halo) pair."""
    n = hidx.shape[0]
    nf = tC.shape[2]
    for p in prange(n):
        i = hidx[p]
        p0 = pptr[i]
        p1 = pptr[i + 1]
        if p1 == p0:
            out[p] = 1.0
            continue
        ax = nx[p]
        ay = ny[p]
        az = nz[p]
        cix = xyz[i, 0]
        ciy = xyz[i, 1]
        ciz = xyz[i, 2]
        ti = cix * ax + ciy * ay + ciz * az
        bi2 = cix * cix + ciy * ciy + ciz * ciz - ti * ti
        if bi2 < 0.0:
            bi2 = 0.0
        rpi = rp[i]
        Li2 = rpi * rpi - bi2
        if Li2 <= 0.0:
            out[p] = 1.0
            continue
        Li = np.sqrt(Li2)
        Ri2 = r200[i] * r200[i]
        lmi = logm[i]
        npart = p1 - p0
        lj = np.empty(npart)
        Lj = np.empty(npart)
        bj2 = np.empty(npart)
        Rj2 = np.empty(npart)
        kap = np.empty(npart)
        jid = np.empty(npart, dtype=np.int64)
        breaks = np.empty(2 + 4 * npart)
        nb = 0
        breaks[nb] = -Li
        nb += 1
        breaks[nb] = Li
        nb += 1
        nint = 0
        for q in range(p0, p1):
            j = pidx[q]
            cjx = xyz[j, 0]
            cjy = xyz[j, 1]
            cjz = xyz[j, 2]
            tj = cjx * ax + cjy * ay + cjz * az
            b2 = cjx * cjx + cjy * cjy + cjz * cjz - tj * tj
            if b2 < 0.0:
                b2 = 0.0
            rpj = rp[j]
            if b2 >= rpj * rpj:
                continue  # ray misses j's sphere
            L = np.sqrt(rpj * rpj - b2)
            l0 = tj - ti
            if l0 + L <= -Li or l0 - L >= Li:
                continue  # j's chord does not touch i's chord
            if lmi > logm[j]:
                kappa = c_roche
            elif lmi < logm[j]:
                kappa = 1.0 / c_roche
            else:
                kappa = 1.0
            lj[nint] = l0
            Lj[nint] = L
            bj2[nint] = b2
            R2 = r200[j] * r200[j]
            Rj2[nint] = R2
            kap[nint] = kappa
            jid[nint] = j
            nint += 1
            e1 = l0 - L
            e2 = l0 + L
            if e1 > -Li and e1 < Li:
                breaks[nb] = e1
                nb += 1
            if e2 > -Li and e2 < Li:
                breaks[nb] = e2
                nb += 1
            # ownership boundary: s_i^2 - kappa s_j^2 = 0  (quadratic in l)
            A = 1.0 / Ri2 - kappa / R2
            B = 2.0 * kappa * l0 / R2
            Cc = bi2 / Ri2 - kappa * (b2 + l0 * l0) / R2
            if abs(A) < 1e-14:
                if abs(B) > 1e-14:
                    rt = -Cc / B
                    if rt > -Li and rt < Li:
                        breaks[nb] = rt
                        nb += 1
            else:
                disc = B * B - 4.0 * A * Cc
                if disc >= 0.0:
                    sq = np.sqrt(disc)
                    r1 = (-B - sq) / (2.0 * A)
                    r2 = (-B + sq) / (2.0 * A)
                    if r1 > -Li and r1 < Li:
                        breaks[nb] = r1
                        nb += 1
                    if r2 > -Li and r2 < Li:
                        breaks[nb] = r2
                        nb += 1
        if nint == 0:
            out[p] = 1.0
            continue
        bs = np.sort(breaks[:nb])
        prof = np.empty(nf)
        _profile_in_f(tu, tm, tC, np.sqrt(bi2) / rpi, lmi, prof)
        kept = 0.0
        for k in range(nb - 1):
            lo = bs[k]
            hi = bs[k + 1]
            if hi - lo <= 0.0:
                continue
            m = 0.5 * (lo + hi)
            si2 = (bi2 + m * m) / Ri2
            owned = True
            for q in range(nint):
                dm = m - lj[q]
                if dm < 0.0:
                    dm = -dm
                if dm < Lj[q]:  # inside j's sphere
                    sj2 = (bj2[q] + (m - lj[q]) * (m - lj[q])) / Rj2[q]
                    ks = kap[q] * sj2
                    # exact tie: the lower index owns
                    if si2 > ks or (si2 == ks and jid[q] < i):
                        owned = False
                        break
            if owned:
                kept += _C_of_f(prof, hi / Li) - _C_of_f(prof, lo / Li)
        if kept < 0.0:
            kept = 0.0
        if kept > 1.0:
            kept = 1.0
        out[p] = kept


def compute_kept_fractions(
    pix: np.ndarray,
    halo_starts: np.ndarray,
    halo_counts: np.ndarray,
    global_halo_idx: np.ndarray,
    geometry: OwnershipGeometry,
    table: CumulativeLOSTable,
    nside: int,
    nest: bool = True,
    chunk: int = 20_000_000,
) -> np.ndarray:
    """Kept fraction per (pixel, halo) pair; global_halo_idx maps batch halos into geometry."""
    t0 = perf_counter()
    n = len(pix)
    out = np.empty(n, dtype=np.float32)
    starts = np.asarray(halo_starts, dtype=np.int64)
    gidx = np.asarray(global_halo_idx, dtype=np.int64)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        local = np.searchsorted(starts, np.arange(a, b, dtype=np.int64), side="right") - 1
        hidx = gidx[local]
        x, y, z = hp.pix2vec(nside, pix[a:b], nest=nest)
        buf = np.empty(b - a, dtype=np.float32)
        _kept_kernel(
            np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(z), hidx,
            geometry.xyz, geometry.r200, geometry.r_paint, geometry.logm,
            geometry.partner_ptr, geometry.partner_idx, geometry.c_roche,
            table.u, table.logm, table.C, buf,
        )
        out[a:b] = buf
    affected = out < 0.999
    logger.info(
        f"ownership kept fractions: {n:,} pairs in {perf_counter() - t0:.0f}s; "
        f"{affected.mean():.1%} of pairs reduced, mean kept over those "
        f"{out[affected].mean() if affected.any() else 1.0:.3f}"
    )
    return out
