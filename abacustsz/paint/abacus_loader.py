import os
from dataclasses import dataclass
from pathlib import Path

import asdf
import healpy as hp
import numpy as np
from loguru import logger
from abacusnbody.data.compaso_halo_catalog import (
    _unpack_euler16 as unpack_euler16,  # data is compressed using this custom abacus encoding
)

from abacustsz.converters import convert_comoving_to_sky, convert_rad_to_cart
from abacustsz.cosmology.model import (
    H_ABACUS,
    MAX_M200C_MSUN,
    m200m_to_m200c,
    mdeltamean_to_m200m,
)


@dataclass
class SimulationData:
    theta: np.ndarray
    phi: np.ndarray
    m_halos: np.ndarray
    particle_counts: np.ndarray
    redshift: float
    radii_halos: np.ndarray
    eigenvalues: np.ndarray  # (n_halos, 3) semi-axis lengths in physical Mpc
    eigenvectors: np.ndarray  # (n_halos, 3, 3) rows are maj > mid > min eigenvectors
    chi: np.ndarray = None  # comoving distance per halo [Mpc/h]
    # shell edges and neighbouring-shell halos (ownership only, never painted)
    chi_min: float | None = None
    chi_max: float | None = None
    ghost_xyz: np.ndarray | None = None  # (n_ghost, 3) comoving Mpc/h
    ghost_m_halos: np.ndarray | None = None  # M200c [Msun]
    ghost_radii: np.ndarray | None = None  # R200c, comoving Mpc/h

    @property
    def halo_xyz(self):
        return convert_rad_to_cart(self.theta, self.phi)

    @property
    def halo_xyz_comoving(self):
        """(n_halos, 3) comoving positions [Mpc/h], observer at the origin."""
        return convert_rad_to_cart(self.theta, self.phi) * np.asarray(self.chi, dtype=np.float64)[:, None]


def load_abacus_healcounts(filepath: Path):
    with asdf.open(filepath) as f:
        particle_counts = np.array(f["data"]["heal-counts"])
    return particle_counts


def obtain_healcount_edges(filepath: Path):
    """Obtain the inner and outer edges (in comoving distance in Mpc/h) of the lightcone from a heal-counts file."""
    with asdf.open(filepath) as f:
        headers = f.tree["headers"]
        chis = [h["CoordinateDistanceHMpc"] for h in headers]
    return min(chis), max(chis)


def obtain_healcount_mean_redshift(filepath: Path) -> float:
    """Return the mean redshift of the healcounts shell from its headers."""
    with asdf.open(filepath) as f:
        headers = f.tree["headers"]
        zs = [h["Redshift"] for h in headers]
    return float(np.mean(zs))


def find_duplicate_halos(
    halo_timeslice_index: np.ndarray,
    halo_xyz: np.ndarray,
    comoving_distance: np.ndarray,
    max_separation: float = 1.0,
    n_neighbours: int = 4,
) -> np.ndarray:
    """Mask of rows repeating an earlier row: same time-slice halo within max_separation [Mpc/h].

    Periodic images (one box length apart) are kept.
    """
    n = len(halo_timeslice_index)
    dup = np.zeros(n, dtype=bool)
    if n < 2:
        return dup
    order = np.lexsort((comoving_distance, halo_timeslice_index))
    ti = np.asarray(halo_timeslice_index)[order]
    for k in range(1, n_neighbours + 1):
        cand = np.flatnonzero(ti[k:] == ti[:-k])
        if len(cand) == 0:
            break  # no group has more than k members
        a, b = order[cand], order[cand + k]
        d = np.linalg.norm(
            np.asarray(halo_xyz[a], dtype=np.float64) - np.asarray(halo_xyz[b], dtype=np.float64), axis=1
        )
        close = d < max_separation
        dup[np.maximum(a[close], b[close])] = True
    return dup


def load_abacus_halos(
    halo_dir: Path,
    logm_min: float = 11.5,
    dedupe: bool = True,
):
    with asdf.open(halo_dir) as af:
        h = af["header"]
        particle_mass = h["ParticleMassHMsun"]
        redshift = h["Redshift"]
        box_size = h["BoxSizeMpc"]
        so_density_l1 = h["SODensityL1"]

        d = af["halo_lightcone"]
        halo_xyz = np.asarray(d["Interpolated_x_L2com"])
        num_particles = np.asarray(d["Interpolated_N"])
        comoving_distance = np.asarray(d["Interpolated_ComovingDist"])
        halo_timeslice_index = np.asarray(d["halo_timeslice_index"])

        m_halos = num_particles.astype(np.float64) * particle_mass

        # mass cut to limit memory
        threshold = 10**logm_min
        cut = m_halos > threshold

        halo_xyz = halo_xyz[cut]

        num_particles = num_particles[cut]
        m_halos = m_halos[cut]
        halo_timeslice_index = halo_timeslice_index[cut]
        comoving_distance = comoving_distance[cut]

        if dedupe:
            # halos written twice at the faces between periodic box copies
            dup = find_duplicate_halos(halo_timeslice_index, halo_xyz, comoving_distance)
            if dup.any():
                logger.info(
                    f"{Path(halo_dir).parent.name}: dropping {int(dup.sum()):,} duplicated light-cone halos "
                    f"({dup.mean():.3%} of those above the mass cut)"
                )
                keep = ~dup
                halo_xyz = halo_xyz[keep]
                num_particles = num_particles[keep]
                m_halos = m_halos[keep]
                halo_timeslice_index = halo_timeslice_index[keep]
                comoving_distance = comoving_distance[keep]

        # CompaSO masses are defined at SODensityL1 x mean density, not 200m
        m_halos = mdeltamean_to_m200m(m_halos, so_density_l1, redshift)

        e = af["halo_timeslice"]
        r100_ref = np.asarray(e["r100_L2com"], dtype=np.float64)

        # triaxial halos
        eigenvalues = np.asarray(e["sigman_L2com_i16"], dtype=np.float64)
        eigenvectors = np.asarray(e["sigman_eigenvecs_L2com_u16"], dtype=np.float64)

        # eigenvectors are stored in Abacus's compressed uint16 encoding
        sigmar_min, sigmar_mid, sigmar_maj = unpack_euler16(eigenvectors)

        r100_ref = r100_ref[halo_timeslice_index]

        eigenvalues = eigenvalues[halo_timeslice_index]
        sigmar_min = sigmar_min[halo_timeslice_index]
        sigmar_mid = sigmar_mid[halo_timeslice_index]
        sigmar_maj = sigmar_maj[halo_timeslice_index]

        INT16SCALE = 32000
        eigenvalues = eigenvalues * r100_ref[:, np.newaxis] * box_size / INT16SCALE

        eigenvectors = np.stack(
            [sigmar_maj, sigmar_mid, sigmar_min], axis=1
        )  # eigenvalues given in descending order 

    return (
        halo_xyz,
        num_particles,
        particle_mass,
        redshift,
        comoving_distance,
        eigenvalues,
        eigenvectors,
        m_halos,
    )


def load_abacus_for_painting(
    halo_catalog_index: list,  # list[HaloCatalogInfo] from load_halo_catalog_index()
    healcounts_file_1: Path,
    nside: int | None = None,
    logm_min: float = 11.5,
):
    chi_min, chi_max = obtain_healcount_edges(healcounts_file_1)
    redshift = obtain_healcount_mean_redshift(healcounts_file_1)
    chi_min_cut = chi_min
    chi_max_cut = chi_max

    matching = [
        cat
        for cat in halo_catalog_index
        if max(cat.chi_min, chi_min) <= min(cat.chi_max, chi_max)
    ]
    if not matching:
        raise ValueError(
            f"No halo catalogs overlap chi=[{chi_min:.2f}, {chi_max:.2f}]. "
            "Check your halo_catalog_index."
        )

    all_xyz, all_npart, all_evals, all_evecs, all_chi, all_m200m = [], [], [], [], [], []
    particle_mass = None
    for cat in matching:
        xyz, npart, pm, _, comoving_distance, evals, evecs, m200m = load_abacus_halos(
            cat.file_path, logm_min=logm_min
        )
        if particle_mass is None:
            particle_mass = pm  # simulation-wide constant, same across files
        in_shell = (comoving_distance >= chi_min_cut) & (
            comoving_distance <= chi_max_cut
        )
        all_xyz.append(xyz[in_shell])
        all_npart.append(npart[in_shell])
        all_evals.append(evals[in_shell])
        all_evecs.append(evecs[in_shell])
        all_chi.append(comoving_distance[in_shell])
        all_m200m.append(m200m[in_shell])

    halo_xyz = np.concatenate(all_xyz)
    num_particles = np.concatenate(all_npart)
    eigenvalues = np.concatenate(all_evals)
    eigenvectors = np.concatenate(all_evecs)
    chi = np.concatenate(all_chi)
    m_halos = np.concatenate(all_m200m)

    theta, phi = convert_comoving_to_sky(halo_xyz)
    m_halos, r200c_physical_kpch = m200m_to_m200c(m_halos, redshift)  # M200c [Msun/h], r200c [physical kpc/h]
    m_halos = m_halos / H_ABACUS  # M200c [Msun], as expected by Battaglia interpolator
    radius = (r200c_physical_kpch / 1000) * (1 + redshift)  # physical kpc/h -> comoving Mpc/h
    particle_counts = load_abacus_healcounts(healcounts_file_1)
    chi = np.linalg.norm(halo_xyz, axis=1)

    # remove halos that are masked in ACT DR5
    mass_cut = m_halos <= MAX_M200C_MSUN
    theta = theta[mass_cut]
    phi = phi[mass_cut]
    m_halos = m_halos[mass_cut]
    radius = radius[mass_cut]
    eigenvalues = eigenvalues[mass_cut]
    eigenvectors = eigenvectors[mass_cut]
    chi = chi[mass_cut]

    return SimulationData(
        theta,
        phi,
        m_halos,
        particle_counts,
        redshift,
        radius,
        eigenvalues,
        eigenvectors,
        chi=chi,
        chi_min=float(chi_min),
        chi_max=float(chi_max),
    )


def load_ghost_halos(
    halo_catalog_index: list,
    chi_min: float,
    chi_max: float,
    buffer: float,
    redshift: float,
    logm_min: float = 11.5,
):
    """Neighbouring-shell halos within buffer [Mpc/h] of the shell edges, with the shell's cuts.

    Returns (xyz [comoving Mpc/h], M200c [Msun], R200c [comoving Mpc/h]).
    """
    lo, hi = chi_min - buffer, chi_max + buffer
    xyz_l, m_l = [], []
    for cat in halo_catalog_index:
        if max(cat.chi_min, lo) > min(cat.chi_max, hi):
            continue
        xyz, _, _, _, chi_cat, _, _, m200m = load_abacus_halos(cat.file_path, logm_min=logm_min)
        sel = ((chi_cat >= lo) & (chi_cat < chi_min)) | ((chi_cat > chi_max) & (chi_cat <= hi))
        xyz_l.append(xyz[sel])
        m_l.append(m200m[sel])
    if not xyz_l or sum(len(a) for a in xyz_l) == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    xyz = np.concatenate(xyz_l).astype(np.float64)
    m200c_h, r200c_phys_kpch = m200m_to_m200c(np.concatenate(m_l), redshift)
    m = m200c_h / H_ABACUS
    r200 = (r200c_phys_kpch / 1000.0) * (1.0 + redshift)
    keep = m <= MAX_M200C_MSUN
    return xyz[keep], m[keep], r200[keep]


def attach_ghost_halos(
    data: SimulationData,
    halo_catalog_index: list,
    buffer: float,
    logm_min: float = 11.5,
) -> SimulationData:
    """Fill data.ghost_* from the neighbouring shells (in place; returns data)."""
    if data.chi_min is None or data.chi_max is None:
        raise ValueError("SimulationData has no shell edges; load it with load_abacus_for_painting")
    xyz, m, r200 = load_ghost_halos(
        halo_catalog_index, data.chi_min, data.chi_max, buffer, float(data.redshift), logm_min
    )
    data.ghost_xyz, data.ghost_m_halos, data.ghost_radii = xyz, m, r200
    return data
