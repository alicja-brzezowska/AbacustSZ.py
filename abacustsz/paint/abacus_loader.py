import os
from dataclasses import dataclass
from pathlib import Path

import asdf
import healpy as hp
import numpy as np
from abacusnbody.data.compaso_halo_catalog import (
    _unpack_euler16 as unpack_euler16,  # data is compressed using this custom abacus encoding
)

from abacustsz.converters import convert_comoving_to_sky, convert_rad_to_cart
from abacustsz.cosmology.mass_conversion import m200m_to_m200c


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

    @property
    def halo_xyz(self):
        return convert_rad_to_cart(self.theta, self.phi)


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


def load_abacus_halos(
    halo_dir: Path,
    logm_min: float = 11.5,
):
    with asdf.open(halo_dir) as af:
        h = af["header"]
        particle_mass = h["ParticleMassHMsun"]
        redshift = h["Redshift"]
        box_size = h["BoxSizeMpc"]

        d = af["halo_lightcone"]
        halo_xyz = np.asarray(d["Interpolated_x_L2com"])
        num_particles = np.asarray(d["Interpolated_N"])
        comoving_distance = np.asarray(d["Interpolated_ComovingDist"])
        halo_timeslice_index = np.asarray(d["halo_timeslice_index"])

        m_halos = num_particles.astype(np.float64) * particle_mass

        # need a cut-off for memory reasons: 11.5 comfortably below
        threshold = 10**logm_min
        cut = m_halos > threshold

        halo_xyz = halo_xyz[cut]

        num_particles = num_particles[cut]
        m_halos = m_halos[cut]
        halo_timeslice_index = halo_timeslice_index[cut]
        comoving_distance = comoving_distance[cut]

        e = af["halo_timeslice"]
        r100_ref = np.asarray(e["r100_L2com"], dtype=np.float64)

        # triaxial halos
        eigenvalues = np.asarray(e["sigman_L2com_i16"], dtype=np.float64)
        eigenvectors = np.asarray(e["sigman_eigenvecs_L2com_u16"], dtype=np.float64)

        # u16 is a smart custom encoding of the eigenvectors, again convert to float
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
    )


def load_abacus_for_painting(
    halo_catalog_index: list,  # list[HaloCatalogInfo] from load_halo_catalog_index()
    healcounts_file_1: Path,
    nside: int | None = None,
    logm_min: float = 11.5,  # want to capture as many small halos as possible, but memory blows up for querying large halo catalogs
):
    """ """
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

    all_xyz, all_npart, all_evals, all_evecs, all_chi = [], [], [], [], []
    particle_mass = None
    for cat in matching:
        xyz, npart, pm, _, comoving_distance, evals, evecs = load_abacus_halos(
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

    halo_xyz = np.concatenate(all_xyz)
    num_particles = np.concatenate(all_npart)
    eigenvalues = np.concatenate(all_evals)
    eigenvectors = np.concatenate(all_evecs)
    chi = np.concatenate(all_chi)

    theta, phi = convert_comoving_to_sky(halo_xyz)
    H_ABACUS = 0.6736
    m_halos = num_particles.astype(np.float64) * particle_mass  # M200m [Msun/h]
    m_halos, r200c_physical_kpch = m200m_to_m200c(m_halos, redshift)  # M200c [Msun/h], r200c [physical kpc/h]
    m_halos = m_halos / H_ABACUS  # M200c [Msun], as expected by Battaglia interpolator
    radius = (r200c_physical_kpch / 1000) * (1 + redshift)  # physical kpc/h -> comoving Mpc/h
    particle_counts = load_abacus_healcounts(healcounts_file_1)
    chi = np.linalg.norm(halo_xyz, axis=1)

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
    )
