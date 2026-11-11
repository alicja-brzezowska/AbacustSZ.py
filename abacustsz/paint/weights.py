"""Redistribution of particle weights to pixels based on their distance from halo centers."""

from typing import Literal

import healpy as hp
import numpy as np
from numba import jit, prange

from abacustsz.logging import array_size, memory_usage, time_calls, trace_calls
from abacustsz.paint.config import PainterConfig


@jit(nopython=True, parallel=True)
def _fast_init_weights(
    particle_counts: np.ndarray, pixel_indices: np.ndarray
) -> np.ndarray:
    init_weights = np.empty(len(pixel_indices), dtype=np.float32)
    for i in prange(len(pixel_indices)):
        idx = pixel_indices[i]
        count = particle_counts[idx]
        init_weights[i] = np.power(count + 1e-10, 5.0 / 3.0)
    return init_weights


@memory_usage
@array_size
@time_calls
@trace_calls
@jit(nopython=True, parallel=True)
def weights_mechanism(
    search_radius: np.ndarray,
    bin_width: float,
    pixel_size: float,
    distances: np.ndarray,
    halo_starts: np.ndarray,
    halo_counts: np.ndarray,
    r_90: np.ndarray,
    init_weights: np.ndarray,
):
    """Compute normalized weights for each particle contribution within halos."""

    N_halos = len(r_90)
    weights = np.ones_like(distances, dtype=np.float32)

    for h in prange(N_halos):
        start = halo_starts[h]
        count = halo_counts[h]

        if count == 0:
            continue

        d = distances[start : start + count]
        w = init_weights[start : start + count]

        search_radius_h = search_radius[h] * r_90[h]
        # at least one pixel per bin, otherwise normalisation cancels the weight
        max_bins = max(1, int(search_radius_h / pixel_size))
        n_bins_h = min(max(1, int(np.rint(search_radius_h / bin_width))), max_bins)
        bin_width_h = search_radius_h / n_bins_h

        bin_ids = np.floor(d / bin_width_h).astype(np.int64)
        bin_ids = np.minimum(bin_ids, n_bins_h - 1)

        bin_pixel_tot = np.zeros(n_bins_h, dtype=np.float32)
        bin_weight_tot = np.zeros(n_bins_h, dtype=np.float32)
        for i in range(count):
            b = bin_ids[i]
            bin_pixel_tot[b] += 1.0
            bin_weight_tot[b] += w[i]

        normalization_per_bin = np.ones(n_bins_h, dtype=np.float32)
        for b in range(n_bins_h):
            if bin_pixel_tot[b] > 0.0:
                normalization_per_bin[b] = bin_pixel_tot[b] / bin_weight_tot[b]

        result = np.ones(count, dtype=np.float32)
        for i in range(count):
            b = bin_ids[i]
            result[i] = w[i] * normalization_per_bin[b]

        weights[start : start + count] = result

    return weights


@memory_usage
@array_size
@time_calls
@trace_calls
def compute_weights(
    config: PainterConfig,
    pixel_indices: np.ndarray,
    distances: np.ndarray,
    halo_starts: np.ndarray,
    halo_counts: np.ndarray,
    r_90: np.ndarray,
    search_radius: np.ndarray,
    particle_counts: np.ndarray,
    method: Literal["normal", "vectorized"] = "vectorized",
):
    """
    Calculate the proportional weights for pixels based on particle counts.
    """
    init_weights = _fast_init_weights(particle_counts, pixel_indices)

    pixel_size = float(hp.nside2resol(config.nside))

    if method == "normal":
        return weights_mechanism(
            search_radius,
            config.weight_bin_width,
            pixel_size,
            distances,
            halo_starts,
            halo_counts,
            r_90,
            init_weights,
        )
    else:
        return weights_mechanism_vec(
            search_radius,
            config.weight_bin_width,
            pixel_size,
            distances,
            halo_counts,
            r_90,
            init_weights,
        )


@time_calls
def weights_mechanism_vec(
    search_radius: np.ndarray,
    bin_width: float,
    pixel_size: float,
    distances: np.ndarray,
    halo_counts: np.ndarray,
    r_90: np.ndarray,
    raw_weights: np.ndarray,
) -> np.ndarray:
    """Normalised weights in radial bins around each halo."""
    halo_ids = np.repeat(np.arange(len(r_90)), halo_counts)

    search_radii = search_radius * r_90
    # at least one pixel per bin, otherwise normalisation cancels the weight
    max_bins_per_halo = np.maximum(1, (search_radii / pixel_size).astype(np.int64))
    n_bins_per_halo = np.minimum(
        np.maximum(1, np.rint(search_radii / bin_width).astype(np.int64)),
        max_bins_per_halo,
    )
    bin_width_per_halo = search_radii / n_bins_per_halo

    bin_ids = np.floor(distances / bin_width_per_halo[halo_ids]).astype(np.int64)
    bin_ids = np.minimum(bin_ids, n_bins_per_halo[halo_ids] - 1)

    max_bins = int(n_bins_per_halo.max())
    composite_key = halo_ids.astype(np.int64) * max_bins + bin_ids

    unique_keys, inverse_indices = np.unique(composite_key, return_inverse=True)

    bin_counts = np.bincount(inverse_indices, minlength=len(unique_keys))
    bin_sums = np.bincount(
        inverse_indices, weights=raw_weights, minlength=len(unique_keys)
    )

    normalization = np.where(bin_sums > 0, bin_counts / bin_sums, 1.0)

    weights = raw_weights * normalization[inverse_indices]

    return weights
