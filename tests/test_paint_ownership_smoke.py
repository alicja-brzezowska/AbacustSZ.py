"""Painter with ownership on a synthetic shell: off vs on, batched vs unbatched."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TSZ_LOS_GEOMETRY", "sphere")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import healpy as hp
import numpy as np

from abacustsz.converters.coordinate_systems import convert_rad_to_cart
from abacustsz.paint.abacus_loader import SimulationData
from abacustsz.paint.config import PainterConfig
from abacustsz.paint.paint import (
    _slice_simulation_data,
    build_paint_context,
    cumulative_table_for,
    paint_one_interpolator,
    prepare_ownership,
)
from abacustsz.y_profile.interpolator import BattagliaLogInterpolator

rng = np.random.default_rng(1)
NSIDE = 256
Z = 0.634
N_HALOS = 400

# --- synthetic shell: halos clustered in a ~15 Mpc/h patch at chi ~ 1600 ---
chi = rng.uniform(1590.0, 1620.0, N_HALOS)
theta0, phi0 = 1.2, 0.8
ang = 15.0 / 1600.0
theta = theta0 + rng.uniform(-ang, ang, N_HALOS)
phi = phi0 + rng.uniform(-ang, ang, N_HALOS) / np.sin(theta0)
logm = rng.uniform(11.6, 14.3, N_HALOS)
m = 10.0**logm
r200 = 0.20 * 10 ** ((logm - 12.0) / 3.0)  # comoving Mpc/h
eig = np.repeat(r200[:, None], 3, axis=1)
evecs = np.repeat(np.eye(3)[None], N_HALOS, axis=0)
particle_counts = rng.poisson(5.0, hp.nside2npix(NSIDE)).astype(np.float32)

data = SimulationData(theta, phi, m, particle_counts, Z, r200, eig, evecs, chi=chi,
                      chi_min=1590.0, chi_max=1620.0)

# --- synthetic interpolator: y(theta, z, M) = A(M) * (1 + (theta/theta_s)^2)^-1 ---
log_thetas = np.linspace(np.log(1e-5), np.log(1e-1), 64)
redshifts = np.linspace(0.3, 1.0, 8)
log_masses = np.linspace(11.0, 15.5, 32)
T, Zg, Mg = np.meshgrid(log_thetas, redshifts, log_masses, indexing="ij")
theta_s = 1e-3 * 10 ** ((Mg - 13.0) / 3.0)
prof = 1e-6 * 10 ** (1.6 * (Mg - 13.0)) / (1.0 + (np.exp(T) / theta_s) ** 2)
interp = BattagliaLogInterpolator.from_matrices(log_thetas, redshifts, log_masses, prof)

base_cfg = dict(nside=NSIDE, search_radius_mode="closure", halo_geometry="spherical",
                pixel_search_workers=2, weight_bin_width=2e-5)

# --- ownership off ---
cfg_off = PainterConfig(**base_cfg)
ctx_off = build_paint_context(cfg_off, data, use_weights=True)
map_off = paint_one_interpolator(ctx_off, interp)

# --- ownership on, unbatched ---
cfg_on = PainterConfig(**base_cfg, ownership="density", ownership_c_roche=1.0, ownership_pair_chunk=1000)
geom = prepare_ownership(cfg_on, data, halo_catalog_index=[], workers=2)
assert geom is not None and geom.n_ghost == 0 and geom.n_pairs > 0
table = cumulative_table_for(data, 1.457, 1.383, -0.143)
ctx_on = build_paint_context(cfg_on, data, use_weights=True, ownership=geom)
map_on = paint_one_interpolator(ctx_on, interp, table)
map_on2 = paint_one_interpolator(ctx_on, interp, table)  # cache hit
assert np.array_equal(map_on, map_on2)
# one-pass baseline: the "unowned" map must equal the ownership-off paint exactly
owned, unowned = paint_one_interpolator(ctx_on, interp, table, also_unowned=True)
assert np.array_equal(owned, map_on) and np.array_equal(unowned, map_off)

assert np.all(map_on <= map_off * (1 + 1e-6) + 1e-12), "ownership increased a pixel"
red = 1.0 - map_on.sum() / map_off.sum()
assert 0.005 < red < 0.6, f"total reduction {red:.3f} outside the plausible range"
try:
    paint_one_interpolator(ctx_on, interp)  # no table -> must refuse
    raise AssertionError("missing table was not caught")
except ValueError:
    pass

# --- ownership on, batched with permuted global indices ---
order = rng.permutation(N_HALOS)
map_b = np.zeros_like(map_on)
for halo_idx in np.array_split(order, 3):
    batch = _slice_simulation_data(data, halo_idx)
    ctx_b = build_paint_context(cfg_on, batch, use_weights=True, ownership=geom, global_halo_idx=halo_idx)
    map_b += paint_one_interpolator(ctx_b, interp, table)
assert np.allclose(map_b, map_on, rtol=1e-5, atol=1e-12), np.abs(map_b - map_on).max()

print(f"smoke test ok: {len(ctx_on.pix_in_halos)} pairs, {geom.n_pairs} overlap pairs, "
      f"total y reduced by {100 * red:.1f}% with ownership; batched == unbatched")
