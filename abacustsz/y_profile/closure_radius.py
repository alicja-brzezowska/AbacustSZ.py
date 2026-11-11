"""Truncation radius of the gas pressure profile.

The profile is truncated at the baryon closure radius R_c(M): the radius at which
the enclosed baryon fraction f_b(<r) returns to within 5% (rc5) or 10% (rc10) of
Omega_b/Omega_m (Ayromlou, Nelson & Pillepich 2023, arXiv:2211.07659).

R_c / R200c is tabulated in 0.1 dex bins of log10 M200c [h^-1 Msun], measured
from the FLAMINGO simulations (Schaye et al. 2023, arXiv:2306.04024):
    flamingo_hifb : high-feedback (fgas-8sigma) run, default
    z0p75         : fiducial run at z = 0.75
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np

from abacustsz.cosmology.model import H_ABACUS

_BIN_WIDTH_DEX = 0.1
_LOG10_M_HINV_LO = np.round(np.arange(11.2, 14.6 - 1e-9, _BIN_WIDTH_DEX), 1)

_RC5 = np.array([
    3.14, 3.42, 3.56, 3.69, 4.12, 5.11, 6.58, 7.68, 8.42, 8.89,
    8.97, 8.96, 8.81, 8.53, 8.21, 7.80, 7.38, 6.95, 6.46, 6.01,
    5.58, 5.16, 4.77, 4.38, 3.98, 3.67, 3.37, 3.09, 2.79, 2.63,
    2.40, 2.23, 2.12, 1.97,
])
_RC10 = np.array([
    2.08, 2.35, 2.46, 2.52, 2.74, 3.31, 4.28, 5.19, 5.90, 6.41,
    6.64, 6.79, 6.77, 6.64, 6.45, 6.15, 5.85, 5.52, 5.12, 4.77,
    4.39, 4.03, 3.68, 3.37, 3.03, 2.75, 2.49, 2.28, 2.08, 1.89,
    1.74, 1.59, 1.47, 1.40,
])

_RC5_Z0P75 = np.array([
    3.60, 4.16, 4.47, 4.56, 4.60, 4.69, 5.02, 5.51, 5.92, 6.13,
    6.20, 6.15, 5.96, 5.70, 5.40, 5.05, 4.70, 4.34, 4.00, 3.67,
    3.37, 3.11, 2.89, 2.69, 2.51, 2.37, 2.25, 2.15, 2.06, 1.96,
    1.90, 1.82, 1.77, 1.70,
])
_RC10_Z0P75 = np.array([
    2.49, 3.00, 3.28, 3.36, 3.38, 3.41, 3.62, 3.98, 4.34, 4.57,
    4.68, 4.71, 4.62, 4.45, 4.25, 3.99, 3.71, 3.41, 3.13, 2.85,
    2.60, 2.37, 2.18, 2.00, 1.85, 1.72, 1.59, 1.46, 1.36, 1.19,
    1.10, 0.98, 0.88, np.nan,
])
assert _LOG10_M_HINV_LO.shape == _RC5.shape == _RC10.shape == _RC5_Z0P75.shape == _RC10_Z0P75.shape == (34,)

_TABLES = {
    "flamingo_hifb": {"rc5": _RC5, "rc10": _RC10},
    "z0p75": {"rc5": _RC5_Z0P75, "rc10": _RC10_Z0P75},
}
CLOSURE_TABLE = os.environ.get("TSZ_CLOSURE_TABLE", "flamingo_hifb").lower()
if CLOSURE_TABLE not in _TABLES:
    raise ValueError(f"TSZ_CLOSURE_TABLE must be one of {sorted(_TABLES)}, got {CLOSURE_TABLE!r}")
_COLUMNS = _TABLES[CLOSURE_TABLE]

# table bins are in h^-1 Msun; the pipeline uses Msun
LOG10_H = float(np.log10(H_ABACUS))
LOG10_M_MSUN_EDGES = np.append(_LOG10_M_HINV_LO, _LOG10_M_HINV_LO[-1] + _BIN_WIDTH_DEX) - LOG10_H
LOG10_M_MSUN_CENTRES = _LOG10_M_HINV_LO + 0.5 * _BIN_WIDTH_DEX - LOG10_H

# legacy truncation: 10 R200c below 1e14 Msun, 4 R200c above
LEGACY_RTRUNC_LOW_MASS = 10.0
LEGACY_RTRUNC_HIGH_MASS = 4.0
LEGACY_RTRUNC_MASS_THRESHOLD_MSUN = 1e14

TRUNCATION_MODE = os.environ.get("TSZ_TRUNCATION_MODE", "closure").lower()
CLOSURE_COLUMN = os.environ.get("TSZ_CLOSURE_COLUMN", "rc5").lower()
CLOSURE_INTERP = os.environ.get("TSZ_CLOSURE_INTERP", "interp").lower()
if TRUNCATION_MODE not in ("closure", "legacy"):
    raise ValueError(f"TSZ_TRUNCATION_MODE must be 'closure' or 'legacy', got {TRUNCATION_MODE!r}")
if CLOSURE_COLUMN not in _COLUMNS:
    raise ValueError(f"TSZ_CLOSURE_COLUMN must be one of {sorted(_COLUMNS)}, got {CLOSURE_COLUMN!r}")
if CLOSURE_INTERP not in ("interp", "step"):
    raise ValueError(f"TSZ_CLOSURE_INTERP must be 'interp' or 'step', got {CLOSURE_INTERP!r}")

# cylinder: LOS integral over |l| < R_t at every projected radius
# sphere:   P(r) = 0 for r > R_t in 3D
LOS_GEOMETRY = os.environ.get("TSZ_LOS_GEOMETRY", "cylinder").lower()
if LOS_GEOMETRY not in ("cylinder", "sphere"):
    raise ValueError(f"TSZ_LOS_GEOMETRY must be 'cylinder' or 'sphere', got {LOS_GEOMETRY!r}")


def _table(column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rc = _COLUMNS[column]
    ok = np.isfinite(rc)
    edges = np.append(LOG10_M_MSUN_EDGES[:-1][ok], LOG10_M_MSUN_EDGES[1:][ok][-1])
    return LOG10_M_MSUN_CENTRES[ok], edges, rc[ok]


def closure_radius_r200c(
    log10_m200c_msun,
    column: str = CLOSURE_COLUMN,
    interp: str = CLOSURE_INTERP,
) -> np.ndarray:
    """R_c / R200c at log10 M200c [Msun]; clamped outside the table."""
    logm = np.asarray(log10_m200c_msun, dtype=np.float64)
    centres, edges, rc = _table(column)
    if interp == "interp":
        return np.interp(logm, centres, rc)
    idx = np.clip(np.searchsorted(edges, logm, side="right") - 1, 0, len(rc) - 1)
    return rc[idx]


def closure_radius_r200c_jax(
    log10_m200c_msun,
    column: str = CLOSURE_COLUMN,
    interp: str = CLOSURE_INTERP,
):
    centres, edges, rc = _table(column)
    logm = jnp.asarray(log10_m200c_msun)
    if interp == "interp":
        return jnp.interp(logm, jnp.asarray(centres), jnp.asarray(rc))
    idx = jnp.clip(jnp.searchsorted(jnp.asarray(edges), logm, side="right") - 1, 0, len(rc) - 1)
    return jnp.asarray(rc)[idx]


def legacy_truncation_r200c(log10_m200c_msun) -> np.ndarray:
    mass = 10.0 ** np.asarray(log10_m200c_msun, dtype=np.float64)
    return np.where(
        mass < LEGACY_RTRUNC_MASS_THRESHOLD_MSUN, LEGACY_RTRUNC_LOW_MASS, LEGACY_RTRUNC_HIGH_MASS
    )


def legacy_truncation_r200c_jax(log10_m200c_msun):
    mass = 10.0 ** jnp.asarray(log10_m200c_msun)
    return jnp.where(
        mass < LEGACY_RTRUNC_MASS_THRESHOLD_MSUN, LEGACY_RTRUNC_LOW_MASS, LEGACY_RTRUNC_HIGH_MASS
    )


def truncation_radius_r200c(log10_m200c_msun, mode: str = TRUNCATION_MODE) -> np.ndarray:
    if mode == "legacy":
        return legacy_truncation_r200c(log10_m200c_msun)
    return closure_radius_r200c(log10_m200c_msun)


def truncation_radius_r200c_jax(log10_m200c_msun, mode: str = TRUNCATION_MODE):
    if mode == "legacy":
        return legacy_truncation_r200c_jax(log10_m200c_msun)
    return closure_radius_r200c_jax(log10_m200c_msun)


def truncation_tag(mode: str = TRUNCATION_MODE, geometry: str = LOS_GEOMETRY) -> str:
    suffix = "_sphere" if geometry == "sphere" else ""
    if mode == "legacy":
        return (
            f"legacy_{LEGACY_RTRUNC_LOW_MASS:g}_{LEGACY_RTRUNC_HIGH_MASS:g}"
            f"_at{LEGACY_RTRUNC_MASS_THRESHOLD_MSUN:.0e}{suffix}"
        )
    return f"closure_{CLOSURE_COLUMN}_{CLOSURE_INTERP}_{CLOSURE_TABLE}{suffix}"


def describe_table(column: str = CLOSURE_COLUMN) -> str:
    centres, edges, rc = _table(column)
    lines = [f"closure radius table ({column}), log10 M200c [Msun] bins (h={H_ABACUS}):"]
    for lo, hi, r in zip(edges[:-1], edges[1:], rc):
        lines.append(f"  {lo:6.3f} - {hi:6.3f} : Rc = {r:.2f} R200c")
    return "\n".join(lines)
