import numpy as np
from scipy.integrate import quad
from colossus.cosmology import cosmology as colossus_cosmo
from colossus.halo import concentration, mass_defs

# abacus_backlight_c9999_cp999 cosmology parameters, from header
H_ABACUS = 0.6711
OMEGA_B_ABACUS = 0.049
OMEGA_M_ABACUS = 0.3175
OMEGA_C_ABACUS = OMEGA_M_ABACUS - OMEGA_B_ABACUS
N_S_ABACUS = 0.9624
SIGMA8_ABACUS = 0.834

colossus_cosmo.setCosmology(
    "abacus_backlight_c9999",
    params=dict(
        flat=True,
        H0=100.0 * H_ABACUS,
        Om0=OMEGA_M_ABACUS,
        Ob0=OMEGA_B_ABACUS,
        sigma8=SIGMA8_ABACUS,
        ns=N_S_ABACUS,
    ),
)

MAX_M500C_MSUN = 3.8e14  # ACT DR5 cluster mask threshold (Hilton et al. 2021)
Z_REF = 0.5

C_MODEL = "diemer19"


def m500c_to_m200m(m500c_h: float, z: float, c_model: str = C_MODEL) -> float:
    """Convert M500c [M_sun/h] -> M200m [M_sun/h] at redshift z."""
    c = concentration.concentration(m500c_h, "500c", z, model=c_model)
    m200m_h, _, _ = mass_defs.changeMassDefinition(m500c_h, c, z, "500c", "200m")
    return float(m200m_h)


def concentration_200c(m200c_h, z: float, c_model: str = C_MODEL):
    """c200c(M200c [M_sun/h], z)."""
    return concentration.concentration(m200c_h, "200c", z, model=c_model)


def mdeltamean_to_m200m(
    m_delta_h: np.ndarray, delta: float, z: float, c_model: str = C_MODEL, n_grid: int = 100
) -> np.ndarray:

    mdef_in = f"{int(round(delta))}m"
    m_delta_h = np.asarray(m_delta_h, dtype=np.float64)
    if m_delta_h.size == 0:
        return m_delta_h.copy()

    log_m_grid = np.linspace(np.log10(m_delta_h.min()), np.log10(m_delta_h.max()), n_grid)
    m_grid = 10**log_m_grid
    c_grid = concentration.concentration(m_grid, mdef_in, z, model=c_model)
    m200m_grid, _, _ = mass_defs.changeMassDefinition(m_grid, c_grid, z, mdef_in, "200m")
    ratio_grid = m200m_grid / m_grid

    ratio = np.interp(np.log10(m_delta_h), log_m_grid, ratio_grid)
    return m_delta_h * ratio


def m200m_to_m200c(
    m200m_h: np.ndarray, z: float, c_model: str = C_MODEL, n_grid: int = 100
) -> tuple[np.ndarray, np.ndarray]:

    m200m_h = np.asarray(m200m_h, dtype=np.float64)
    if m200m_h.size == 0:
        return m200m_h.copy(), m200m_h.copy()

    log_m_grid = np.linspace(np.log10(m200m_h.min()), np.log10(m200m_h.max()), n_grid)
    m_grid = 10**log_m_grid
    c_grid = concentration.concentration(m_grid, "200m", z, model=c_model)
    m200c_grid, r200c_grid, _ = mass_defs.changeMassDefinition(m_grid, c_grid, z, "200m", "200c")
    mass_ratio_grid = m200c_grid / m_grid
    r_ratio_grid = r200c_grid / m_grid ** (1.0 / 3.0)

    log_m = np.log10(m200m_h)
    mass_ratio = np.interp(log_m, log_m_grid, mass_ratio_grid)
    r_ratio = np.interp(log_m, log_m_grid, r_ratio_grid)

    m200c_h = m200m_h * mass_ratio
    r200c_physical_Mpch = r_ratio * m200m_h ** (1.0 / 3.0)
    return np.asarray(m200c_h, dtype=np.float64), np.asarray(r200c_physical_Mpch, dtype=np.float64)


def m200c_to_m200m(m200c_h: float, z: float, c_model: str = C_MODEL) -> float:
    """Convert M200c [M_sun/h] -> M200m [M_sun/h]."""
    c = concentration.concentration(m200c_h, "200c", z, model=c_model)
    m200m_h, _, _ = mass_defs.changeMassDefinition(m200c_h, c, z, "200c", "200m")
    return float(m200m_h)


def m200m_to_mdeltamean(m200m_h: float, delta: float, z: float, c_model: str = C_MODEL) -> float:
    """Convert M200m [M_sun/h] -> M_{Delta,m} [M_sun/h]."""
    mdef_out = f"{int(round(delta))}m"
    c = concentration.concentration(m200m_h, "200m", z, model=c_model)
    m_delta_h, _, _ = mass_defs.changeMassDefinition(m200m_h, c, z, "200m", mdef_out)
    return float(m_delta_h)


MAX_M200M_H: float = m500c_to_m200m(MAX_M500C_MSUN * H_ABACUS, Z_REF)
_MAX_M200C_H, _ = m200m_to_m200c(np.array([MAX_M200M_H]), Z_REF)
MAX_M200C_MSUN: float = float(_MAX_M200C_H[0]) / H_ABACUS


# sigma(M) with the Eisenstein & Hu (1998) no-wiggle transfer function.
# Masses in M_sun/h, distances in Mpc/h.

DELTA_C = 1.686
RHO_CRIT0 = 2.775e11  # (M_sun/h) / (Mpc/h)^3


def _efunc(z):
    return np.sqrt(OMEGA_M_ABACUS * (1.0 + z) ** 3 + (1.0 - OMEGA_M_ABACUS))


def _growth_integrand(z):
    return (1.0 + z) / _efunc(z) ** 3


_D0 = quad(_growth_integrand, 0.0, np.inf, limit=300)[0]


def growth_factor(z):
    """Linear growth factor D(z), normalized to D(0)=1."""
    Dz = quad(_growth_integrand, z, np.inf, limit=300)[0]
    return _efunc(z) * Dz / _D0


def _eh98_transfer(k_h):
    """k_h in h/Mpc."""
    Gamma = OMEGA_M_ABACUS * H_ABACUS * np.exp(
        -OMEGA_B_ABACUS * (1.0 + np.sqrt(2.0 * H_ABACUS) / OMEGA_M_ABACUS)
    )
    q = k_h / Gamma
    L0 = np.log(2.0 * np.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return L0 / (L0 + C0 * q**2)


def _tophat_w(x):
    return np.where(
        x < 1e-3,
        1.0 - x**2 / 10.0,
        3.0 * (np.sin(x) - x * np.cos(x)) / x**3,
    )


def _sigma_integrand(lnk, R):
    k = np.exp(lnk)
    T = _eh98_transfer(k)
    Pk = k**N_S_ABACUS * T**2
    W = _tophat_w(k * R)
    return k**3 * Pk * W**2 / (2.0 * np.pi**2)


_R8 = 8.0
_SIG8_UNNORM_SQ = quad(
    _sigma_integrand, np.log(1e-5), np.log(1e4), args=(_R8,), limit=1000
)[0]


def sigma_m(M_h, z=0.0):
    """sigma(M, z), normalised to SIGMA8_ABACUS."""
    rho_m0 = OMEGA_M_ABACUS * RHO_CRIT0
    R = (3.0 * M_h / (4.0 * np.pi * rho_m0)) ** (1.0 / 3.0)
    sig_sq = quad(_sigma_integrand, np.log(1e-5), np.log(1e4), args=(R,), limit=1000)[0]
    sig0 = SIGMA8_ABACUS * np.sqrt(sig_sq / _SIG8_UNNORM_SQ)
    return sig0 * growth_factor(z)


def peak_height(M_h, z):
    """Peak height v(M, z) = delta_c / sigma(M, z)."""
    return DELTA_C / sigma_m(M_h, z)
