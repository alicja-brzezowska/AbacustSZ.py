"""
Mass conversions: M 500 c to M 200 m and  M 200 m to M 200 c. (Diemer 2018, ApJS 239, 35)

M 500 c to M 200 m is used to apply the cluster mask (M 500 c < 3.8e14 M_sun) in the ACT DR5 cluster catalog (Hilton+2021, arxiv:2009.11043).
-> simplifications: does not capture the redshift dependence, however over  0.4 < z < 0.8 expected to be negligible.

"""

import numpy as np
from colossus.cosmology import cosmology as colossus_cosmo
from colossus.halo import concentration, mass_defs

colossus_cosmo.setCosmology("planck18")

H_ABACUS      = 0.6736
OMEGA_B_ABACUS = 0.02237 / H_ABACUS**2   
OMEGA_C_ABACUS = 0.12000 / H_ABACUS**2   
OMEGA_M_ABACUS = OMEGA_B_ABACUS + OMEGA_C_ABACUS  
N_S_ABACUS    = 0.9649
SIGMA8_ABACUS = 0.8159


MAX_M500C_MSUN = 3.8e14  
Z_REF = 0.5 

# Concentration model for mass conversion 
C_MODEL = "diemer19"


def m500c_to_m200m(m500c_h: float, z: float, c_model: str = C_MODEL) -> float:
    """Convert M500c [M_sun/h] -> M200m [M_sun/h] at redshift z."""
    c = concentration.concentration(m500c_h, "500c", z, model=c_model)
    m200m_h, _, _ = mass_defs.changeMassDefinition(m500c_h, c, z, "500c", "200m")
    return float(m200m_h)


def m200m_to_m200c(
    m200m_h: np.ndarray, z: float, c_model: str = C_MODEL
) -> tuple[np.ndarray, np.ndarray]:
    """Convert M200m [Msun/h] -> M200c [Msun/h] and r200c [physical kpc/h]."""
    c = concentration.concentration(m200m_h, "200m", z, model=c_model)
    m200c_h, r200c_physical_Mpch, _ = mass_defs.changeMassDefinition(
        m200m_h, c, z, "200m", "200c"
    )
    return np.asarray(m200c_h, dtype=np.float64), np.asarray(r200c_physical_Mpch, dtype=np.float64)


MAX_M200M_H: float = m500c_to_m200m(MAX_M500C_MSUN * H_ABACUS, Z_REF)


