import numpy as np


def convert_rad_to_cart(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Spherical (theta, phi) to Cartesian unit vectors."""
    sin_theta = np.sin(theta)
    return np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)], axis=1
    )


def normalize_angle(angle: np.ndarray):
    """Normalize angles to the range [0, 2pi)."""
    return np.where(angle < 0, angle + 2 * np.pi, angle)


def convert_cart_to_rad(xyz: np.ndarray):
    """Cartesian unit vectors to spherical (theta, phi)."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    phi = normalize_angle(phi)
    return theta, phi


def convert_comoving_to_sky(xyz: np.ndarray):
    """
    Convert comoving box coordinates to healpix sky coordinates.
    Abacus has the origin at the center of the box.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x) 
    phi = normalize_angle(phi)
    return theta, phi
