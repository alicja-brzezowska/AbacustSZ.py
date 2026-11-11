"""Gaussian-process emulator of the stacked tSZ profile.

One GP per aperture maps the gNFW parameters to log10 Y. The stacked profile is
exactly proportional to P0, so log10 P0 is removed from the GP inputs and added
back to the prediction.
"""

import argparse
import pickle

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler

RING_INNER_ARCMIN = 1.0


def load_training_set(path, exclude_rows=()):
    """Simulated ring-ring profiles from the stacked LHC maps, converted to Y [arcmin^2]."""
    d = np.load(path, allow_pickle=True)
    apertures = d["apertures_arcmin"]
    ring_area = np.pi * (apertures**2 - RING_INNER_ARCMIN**2)
    keep = np.setdiff1d(np.arange(len(d["params"])), exclude_rows)
    return d["params"], d["profiles"][keep] * ring_area, d["params"][keep], apertures, list(d["param_names"])


class GPEmulator:
    def __init__(self, params, profiles, p0_index=3, nu=2.5, n_restarts=15, seed=0):
        self.p0_index = p0_index
        log_p0 = params[:, p0_index]
        X = np.delete(params, p0_index, axis=1)
        self.x_scaler = StandardScaler().fit(X)
        X = self.x_scaler.transform(X)
        targets = np.log10(profiles) - log_p0[:, None]

        n_dim = X.shape[1]
        self.gps, self.y_scalers = [], []
        for j in range(profiles.shape[1]):
            y_scaler = StandardScaler().fit(targets[:, j : j + 1])
            y = y_scaler.transform(targets[:, j : j + 1]).ravel()
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                np.ones(n_dim), [(1e-2, 1e3)] * n_dim, nu=nu
            )
            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=n_restarts, alpha=1e-10, random_state=seed
            )
            self.gps.append(gp.fit(X, y))
            self.y_scalers.append(y_scaler)

    def predict(self, params):
        """Mean and 1-sigma emulator error of Y, both (M, N_ap)."""
        params = np.atleast_2d(np.asarray(params, dtype=float))
        log_p0 = params[:, self.p0_index]
        X = self.x_scaler.transform(np.delete(params, self.p0_index, axis=1))
        means, stds = [], []
        for gp, y_scaler in zip(self.gps, self.y_scalers):
            m, s = gp.predict(X, return_std=True)
            log_y = y_scaler.inverse_transform(m[:, None]).ravel() + log_p0
            y = 10.0**log_y
            means.append(y)
            stds.append(y * np.log(10) * s * y_scaler.scale_[0])
        return np.column_stack(means), np.column_stack(stds)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_set", help="stacked LHC profiles (.npz from run_stacking.py)")
    parser.add_argument("--out", default="emulator.pkl")
    parser.add_argument("--exclude-rows", type=int, nargs="*", default=[])
    args = parser.parse_args()

    _, profiles, params, apertures, names = load_training_set(args.training_set, args.exclude_rows)
    print(f"Training on {len(params)} points, {len(apertures)} apertures, params {names}")
    GPEmulator(params, profiles, p0_index=names.index("log10P0")).save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
