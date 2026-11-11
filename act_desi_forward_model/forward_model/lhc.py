"""Latin hypercube over the gNFW pressure-profile parameters.

Writes the design to lhc_samples.txt and, with --submit, launches one
y-profile interpolator job per point on SLURM.
"""

import argparse

import numpy as np
from scipy.stats import qmc

PARAM_NAMES = ["alpha", "beta0", "gamma", "log10_P0"]
PARAM_BOUNDS = np.array([
    [0.1, 2.0],
    [0.1, 6.0],
    [-1.0, -0.1],
    [-2.0, 2.0],
])


def latin_hypercube(n, seed=42):
    unit = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed).random(n=n)
    return qmc.scale(unit, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])


def submit(samples, account, partition, log_dir):
    import submitit

    from abacustsz.conf.config_schema import YProfileConfig
    from abacustsz.y_profile.y_profile_jax import run_y_profile

    executor = submitit.AutoExecutor(folder=f"{log_dir}/%j")
    executor.update_parameters(
        slurm_account=account,
        slurm_partition=partition,
        cpus_per_task=32,
        mem_gb=64,
        timeout_min=6 * 60,
        slurm_array_parallelism=50,
    )
    configs = [YProfileConfig(*map(float, row)) for row in samples]
    jobs = executor.map_array(run_y_profile, configs)
    print(f"Submitted {len(jobs)} interpolator jobs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=130)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="lhc_samples.txt")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--account")
    parser.add_argument("--partition")
    parser.add_argument("--log-dir", default="logs/lhc")
    args = parser.parse_args()

    samples = latin_hypercube(args.n, args.seed)
    np.savetxt(args.out, samples, header=" ".join(PARAM_NAMES))
    print(f"Wrote {len(samples)} points to {args.out}")

    if args.submit:
        submit(samples, args.account, args.partition, args.log_dir)


if __name__ == "__main__":
    main()
