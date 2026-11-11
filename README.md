# AbacustSZ

AbacustSZ is a package for generating thermal Sunyaev-Zel'dovich (Compton-y) maps from the
HEALPix light cones of the AbacusSummit N-body simulation suite (Maksimova et al. 2021,
arXiv:2110.11398), using the backlight light cone. The repository also contains a
simulation-based forward model, emulator and Bayesian inference of the gas pressure profile
from the ACT DR5 x DESI LRG tSZ measurement of Liu et al. (2025, arXiv:2502.08850) as a use case.
Code for my Master's thesis (University of Cambridge).

## Repository layout

```
AbacustSZ/
│
├── abacustsz/                  # The package: painting tSZ maps onto AbacusSummit halos
│   ├── y_profile/              # gNFW pressure profile, y(θ, z, M) interpolators, closure-radius truncation
│   ├── paint/                  # Halo loading, HEALPix pixel search, painting (entry point: main.py)
│   ├── cosmology/              # AbacusSummit cosmology and mass-definition conversions
│   └── scripts/                # Halo-catalogue index, radial profiles
│
├── act_desi_forward_model/
│   ├── forward_model/          # Pipeline from painted maps to stacked LRG profiles
│   │   ├── config.py           # Light-cone shells and LRG snapshot redshifts
│   │   ├── halo_index.py       # Which halo snapshots fill each shell
│   │   ├── lhc.py              # Latin hypercube over the gNFW parameters
│   │   ├── sum_y_maps.py       # Sum shells into light-cone maps
│   │   ├── preconvolve_maps.py # ACT beam convolution
│   │   ├── stack_lrgs.py       # Ring-ring stacking on mock LRGs -> emulator training set
│   │   └── hod/                # LRG mocks with AbacusHOD (modified abacusutils for lightcone format)
│   │
│   ├── inference/
│   │   ├── emulator.py         # Gaussian-process emulator of the stacked profile
│   │   └── inference.py        # Nested-sampling fit to the ACT data (dynesty)
│   │
│   └── data/
│       ├── fig15.csv, fig15_cov.npz   # ACT x DESI LRG profiles + covariance (Liu et al. 2025)
│       ├── light_cone_shells.txt      # Painted shells and their halo snapshots
│       └── training_sets/             # Stacked profiles of the 128 LHC maps (emulator training set)
│
├── README.md
├── pyproject.toml
└── uv.lock
```

## Inference (laptop)

```bash
uv sync --extra inference
cd act_desi_forward_model
uv run python inference/inference.py data/training_sets/lhc128_ringring_f150.npz --exclude-rows 115
```

## Forward model (SLURM cluster)

Needs the AbacusSummit_backlight_cp999 light cone.

```bash
uv sync --all-extras
export ABACUS_DATA_PATH=/path/to/backlight_cp999
export ABACUSTSZ_OUTPUT=/path/to/output
cd act_desi_forward_model

uv run python forward_model/halo_index.py
uv run python forward_model/lhc.py --submit --account <ACCOUNT> --partition <PARTITION>
uv run python -m abacustsz.paint.main --step LightCone0_total_heal-counts_<STEP>.asdf   # per shell in config.py
uv run python forward_model/sum_y_maps.py
uv run python forward_model/preconvolve_maps.py --freq f150
uv run python forward_model/hod/prepare_sim.py --path2config forward_model/hod/config.yaml --alt_z <z>   # per z in config.py
uv run python forward_model/hod/run_hod.py --z <z>
uv run python forward_model/stack_lrgs.py $ABACUSTSZ_OUTPUT/preconvolved_Step0571-0576_f150 --out training_f150.npz
uv run python inference/inference.py training_f150.npz
```

Truncation options (environment variables): `TSZ_TRUNCATION_MODE` (`closure` / `legacy`),
`TSZ_CLOSURE_TABLE` (`flamingo_hifb` / `z0p75`), `TSZ_LOS_GEOMETRY` (`cylinder` / `sphere`).

The included training set uses the legacy truncation (10 R200c below 1e14 Msun, 4 R200c above).
Row 115 has 6 of 20 shells summed and is excluded.

## Credits

- ACT x DESI LRG profiles: Liu et al. 2025, PRD 112, 083561, arXiv:2502.08850
- Pressure profile: Battaglia et al. 2012, ApJ 758, 75
- Closure radius: Ayromlou, Nelson & Pillepich 2023, arXiv:2211.07659; FLAMINGO tables (Schaye et al. 2023, arXiv:2306.04024)
- LRG mocks: AbacusHOD (Yuan et al. 2022, MNRAS 510, 3301); `hod/prepare_sim.py` is modified from abacusutils (GPL-3.0)
- Simulations: AbacusSummit (Maksimova et al. 2021, MNRAS 508, 4017)
