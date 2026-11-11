# AbacustSZ.py

AbacustSZ is a package for generating thermal Sunyaev-Zel'dovich (Compton-y) maps from the
HEALPix Backlight light cones of the AbacusSummit N-body simulation suite (Maksimova et al. 2021,
arXiv:2110.11398). 
The repository also contains a simulation-based forward model, emulator and Bayesian inference of the gas pressure profile
from the ACT DR5 x DESI LRG tSZ measurement of Liu et al. (2025, arXiv:2502.08850) as a use case.

Code for my Master's thesis under Dr. Boryana Hadzhiyska (University of Cambridge).

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

## Using abacustSZ.py

### Requirements

```bash
uv sync --all-extras
```

Simulation products requirements:
The maps are painted from two AbacusSummit_backlight_cp999 light-cone products, which must
cover the redshift range of the y-map you want:

- heal-counts shells (`LightCone0_total_heal-counts_Step<NNNN-NNNN>.asdf`): each file is one
  shell of the light cone, and its header gives the shell's redshift and comoving distance;
- light-cone halo catalogues (`z<z>/lightcone_halo_info_000.asdf`): each shell is painted with
  the halos of the snapshots whose distance range overlaps it, so you need the snapshots that
  bracket your redshift range.


```bash
export ABACUS_DATA_PATH=/path/to/backlight_cp999
export ABACUSTSZ_OUTPUT=/path/to/output
uv run python -m abacustsz.scripts.match_redshifts      # index the halo catalogues (once)
```

### Generating y-maps

**1. Choose the pressure profile.** The gas pressure follows a generalised NFW profile
(Battaglia et al. 2012),

$$
\frac{P(x)}{P_{200}} = P_0(M,z) \left(\frac{x}{x_c}\right)^{\gamma} \left[1 + \left(\frac{x}{x_c}\right)^{\alpha}\right]^{-\beta(M,z)},
\qquad x = \frac{r}{R_{200c}},
$$

with the mass and redshift dependence

$$
P_0(M,z) = P_0 \left(\frac{M}{10^{14}\,M_\odot}\right)^{0.154} (1+z)^{-0.758}, \qquad
x_c(M,z) = 0.497 \left(\frac{M}{10^{14}\,M_\odot}\right)^{-0.00865} (1+z)^{0.731},
$$

$$
\beta(M,z) = \beta_0 \left(\frac{M}{10^{14}\,M_\odot}\right)^{0.0393} (1+z)^{0.415}.
$$

The free parameters are $(\alpha, \beta_0, \gamma, \log_{10} P_0)$, with $M \equiv M_{200c}$; Battaglia et al. find $(1, 4.35, -0.3, 1.26)$.

The profile is truncated at the baryon closure radius by default. Environment variables:
`TSZ_TRUNCATION_MODE` (`closure` / `legacy`), `TSZ_CLOSURE_TABLE` (`flamingo_hifb` / `z0p75`),
`TSZ_LOS_GEOMETRY` (`cylinder` / `sphere`).

**2. Tabulate y(theta, z, M)** for those parameters, (default grid: 0.2 < z < 1, 10^11 < M < 10^15.7 Msun):

```bash
uv run python -m abacustsz.y_profile.y_profile_jax --alpha 1 --beta0 4.35 --gamma -0.3 --log10-P0 1.26
```

This takes ~15 min on 32 cores and writes the table to `$ABACUSTSZ_OUTPUT/interpolators/`.

**3. Paint a shell** with every table in that folder (or `--interp-dir <folder>`):

```bash
uv run python -m abacustsz.paint.main --step LightCone0_total_heal-counts_Step0628-0634.asdf
```

Each table gives one full-sky HEALPix map (NESTED, nside 8192) in `$ABACUSTSZ_OUTPUT/Step0XXX-0XXX/`:

```python
import asdf
with asdf.open(path) as f:
    y = f["data"]["y_map"][:]
```

Halos are painted as triaxial ellipsoids, weighted by the particle counts of the heal-counts map.
Halos with N m_p < 10^11.5 Msun/h are skipped, and halos above the ACT DR5 cluster-mask threshold
(M500c > 3.8e14 Msun) are removed (manually change the upper and lower mass threshold for needs).


On a SLURM cluster, submit one job per shell, e.g.

```bash
sbatch -A <ACCOUNT> -p <PARTITION> -c 32 --mem=100G -t 4:00:00 --wrap \
    "uv run python -m abacustsz.paint.main --step LightCone0_total_heal-counts_Step0628-0634.asdf"
```

On a workstation, painting a few tables needs ~64 GB of memory; `--n-halo-batches N` runs the
pixel search in N batches to lower it (all maps are then kept in memory, 3.2 GB each).

`act_desi_forward_model/forward_model/sum_y_maps.py` sums painted shells into one light-cone map.


## Running the Forward model for stacked tSZ profiles around DESI LRGs: 

### Inference (laptop)

```bash
uv sync --extra inference
cd act_desi_forward_model
uv run python inference/inference.py data/training_sets/lhc128_ringring_f150.npz 
```

### Full Forward model (SLURM cluster)

Needs the AbacusSummit_backlight_cp999 light cone products and Halo Catalogs.

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
