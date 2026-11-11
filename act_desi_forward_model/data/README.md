# Data

## ACT x DESI LRG tSZ profiles

R. H. Liu, S. Ferraro, E. Schaan, R. Zhou et al., *Measurements of the Thermal
Sunyaev-Zel'dovich Effect with ACT and DESI Photometric Galaxies*,
Phys. Rev. D 112, 083561 (2025), arXiv:2502.08850.
Supplementary data: https://doi.org/10.5281/zenodo.14706729 (CC-BY-4.0).
The Fig. 15 files below were provided by the authors.

| file | contents |
|---|---|
| `fig15.csv` | semi-cumulative ring-ring profiles (inner radius 1 arcmin) at 2-6 arcmin; columns `DESI_pz{1-4}_act_dr5_f{90,150}_ringring2` and `_err` |
| `fig15_cov.npz` | covariance for each column |

ACT DR5 f090 and f150 temperature maps, DESI photometric LRGs in four photo-z bins.
This analysis uses bin 2 (`pz2`).

## Light-cone shells

`light_cone_shells.txt`: mean redshift and comoving-distance range of each painted
heal-counts shell of AbacusSummit_backlight_cp999, and the halo snapshots whose
halos fill it. Written by `forward_model/halo_index.py`.

## Emulator training set

See `training_sets/README.md`.
