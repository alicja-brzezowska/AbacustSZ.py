# Emulator training set

`lhc128_ringring_f150.npz` holds the stacked ring-ring profiles of 128 Latin-hypercube
y-maps on the mock DESI-like LRGs, produced by `forward_model/stack_lrgs.py` with the ACT f150 beam.

| key | shape | |
|---|---|---|
| `params` | (128, 4) | alpha, beta0, gamma, log10P0 |
| `param_names` | (4,) | |
| `profiles` | (128, 9) | mean ring-ring filtered y per aperture |
| `errors` | (128, 9) | standard error of the mean over galaxies |
| `apertures_arcmin` | (9,) | 2.0 to 6.0 arcmin |

- The maps were painted with the legacy truncation (10 R200c below M200c = 1e14 Msun,
  4 R200c above), not the closure radius that is now the default. To reproduce them,
  set `TSZ_TRUNCATION_MODE=legacy` and `SEARCH_RADIUS_MODE = "legacy"` in `abacustsz/paint/main.py`.
- Row 115 (alpha = 1.808) has only 6 of the 20 light-cone shells summed; leave it out
  of training with `--exclude-rows 115`.
- These profiles produced the fits in the thesis:

```
python inference/inference.py data/training_sets/lhc128_ringring_f150.npz --exclude-rows 115 --freq 150
```
