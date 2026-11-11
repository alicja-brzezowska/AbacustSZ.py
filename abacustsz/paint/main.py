import argparse

from loguru import logger

from abacustsz.logging import setup_logging
from abacustsz.paint.config import PainterConfig
from abacustsz.paint.paint import paint_abacus
from abacustsz.paths import HEALCOUNTS_TOTAL_PATH, OUTPUT_PATH
from abacustsz.scripts.match_redshifts import load_halo_catalog_index

NSIDE = 8192
SEARCH_RADIUS_MODE = "closure"
SEARCH_RADIUS_LOW_MASS = 10.0
SEARCH_RADIUS_HIGH_MASS = 4.0
SEARCH_RADIUS_MASS_THRESHOLD = 1e14
WEIGHT_BIN_WIDTH = 2e-5
USE_WEIGHTS = True
LOGM_MIN = 11.5  # log10 N*m_particle [Msun/h]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, help="Healcounts filename, e.g. LightCone0_total_heal-counts_Step0617-0622.asdf")
    parser.add_argument("--run-label", default="", help="Prefix for output step directory, e.g. 'focused'")
    parser.add_argument("--interp-dir", default=None, help="Path to interpolator .pkl files (overrides default INTERPOLATORS_PATH)")
    parser.add_argument("--n-halo-batches", type=int, default=1, help="Split halos into this many sequential batches to cap peak memory (1 = no batching)")
    args = parser.parse_args()

    setup_logging("main")
    config = PainterConfig(
        nside=NSIDE,
        search_radius_mode=SEARCH_RADIUS_MODE,
        search_radius_low_mass=SEARCH_RADIUS_LOW_MASS,
        search_radius_high_mass=SEARCH_RADIUS_HIGH_MASS,
        search_radius_mass_threshold_msun=SEARCH_RADIUS_MASS_THRESHOLD,
        weight_bin_width=WEIGHT_BIN_WIDTH,
        n_halo_batches=args.n_halo_batches,
    )
    halo_catalog_index = load_halo_catalog_index(
        OUTPUT_PATH / "halo_catalog_index.json"
    )
    healcounts_file1 = HEALCOUNTS_TOTAL_PATH / args.step
    output_file = OUTPUT_PATH
    logger.info("Painting Abacus tSZ map...")
    logger.info(f"Halo catalog index: {len(halo_catalog_index)} entries")
    logger.info(f"Healcounts file 1: {healcounts_file1}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"logM min: {LOGM_MIN}")
    logger.info(f"Search radius mode: {SEARCH_RADIUS_MODE}")
    logger.info(f"Run label: {args.run_label!r}")
    logger.info(f"Interp dir: {args.interp_dir}")

    paint_abacus(
        config,
        halo_catalog_index=halo_catalog_index,
        healcounts_file_1=healcounts_file1,
        output_file=output_file,
        use_weights=USE_WEIGHTS,
        logm_min=LOGM_MIN,
        run_label=args.run_label,
        interp_dir=args.interp_dir,
    )


if __name__ == "__main__":
    main()
