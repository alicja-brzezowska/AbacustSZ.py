import os
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"
Y_PROFILE_DATA_PATH = DATA_PATH / "y_profile"

OUTPUT_PATH = Path(os.environ.get("ABACUSTSZ_OUTPUT", Path(__file__).parent.parent / "output"))
LOGS_PATH = Path(os.environ.get("ABACUSTSZ_LOGS", OUTPUT_PATH / "logs"))
INTERPOLATORS_PATH = OUTPUT_PATH / "interpolators"

# AbacusSummit_backlight_cp999 light cone
ABACUS_DATA_PATH = Path(os.environ.get("ABACUS_DATA_PATH", DATA_PATH / "backlight_cp999"))
HALO_CATALOGS_PATH = ABACUS_DATA_PATH / "lightcone_halos"
HEALCOUNTS_PATH = ABACUS_DATA_PATH / "lightcone_healpix" / "halo" / "heal-counts"
HEALCOUNTS_TOTAL_PATH = ABACUS_DATA_PATH / "lightcone_healpix" / "total" / "heal-counts"
