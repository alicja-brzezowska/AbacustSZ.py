"""Build the halo-catalogue index and show which halo snapshots fill each painted shell.

Run once before painting; abacustsz.paint.main reads the index it writes.
"""

import argparse
from pathlib import Path

from abacustsz.paint.abacus_loader import obtain_healcount_edges, obtain_healcount_mean_redshift
from abacustsz.paths import HALO_CATALOGS_PATH, HEALCOUNTS_TOTAL_PATH, OUTPUT_PATH
from abacustsz.scripts.match_redshifts import build_halo_catalog_index
from config import STEPS, healcounts_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH / "halo_catalog_index.json")
    parser.add_argument("--table", type=Path, default=Path(__file__).parents[1] / "data" / "light_cone_shells.txt")
    args = parser.parse_args()

    catalogs = build_halo_catalog_index(HALO_CATALOGS_PATH, args.out)
    lines = [f"# {'shell':13s} {'<z>':>6s} {'chi_min':>8s} {'chi_max':>8s}   halo snapshots (chi in Mpc/h)"]
    for step in STEPS:
        path = HEALCOUNTS_TOTAL_PATH / healcounts_file(step)
        lo, hi = obtain_healcount_edges(path)
        used = [f"z{c.redshift:.3f}" for c in catalogs if max(c.chi_min, lo) <= min(c.chi_max, hi)]
        lines.append(f"{step:15s} {obtain_healcount_mean_redshift(path):6.3f} {lo:8.1f} {hi:8.1f}   {','.join(used)}")
    args.table.write_text("\n".join(lines) + "\n")
    print(f"{len(catalogs)} halo snapshots -> {args.out}\nshell table -> {args.table}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
