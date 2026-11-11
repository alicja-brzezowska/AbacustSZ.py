"""Index of the light-cone halo catalogues by comoving distance.

Each halo snapshot covers the comoving-distance range halfway to its
neighbouring snapshots. A heal-counts shell is painted with the halos of every
snapshot whose range overlaps it (see abacustsz.paint.abacus_loader).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import asdf
import numpy as np

from abacustsz.paths import HALO_CATALOGS_PATH, OUTPUT_PATH

HALO_FILENAME = "lightcone_halo_info_000.asdf"


@dataclass
class HaloCatalogInfo:
    file_path: Path
    redshift: float
    chi_center: float
    chi_min: float
    chi_max: float


def discover_halo_catalogs(halo_root: Path, halo_filename: str = HALO_FILENAME) -> list[HaloCatalogInfo]:
    records = []
    for z_dir in sorted(d for d in halo_root.iterdir() if d.is_dir() and d.name.startswith("z")):
        halo_file = z_dir / halo_filename
        if halo_file.exists():
            with asdf.open(halo_file) as af:
                chi = float(af["header"]["CoordinateDistanceHMpc"])
            records.append((float(z_dir.name[1:]), halo_file, chi))
    records.sort(key=lambda rec: rec[0])

    centers = np.array([rec[2] for rec in records])
    if len(records) == 1:
        lowers, uppers = [-np.inf], [np.inf]
    else:
        mid = 0.5 * (centers[:-1] + centers[1:])
        lowers = np.concatenate(([centers[0] - 0.5 * (centers[1] - centers[0])], mid))
        uppers = np.concatenate((mid, [centers[-1] + 0.5 * (centers[-1] - centers[-2])]))

    return [
        HaloCatalogInfo(file_path=fp, redshift=z, chi_center=chi, chi_min=float(lo), chi_max=float(hi))
        for (z, fp, chi), lo, hi in zip(records, lowers, uppers)
    ]


def build_halo_catalog_index(
    halo_root: Path, output_json: Path, halo_filename: str = HALO_FILENAME
) -> list[HaloCatalogInfo]:
    catalogs = discover_halo_catalogs(halo_root, halo_filename)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "file_path": str(c.file_path),
            "redshift": c.redshift,
            "chi_center": c.chi_center,
            "chi_min": c.chi_min,
            "chi_max": c.chi_max,
        }
        for c in catalogs
    ]
    output_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return catalogs


def load_halo_catalog_index(json_path: Path) -> list[HaloCatalogInfo]:
    records = json.loads(json_path.read_text(encoding="utf-8"))
    return [
        HaloCatalogInfo(
            file_path=Path(r["file_path"]),
            redshift=r["redshift"],
            chi_center=r["chi_center"],
            chi_min=r["chi_min"],
            chi_max=r["chi_max"],
        )
        for r in records
    ]


def main():
    output_json = OUTPUT_PATH / "halo_catalog_index.json"
    catalogs = build_halo_catalog_index(HALO_CATALOGS_PATH, output_json)
    print(f"Built index with {len(catalogs)} halo catalogs -> {output_json}")


if __name__ == "__main__":
    main()
