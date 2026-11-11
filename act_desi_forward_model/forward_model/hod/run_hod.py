"""Populate the light-cone halos at one snapshot with LRGs using AbacusHOD."""

import argparse
from pathlib import Path

import abacusnbody.hod.abacus_hod as abacus_hod
import asdf
import yaml
from abacusnbody.hod.abacus_hod import AbacusHOD

_asdf_open = asdf.open


def _open_with_lightcone_origins(*args, **kwargs):
    # the backlight headers call LightConeOrigins "LCOrigins"
    af = _asdf_open(*args, **kwargs)
    header = af.tree.get("header", {})
    if "LCOrigins" in header and "LightConeOrigins" not in header:
        header["LightConeOrigins"] = header["LCOrigins"]
    return af


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--nthread", type=int, default=8)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    sim_params, hod_params = config["sim_params"], config["HOD_params"]
    sim_params["z_mock"] = args.z
    if args.output_dir:
        sim_params["output_dir"] = args.output_dir

    abacus_hod.asdf.open = _open_with_lightcone_origins
    hod = AbacusHOD(sim_params, hod_params)
    mock = hod.run_hod(hod.tracers, want_rsd=True, write_to_disk=True, Nthread=args.nthread)
    print(f"{len(mock['LRG']['x']):,} LRGs written to {hod.mock_dir}")


if __name__ == "__main__":
    main()
