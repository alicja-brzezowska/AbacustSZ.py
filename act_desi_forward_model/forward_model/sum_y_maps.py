"""Sum the per-shell y-maps of each parameter set into one light-cone y-map.

The first shell's directory is used as the accumulator. Each map records the
shells it already contains in its `summed_steps` header, so the script can be
rerun safely after more shells have been painted.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import asdf
import numpy as np

from abacustsz.paths import OUTPUT_PATH
from config import STEPS


def sum_one(fname, data_dir, delete_sources):
    out_path = data_dir / STEPS[0] / fname
    with asdf.open(out_path, lazy_load=True) as f:
        total = np.asarray(f.tree["data"]["y_map"], dtype=np.float32).copy()
        header = dict(f.tree["header"])

    done = set(header.get("summed_steps", []))
    added = []
    for step in STEPS[1:]:
        src = data_dir / step / fname
        if step in done or not src.exists():
            continue
        with asdf.open(src, lazy_load=True) as f:
            total += np.asarray(f.tree["data"]["y_map"], dtype=np.float32)
        added.append(step)
    if not added:
        return f"{fname}: nothing to add"

    header["summed_steps"] = sorted(done | set(added))
    header.pop("healpix_file", None)
    tmp = out_path.with_suffix(".tmp")
    asdf.AsdfFile({"data": {"y_map": total}, "header": header}).write_to(
        tmp, all_array_compression="blsc"
    )
    os.replace(tmp, out_path)

    if delete_sources:
        for step in added:
            (data_dir / step / fname).unlink()
    missing = [s for s in STEPS[1:] if s not in done and s not in added]
    return f"{fname}: added {len(added)} shells" + (f", missing {missing}" if missing else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--nproc", type=int, default=6)
    parser.add_argument("--delete-sources", action="store_true",
                        help="delete each per-shell map once it has been added")
    args = parser.parse_args()

    fnames = sorted(p.name for p in (args.data_dir / STEPS[0]).glob("*.asdf"))
    work = partial(sum_one, data_dir=args.data_dir, delete_sources=args.delete_sources)
    with ProcessPoolExecutor(args.nproc) as ex:
        for msg in ex.map(work, fnames):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
