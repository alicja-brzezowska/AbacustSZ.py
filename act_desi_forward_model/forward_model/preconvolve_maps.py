"""Convolve every summed y-map with the ACT beam once, saving NESTED .npy maps."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from abacustsz.paths import OUTPUT_PATH
from config import STEPS
from filters import BEAM_FWHM_ARCMIN, convolve_beam, load_ymap


def convolve_one(src, out_dir, fwhm):
    dst = out_dir / (src.stem + ".npy")
    np.save(dst, convolve_beam(load_ymap(src), fwhm).astype(np.float32))
    return dst.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freq", choices=sorted(BEAM_FWHM_ARCMIN), required=True)
    parser.add_argument("--input-dir", type=Path, default=OUTPUT_PATH / STEPS[0])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--nproc", type=int, default=5)
    args = parser.parse_args()

    out_dir = args.output_dir or OUTPUT_PATH / f"preconvolved_{STEPS[0]}_{args.freq}"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in sorted(args.input_dir.glob("*.asdf")) if not (out_dir / (p.stem + ".npy")).exists()]
    print(f"{len(todo)} maps to convolve (FWHM {BEAM_FWHM_ARCMIN[args.freq]} arcmin) -> {out_dir}")

    work = partial(convolve_one, out_dir=out_dir, fwhm=BEAM_FWHM_ARCMIN[args.freq])
    with ProcessPoolExecutor(args.nproc, max_tasks_per_child=1) as ex:
        for name in ex.map(work, todo):
            print(name, flush=True)


if __name__ == "__main__":
    main()
