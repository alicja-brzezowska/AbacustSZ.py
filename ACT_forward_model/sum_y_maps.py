"""
Sum-y maps of different redshifts.

Current pass: add steps 0571-0576 and 0582-0587 on top of the existing accumulator
(Step0689-0695 already holds the sum of 0588-0695).
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import asdf
import numpy as np

DATA_DIR = Path("/home/ab2927/rds/hpc-work/tSZPaint_data")

# Step0689-0695 is last — it is both a source and the output.
STEP_DIRS = [
    DATA_DIR / "Step0571-0576",
    DATA_DIR / "Step0582-0587",
    DATA_DIR / "Step0689-0695",  # output; already holds sum of 0588-0695
]

ALL_SUMMED_STEPS = [
    "Step0571-0576", "Step0582-0587",
    "Step0588-0593", "Step0594-0598", "Step0599-0604", "Step0605-0610", "Step0611-0616",
    "Step0617-0622", "Step0623-0627", "Step0628-0634", "Step0635-0640", "Step0641-0646",
    "Step0647-0652",
    "Step0653-0658", "Step0659-0664", "Step0665-0670", "Step0671-0676",
    "Step0677-0682", "Step0683-0688", "Step0689-0695",
]


def sum_one_file(fname: str) -> str:
    out_path = STEP_DIRS[-1] / fname
    combined = None
    header = None

    for step_dir in STEP_DIRS:
        fpath = step_dir / fname
        if not fpath.exists():
            return f"WARNING {fname}: missing {fpath}"

        with asdf.open(str(fpath), lazy_load=True) as f:
            y = np.asarray(f.tree["data"]["y_map"], dtype=np.float32)
            if combined is None:
                combined = y
                header = dict(f.tree["header"])
            else:
                combined += y

    if combined is None:
        return f"ERROR {fname}: no data found"

    header["summed_steps"] = ALL_SUMMED_STEPS
    header.pop("healpix_file", None)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    asdf.AsdfFile({"data": {"y_map": combined}, "header": header}).write_to(
        str(tmp_path), all_array_compression="blsc"
    )
    os.replace(tmp_path, out_path)
    return f"DONE {fname}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", type=int, default=6)
    args = parser.parse_args()

    param_files = sorted(p.name for p in STEP_DIRS[-1].glob("*.asdf"))
    done = failed = 0

    print(f"Using {args.nproc} workers")
    print(f"Output: {STEP_DIRS[-1]}")

    with ProcessPoolExecutor(max_workers=args.nproc) as ex:
        futures = {ex.submit(sum_one_file, f): f for f in param_files}
        for i, fut in enumerate(as_completed(futures), 1):
            fname = futures[fut]
            try:
                msg = fut.result()
                print(f"[{i}/{len(param_files)}] {msg}", flush=True)
                if msg.startswith("DONE"):
                    done += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"[{i}/{len(param_files)}] ERROR {fname}: {e}", flush=True)
                failed += 1

    print(f"\nFinished.  done={done}  failed={failed}")
