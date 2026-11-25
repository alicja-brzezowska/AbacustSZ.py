"""Thin the LRG mocks by host halo to a target mean host mass.

Each halo is kept with probability p(M) = 1 below M_t and (M/M_t)^-s above,
M = host M200m [Msun/h]; a central and its satellites are kept or dropped
together, so the satellite fraction at fixed host mass is unchanged. s is solved
by bisection so that log10 <M200m> of the kept galaxies passing the stacking
cuts (painted shells, cluster mass cut) equals --target.

Reads <lrg-dir>/z<z>/galaxies_rsd/<in-name> for every snapshot in config.py and
writes <out-name> next to it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abacustsz.cosmology.model import mdeltamean_to_m200m
from abacustsz.paths import OUTPUT_PATH
from config import LRG_REDSHIFTS
from stack_lrgs import in_painted_shells, max_raw_mass, so_density

LBOX = 1000.0  # Mpc/h; tile copies of a box halo in the light cone sit LBOX apart


def halo_keys(pos, hid, ncent):
    """Key of each galaxy's light-cone halo copy, and the row that sets its host mass.

    The catalogue id is the box halo id, shared by every tile copy, so the copy
    is identified by (id, round((pos - pos_ref) / LBOX)) with pos_ref the first
    galaxy of that id (a central when the id has one; centrals come first).
    """
    n = len(hid)
    order = np.argsort(hid, kind="stable")
    hs = hid[order]
    starts = np.flatnonzero(np.r_[True, hs[1:] != hs[:-1]])
    first = np.empty(n, np.int64)
    first[order] = order[np.repeat(starts, np.diff(np.r_[starts, n]))]
    tile = np.rint((pos - pos[first]) / LBOX).astype(np.int64) + 8
    if hid.min() < 0 or hid.max() >= 2**50 or tile.min() < 0 or tile.max() > 15:
        raise ValueError("halo id or tile offset outside the key range")
    code = (tile[:, 0] * 16 + tile[:, 1]) * 16 + tile[:, 2]
    key = hid.astype(np.uint64) | (code.astype(np.uint64) << np.uint64(50))

    korder = np.argsort(key, kind="stable")
    ks = key[korder]
    kst = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    host = np.empty(n, np.int64)
    host[korder] = korder[np.repeat(kst, np.diff(np.r_[kst, n]))]
    return key, host


def uniform_from_key(key, seed):
    """One reproducible U(0, 1) number per halo (splitmix64 of the key)."""
    with np.errstate(over="ignore"):
        z = key + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * 2.0**-53


def keep_prob(m200m, log_mt, s):
    return np.minimum(1.0, (m200m / 10**log_mt) ** (-s))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lrg-dir", type=Path, default=OUTPUT_PATH / "hod_mocks" / "lightcone_halos")
    parser.add_argument("--in-name", default="LRGs.dat")
    parser.add_argument("--out-name", default="LRGs_thinned.dat")
    parser.add_argument("--target", type=float, default=13.25, help="log10 <M200m> [Msun/h] after the stacking cuts")
    parser.add_argument("--log-mt", type=float, default=13.5, help="log10 M_t, M200m [Msun/h]")
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--dry-run", action="store_true", help="solve for s without writing catalogues")
    args = parser.parse_args()

    snaps = []
    for i, z in enumerate(LRG_REDSHIFTS):
        t = Table.read(args.lrg_dir / f"z{z}" / "galaxies_rsd" / args.in_name, format="ascii.ecsv")
        ncent = int(t.meta["Ncent"])
        pos = np.column_stack([np.asarray(t[c], np.float64) for c in "xyz"])
        raw = np.asarray(t["mass"], np.float64)
        key, host = halo_keys(pos, np.asarray(t["id"], np.int64), ncent)
        m200m = mdeltamean_to_m200m(raw, so_density(z), float(z))
        stack = in_painted_shells(np.linalg.norm(pos, axis=1)) & (raw <= max_raw_mass(z))
        u = uniform_from_key(key, args.seed + 1000003 * i)
        snaps.append(dict(z=z, t=t, ncent=ncent, m200m=m200m, host_m200m=m200m[host], stack=stack, u=u,
                          sat=np.arange(len(raw)) >= ncent))
        print(f"z{z}: {len(raw):,} galaxies, {ncent:,} centrals, {stack.sum():,} pass the stacking cuts")

    def kept(d, s):
        return d["u"] < keep_prob(d["host_m200m"], args.log_mt, s)

    def stacked_mean(s):
        num = den = 0.0
        for d in snaps:
            k = d["stack"] & kept(d, s)
            num += d["m200m"][k].sum()
            den += k.sum()
        return np.log10(num / den)

    # bisection for s
    m0 = stacked_mean(0.0)
    print(f"unthinned log10<M200m> = {m0:.4f}, target {args.target}")
    if m0 <= args.target:
        raise SystemExit("already at or below the target")
    lo, hi = 0.0, 1.0
    while stacked_mean(hi) > args.target:
        hi *= 2
        if hi > 1e3:
            raise SystemExit("target not reachable by thinning above M_t")
    while hi - lo > 1e-6:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if stacked_mean(mid) > args.target else (lo, mid)
    s = hi

    stack = np.concatenate([d["stack"] for d in snaps])
    keep = np.concatenate([kept(d, s) for d in snaps])
    sat = np.concatenate([d["sat"] for d in snaps])
    print(f"s = {s:.5f}: log10<M200m> = {stacked_mean(s):.4f}; stacking cuts: "
          f"{100 * (1 - (keep & stack).sum() / stack.sum()):.2f}% of galaxies removed, "
          f"f_sat {sat[stack].mean():.3f} -> {sat[stack & keep].mean():.3f}")
    print("keep probability at log10 M200m = 13.5 / 14 / 14.5: "
          + " / ".join(f"{keep_prob(10**lm, args.log_mt, s):.3f}" for lm in (13.5, 14.0, 14.5)))
    if args.dry_run:
        return

    # write, centrals first, with the new central count
    for d in snaps:
        k, nc = kept(d, s), d["ncent"]
        out = d["t"][np.concatenate([np.flatnonzero(k[:nc]), nc + np.flatnonzero(k[nc:])])]
        out.meta = dict(d["t"].meta)
        out.meta["Ncent"] = int(k[:nc].sum())
        out.meta["thinning"] = (f"by halo; p = 1 below M200m = 1e{args.log_mt}, (M/M_t)^-{s:.5f} above; "
                                f"seed {args.seed}; source {args.in_name}")
        path = args.lrg_dir / f"z{d['z']}" / "galaxies_rsd" / args.out_name
        out.write(path, format="ascii.ecsv", overwrite=True)
        print(f"wrote {path} ({len(out):,} galaxies)")


if __name__ == "__main__":
    main()
