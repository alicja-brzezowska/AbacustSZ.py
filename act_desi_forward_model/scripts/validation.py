import argparse
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import dynesty
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dynesty.utils import resample_equal
from matplotlib.ticker import NullFormatter
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from emulator import GPEmulator, load_training_set
from inference import gaussian_loglike, load_act

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

_P = _Y = _P0 = _FIT = None


# worker state
def init_worker(P, Y, p0_index, fit):
    global _P, _Y, _P0, _FIT
    _P, _Y, _P0, _FIT = P, Y, p0_index, fit
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


def train_without(k):
    keep = np.arange(len(_P)) != k
    return GPEmulator(_P[keep], _Y[keep], p0_index=_P0)


def loo_one(k):
    mu, sig = train_without(k).predict(_P[k:k + 1])
    return mu[0], sig[0]


def recover_one(k):
    f = _FIT
    em = train_without(k)
    n_ap = f["n_ap"]
    rng = np.random.default_rng(f["seed"] + k)
    y = _Y[k, :n_ap].copy()
    if f["noise"]:
        y += rng.multivariate_normal(np.zeros(n_ap), f["cov"])

    lo, hi = f["bounds"][:, 0], f["bounds"][:, 1]

    def loglike(theta):
        mu, sig = em.predict(theta)
        return gaussian_loglike(mu[0, :n_ap], y, f["cov"] + np.diag(sig[0, :n_ap] ** 2))

    sampler = dynesty.NestedSampler(loglike, lambda u: lo + u * (hi - lo), len(f["bounds"]),
                                    nlive=f["nlive"], bound="multi", sample="rslice",
                                    slices=f["slices"], rstate=rng)
    sampler.run_nested(dlogz=f["dlogz"], print_progress=False)
    r = sampler.results
    eq = resample_equal(r.samples, np.exp(r.logwt - r.logz[-1]))
    print(f"  recovered row index {k} ({r.niter} iterations)", flush=True)
    return np.percentile(eq, [2.5, 16, 50, 84, 97.5], axis=0), r.samples[np.argmax(r.logl)]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(colors=MUTED, labelsize=8)


# LOO summary table
def loo_summary(th, truth, mu, sig, sig_data, near, label):
    rel = 100 * (mu - truth) / truth
    ed = (mu - truth) / sig_data
    pull = np.abs((mu - truth) / sig)
    lines = [f"leave one out test of {label}"]
    for name, m in [("all points", np.ones(len(truth), bool)), ("near data points", near)]:
        if not m.any():
            continue
        lines += [f"\n{name}: {m.sum()}",
                  f"{'aperture':>9} {'mean err %':>11} {'med |err| %':>12} {'95% |err| %':>12} "
                  f"{'med |err|/sig_data':>19} {'|pull|<1':>9} {'|pull|<2':>9}"]
        for a, t in enumerate(th):
            lines.append(f"{t:>8g}' {np.mean(rel[m, a]):11.2f} {np.median(np.abs(rel[m, a])):12.2f} "
                         f"{np.percentile(np.abs(rel[m, a]), 95):12.2f} {np.median(np.abs(ed[m, a])):19.3f} "
                         f"{np.mean(pull[m, a] < 1):9.2f} {np.mean(pull[m, a] < 2):9.2f}")
        lines.append(f"{'overall':>9} {np.mean(rel[m]):11.2f} {np.median(np.abs(rel[m])):12.2f} "
                     f"{np.percentile(np.abs(rel[m]), 95):12.2f} {np.median(np.abs(ed[m])):19.3f} "
                     f"{np.mean(pull[m] < 1):9.2f} {np.mean(pull[m] < 2):9.2f}")
    lines.append("\nGaussian reference: |pull|<1 0.68, |pull|<2 0.95")
    return "\n".join(lines)


# LOO residuals per aperture
def plot_loo_residuals(th, truth, mu, sig_data, near, out):
    rel = 100 * (mu - truth) / truth
    ed = (mu - truth) / sig_data
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, v, ylab in [(axes[0], rel, "(predicted − true) / true  [%]"),
                        (axes[1], ed, r"(predicted − true) / $\sigma_{\rm ACT}$")]:
        lo, hi = np.percentile(v, [2, 98])
        pad = 0.15 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        for m, c, a, name in [(~near, MUTED, 0.25, "far from data"), (near, BLUE, 0.8, "near data")]:
            for k, row in enumerate(v[m]):
                ax.plot(th, row, color=c, lw=0.8, alpha=a, label=name if k == 0 else None)
        q = np.percentile(v, [2.5, 16, 50, 84, 97.5], axis=0)
        ax.fill_between(th, q[0], q[4], color=MUTED, alpha=0.10, lw=0, label="2.5 to 97.5%")
        ax.fill_between(th, q[1], q[3], color=MUTED, alpha=0.22, lw=0, label="16 to 84%")
        ax.plot(th, q[2], color=INK, lw=2.2, label="median")
        ax.axhline(0, color=MUTED, lw=0.8)
        n_out = np.sum(np.any((v < lo) | (v > hi), axis=1))
        ax.text(0.02, 0.02, f"{n_out} points go outside this y range", transform=ax.transAxes,
                fontsize=9, color=MUTED)
        ax.set_ylim(lo, hi)
        ax.set_xticks(th)
        ax.set_xticklabels([f"{t:g}′" for t in th])
        ax.set_xlabel("aperture", color=MUTED)
        ax.set_ylabel(ylab, color=INK)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        style(ax)
    axes[1].axhspan(-1, 1, color=BLUE, alpha=0.06, lw=0)
    fig.suptitle(f"Leave one out residuals per aperture ({len(truth)} points)",
                 x=0.005, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.savefig(out, dpi=130)
    plt.close(fig)


# LOO mean prediction vs true value
def plot_loo_pred_vs_true(th, truth, mu, sig, y_act, near, out):
    n = len(th)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4 * nrow), layout="constrained", squeeze=False)
    for a, ax in enumerate(axes.flat):
        if a >= n:
            ax.axis("off")
            continue
        t, m, s = truth[:, a], mu[:, a], sig[:, a]
        lim = np.array([min(t.min(), m.min()), max(t.max(), m.max())]) * np.array([0.7, 1.4])
        ax.plot(lim, lim, color=MUTED, lw=0.8)
        ax.axvline(y_act[a], color=GRID, lw=1.5, zorder=0)
        for mask, c, name in [(~near, MUTED, "far from data"), (near, BLUE, "near data")]:
            ax.errorbar(t[mask], m[mask], yerr=s[mask], fmt="o", ms=3.5, color=c, ecolor=c,
                        elinewidth=0.7, capsize=0, alpha=0.8, label=name)
        rel = np.abs(m - t) / t
        ax.text(0.04, 0.96, f"median |err| {100 * np.median(rel):.2f}%\nnear data {100 * np.median(rel[near]):.2f}%"
                if near.any() else f"median |err| {100 * np.median(rel):.2f}%",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(f"{th[a]:g}′", fontsize=11, color=INK)
        style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("true Y (simulation)", color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("LOO mean prediction", color=MUTED)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Leave one out mean prediction vs true value (bars: ±1σ_GP; grey line: ACT data)",
                 x=0.005, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.savefig(out, dpi=120)
    plt.close(fig)


# recovery summary table
def recovery_summary(rows, names, true, q, label, noise):
    half = 0.5 * (q[:, 3] - q[:, 1])
    z = (q[:, 2] - true) / half
    in68 = (true >= q[:, 1]) & (true <= q[:, 3])
    in95 = (true >= q[:, 0]) & (true <= q[:, 4])
    lines = [f"parameter recovery of held out points, {label}, {'noisy' if noise else 'noiseless'} mock data",
             "each row: emulator trained without that point, its simulated profile fit with the ACT covariance\n"]
    for i, r in enumerate(rows):
        lines.append(f"row {r:3d}  " + "   ".join(
            f"{n} true {true[i, j]:.3f} rec {q[i, 2, j]:.3f} +{q[i, 3, j] - q[i, 2, j]:.3f} "
            f"-{q[i, 2, j] - q[i, 1, j]:.3f} z {z[i, j]:+.2f}" for j, n in enumerate(names)))
    lines.append("\nper parameter: within 68%, within 95%, median |z|, mean z")
    for j, n in enumerate(names):
        lines.append(f"  {n:>8}  {np.mean(in68[:, j]):.2f}  {np.mean(in95[:, j]):.2f}  "
                     f"{np.median(np.abs(z[:, j])):.2f}  {np.mean(z[:, j]):+.2f}")
    lines.append(f"  {'all':>8}  {np.mean(in68):.2f}  {np.mean(in95):.2f}  {np.median(np.abs(z)):.2f}  {np.mean(z):+.2f}")
    lines.append("\nexpected for calibrated posteriors with noisy mocks: 0.68, 0.95, 0.67, 0")
    return "\n".join(lines)


# recovery plot
def plot_recovery(names, bounds, true, q, out):
    nd = len(names)
    fig, axes = plt.subplots(1, nd + 1, figsize=(4.2 * (nd + 1), 4.4), layout="constrained")
    in68 = (true >= q[:, 1]) & (true <= q[:, 3])
    for j, ax in enumerate(axes[:nd]):
        lo, hi = bounds[j]
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8)
        for m, c in [(in68[:, j], BLUE), (~in68[:, j], ORANGE)]:
            ax.errorbar(true[m, j], q[m, 2, j], yerr=[q[m, 2, j] - q[m, 0, j], q[m, 4, j] - q[m, 2, j]],
                        fmt="none", ecolor=c, elinewidth=0.8, alpha=0.5)
            ax.errorbar(true[m, j], q[m, 2, j], yerr=[q[m, 2, j] - q[m, 1, j], q[m, 3, j] - q[m, 2, j]],
                        fmt="o", ms=4, color=c, ecolor=c, elinewidth=2, capsize=0)
        ax.text(0.04, 0.96, f"truth in 68%: {np.mean(in68[:, j]):.0%}", transform=ax.transAxes,
                va="top", fontsize=9, color=INK)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"true {names[j]}", color=MUTED)
        ax.set_ylabel(f"recovered {names[j]}", color=MUTED)
        style(ax)
    ax = axes[nd]
    z = (q[:, 2] - true) / (0.5 * (q[:, 3] - q[:, 1]))
    rng = np.random.default_rng(0)
    for j in range(nd):
        ax.scatter(j + rng.uniform(-0.15, 0.15, len(z)), np.clip(z[:, j], -4, 4), s=18,
                   color=np.where(np.abs(z[:, j]) <= 1, BLUE, ORANGE), edgecolor="white", lw=0.4)
    ax.axhspan(-1, 1, color=BLUE, alpha=0.08, lw=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(nd))
    ax.set_xticklabels(names)
    ax.set_ylim(-4.3, 4.3)
    ax.set_ylabel("(recovered − true) / σ₆₈", color=MUTED)
    style(ax)
    fig.suptitle(f"Recovery of {len(true)} held out points (thick: 68%, thin: 95%; orange: truth outside 68%)",
                 x=0.005, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("training_set", help="stacked LHC profiles (.npz from stack_lrgs.py)")
    ap.add_argument("--exclude-rows", type=int, nargs="*", default=[])
    ap.add_argument("--freq", type=int, default=150, choices=[90, 150])
    ap.add_argument("--tests", nargs="+", default=["loo", "recovery"], choices=["loo", "recovery"])
    ap.add_argument("--recovery-rows", type=int, nargs="*", default=None, help="default: all near data rows")
    ap.add_argument("--near", type=float, default=2.0, help="near data if median(profile/ACT) within this factor")
    ap.add_argument("--noise", action="store_true", help="add ACT covariance noise to the recovery mocks")
    ap.add_argument("--nlive", type=int, default=1000)
    ap.add_argument("--dlogz", type=float, default=0.1)
    ap.add_argument("--slices", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--replot", action="store_true", help="reuse saved results, only redo tables and plots")
    ap.add_argument("--out", default="validation")
    args = ap.parse_args()

    # training set
    data = Path(args.training_set)
    all_params, profiles, params, apertures, names = load_training_set(data, args.exclude_rows)
    row_ids = np.setdiff1d(np.arange(len(all_params)), args.exclude_rows)
    bounds = np.column_stack([all_params.min(0), all_params.max(0)])
    p0 = names.index("log10P0")
    y_act, cov = load_act(args.freq)
    n_ap = len(y_act)
    th = np.asarray(apertures[:n_ap], float)
    truth = profiles[:, :n_ap]
    sig_data = np.sqrt(np.diag(cov))
    ratio = np.median(truth / y_act, axis=1)
    near = (ratio > 1 / args.near) & (ratio < args.near)
    label = f"{data.name} vs ACT f{args.freq} ({len(params)} points, excluded rows {args.exclude_rows})"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{label}\n{near.sum()} near data points\noutput {out}")

    # leave one out
    if "loo" in args.tests:
        f = out / "loo.npz"
        if args.replot and f.exists():
            mu, sig = np.load(f)["mu"], np.load(f)["sig"]
        else:
            print(f"LOO: {len(params)} fits on {args.nproc} processes")
            with Pool(args.nproc, initializer=init_worker, initargs=(params, profiles, p0, None)) as pool:
                res = pool.map(loo_one, range(len(params)), chunksize=1)
            mu = np.array([r[0] for r in res])[:, :n_ap]
            sig = np.array([r[1] for r in res])[:, :n_ap]
            np.savez(f, rows=row_ids, params=params, param_names=names, apertures=th, truth=truth,
                     mu=mu, sig=sig, y_act=y_act, sig_data=sig_data, ratio=ratio, near=near)
        report = loo_summary(th, truth, mu, sig, sig_data, near, label)
        print(report)
        (out / "loo_summary.txt").write_text(report + "\n")
        plot_loo_residuals(th, truth, mu, sig_data, near, out / "loo_residuals_per_aperture.png")
        plot_loo_pred_vs_true(th, truth, mu, sig, y_act, near, out / "loo_pred_vs_true.png")

    # parameter recovery of held out points
    if "recovery" in args.tests:
        f = out / ("recovery_noisy.npz" if args.noise else "recovery.npz")
        if args.replot and f.exists():
            d = np.load(f)
            rows, q = d["rows"], d["quantiles"]
        else:
            rows = np.array(args.recovery_rows if args.recovery_rows is not None else row_ids[near], int)
            missing = np.setdiff1d(rows, row_ids)
            if missing.size:
                raise SystemExit(f"rows {missing.tolist()} are excluded or out of range")
            pos = np.searchsorted(row_ids, rows)
            fit = dict(cov=cov, bounds=bounds, n_ap=n_ap, nlive=args.nlive, dlogz=args.dlogz,
                       slices=args.slices, noise=args.noise, seed=args.seed)
            print(f"recovery: {len(rows)} held out points, nlive {args.nlive}, on {args.nproc} processes")
            with Pool(min(args.nproc, len(rows)), initializer=init_worker,
                      initargs=(params, profiles, p0, fit)) as pool:
                res = pool.map(recover_one, pos, chunksize=1)
            q = np.array([r[0] for r in res])
            np.savez(f, rows=rows, true=params[pos], quantiles=q, theta_map=np.array([r[1] for r in res]),
                     param_names=names, bounds=bounds, noise=args.noise)
        true = params[np.searchsorted(row_ids, rows)]
        report = recovery_summary(rows, names, true, q, label, args.noise)
        print(report)
        tag = "_noisy" if args.noise else ""
        (out / f"recovery_summary{tag}.txt").write_text(report + "\n")
        plot_recovery(names, bounds, true, q, out / f"recovery{tag}.png")

    print(f"saved to {out}")


if __name__ == "__main__":
    main()
