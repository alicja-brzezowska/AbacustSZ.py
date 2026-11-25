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

plt.rcParams.update({"font.size": 11, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True})
LABELS = {"alpha": r"$\alpha$", "beta0": r"$\beta_0$", "gamma": r"$\gamma$", "log10P0": r"$\log_{10} P_0$"}

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


# LOO summary table
def loo_summary(th, truth, mu, sig, sig_data, near, ratio, rows, params, names, label):
    rel = 100 * (mu - truth) / truth
    ed = (mu - truth) / sig_data
    pull = np.abs((mu - truth) / sig)
    lines = [f"leave one out test of {label}"]
    for name, m in [("all points", np.ones(len(truth), bool)), ("near data points", near)]:
        if not m.any():
            continue
        lines += [f"\n{name}: {m.sum()}",
                  f"{'aperture':>9} {'mean err %':>11} {'med |err| %':>12} {'95% |err| %':>12} "
                  f"{'max |err| %':>12} {'med |err|/sig_data':>19} {'|pull|<1':>9} {'|pull|<2':>9}"]
        for a, t in enumerate(th):
            lines.append(f"{t:>8g}' {np.mean(rel[m, a]):11.2f} {np.median(np.abs(rel[m, a])):12.2f} "
                         f"{np.percentile(np.abs(rel[m, a]), 95):12.2f} {np.abs(rel[m, a]).max():12.2f} "
                         f"{np.median(np.abs(ed[m, a])):19.3f} "
                         f"{np.mean(pull[m, a] < 1):9.2f} {np.mean(pull[m, a] < 2):9.2f}")
        lines.append(f"{'overall':>9} {np.mean(rel[m]):11.2f} {np.median(np.abs(rel[m])):12.2f} "
                     f"{np.percentile(np.abs(rel[m]), 95):12.2f} {np.abs(rel[m]).max():12.2f} "
                     f"{np.median(np.abs(ed[m])):19.3f} "
                     f"{np.mean(pull[m] < 1):9.2f} {np.mean(pull[m] < 2):9.2f}")
    lines.append("\nGaussian reference: |pull|<1 0.68, |pull|<2 0.95")

    # largest percentage errors
    worst_ap = np.abs(rel).argmax(1)
    worst = np.abs(rel)[np.arange(len(rel)), worst_ap]
    lines += ["\nlargest percentage errors (each point at its worst aperture)",
              f"{'row':>4} {'err %':>8} {'aperture':>9} {'err/sig_data':>13} {'Y/ACT':>7} {'near':>5}  parameters"]
    for k in np.argsort(worst)[::-1][:5]:
        a = worst_ap[k]
        lines.append(f"{rows[k]:4d} {rel[k, a]:+8.2f} {th[a]:>8g}' {ed[k, a]:+13.2f} {ratio[k]:7.2f} "
                     f"{'yes' if near[k] else 'no':>5}  "
                     + " ".join(f"{n}={v:.3f}" for n, v in zip(names, params[k])))
    lines.append(f"\nworst aperture error per point: median {np.median(worst):.2f}%, "
                 f"{np.mean(worst > 1):.0%} of points above 1%, {np.mean(worst > 5):.0%} above 5%")
    if near.any():
        lines.append(f"largest error of any near data point: {worst[near].max():.2f}% (row {rows[near][worst[near].argmax()]})")
    return "\n".join(lines)


# LOO residuals per aperture
def plot_loo_residuals(th, truth, mu, out):
    rel = 100 * (mu - truth) / truth
    lim = np.ceil(np.percentile(np.abs(rel), 98))
    q = np.percentile(rel, [16, 84], axis=0)
    fig, ax = plt.subplots(figsize=(5.5, 3.8), layout="constrained")
    ax.plot(th, rel.T, color="C0", lw=0.6, alpha=0.3)
    ax.plot(th, q[0], color="k", lw=1.4, ls="--", label="16th and 84th percentile")
    ax.plot(th, q[1], color="k", lw=1.4, ls="--")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlim(th[0], th[-1])
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"aperture $\theta$ [arcmin]")
    ax.set_ylabel(r"$(Y_{\rm emu} - Y_{\rm sim}) \, / \, Y_{\rm sim}$  [%]")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"{np.sum(np.any(np.abs(rel) > lim, axis=1))} of {len(rel)} points go outside ±{lim:g}%")


# LOO mean prediction vs true value
def plot_loo_pred_vs_true(th, truth, mu, sig, y_act, near, out):
    ncol = 3
    nrow = int(np.ceil(len(th) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.3 * nrow), layout="constrained", squeeze=False)
    for a, ax in enumerate(axes.flat):
        if a >= len(th):
            ax.axis("off")
            continue
        t, m, s = truth[:, a], mu[:, a], sig[:, a]
        lim = np.array([min(t.min(), m.min()), max(t.max(), m.max())]) * np.array([0.7, 1.4])
        ax.plot(lim, lim, color="0.5", lw=0.8, zorder=0)
        ax.axvline(y_act[a], color="0.5", lw=0.8, ls=":", zorder=0)
        for mask, c, name in [(~near, "0.6", "far from data"), (near, "C0", "near data")]:
            ax.errorbar(t[mask], m[mask], yerr=s[mask], fmt="o", ms=3, color=c, elinewidth=0.7,
                        capsize=0, label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_box_aspect(1)
        ax.set_title(rf"$\theta = {th[a]:g}'$", fontsize=11)
    for ax in axes[-1]:
        ax.set_xlabel(r"$Y_{\rm sim}$ [arcmin$^2$]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$Y_{\rm emu}$ [arcmin$^2$]")
    axes[0, 0].legend(frameon=False, loc="lower right", fontsize=9)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
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
    ncol = 2
    nrow = int(np.ceil(nd / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7, 3.4 * nrow), layout="constrained", squeeze=False)
    for j, ax in enumerate(axes.flat):
        if j >= nd:
            ax.axis("off")
            continue
        lo, hi = bounds[j]
        med = q[:, 2, j]
        ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, zorder=0)
        ax.errorbar(true[:, j], med, yerr=[med - q[:, 0, j], q[:, 4, j] - med], fmt="none",
                    ecolor="C0", elinewidth=0.8, alpha=0.45)
        ax.errorbar(true[:, j], med, yerr=[med - q[:, 1, j], q[:, 3, j] - med], fmt="o", ms=4,
                    color="C0", elinewidth=2, capsize=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_box_aspect(1)
        lab = LABELS.get(names[j], names[j])
        ax.set_xlabel(f"true {lab}")
        ax.set_ylabel(f"recovered {lab}")
    handles = [plt.Line2D([], [], color="C0", lw=2, marker="o", ms=4, label="median and 68%"),
               plt.Line2D([], [], color="C0", lw=0.8, alpha=0.45, label="95%")]
    fig.legend(handles=handles, frameon=False, loc="outside upper center", ncol=2, fontsize=10)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
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
        report = loo_summary(th, truth, mu, sig, sig_data, near, ratio, row_ids, params, names, label)
        print(report)
        (out / "loo_summary.txt").write_text(report + "\n")
        plot_loo_residuals(th, truth, mu, out / "loo_residuals_per_aperture")
        plot_loo_pred_vs_true(th, truth, mu, sig, y_act, near, out / "loo_pred_vs_true")

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
        plot_recovery(names, bounds, true, q, out / f"recovery{tag}")

    print(f"saved to {out}")


if __name__ == "__main__":
    main()
