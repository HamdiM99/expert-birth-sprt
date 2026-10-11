"""Regenerate every data-driven figure of the NeurIPS 2026 submission from the
released supplementary data (no GPU, no API calls).

usage: python make_figures.py <supplementary_root> <out_dir> <submission_pdf>

Each figure is written as vector PDF (for LaTeX) + PNG preview. Numbers that the
script recomputes are printed so they can be checked against the paper text.
"""
import json, math, re, sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

SUP, OUT, PDF = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito (validated: CVD-safe with secondary encoding — markers/labels).
BLUE, VERM, GREEN, PINK, ORANGE, GREY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#7F7F7F"
NEED, NONEED = BLUE, VERM            # NeedsBirth / no-need, used consistently
TEXTW = 5.5                           # NeurIPS \textwidth in inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig); print(f"  wrote {name}")


def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def js(p): return json.load(open(p, encoding="utf-8"))


traces = [t for t in jl(SUP / "data/per_proposal_traces.jsonl") if t["uq_observations"]]
elig = [t for t in traces if t["needs_birth_gt"] is not None]          # the 7 eligible
poc = js(SUP / "data/poc_summary.json")
oracle = {p["id"]: p for p in js(SUP / "data/posthoc_counterfactual.json")["proposals"]}
hole5 = js(SUP / "results/HOLE_5_block_bootstrap.json")
hole4 = js(SUP / "results/HOLE_4_uq_calibration.json")
batched = js(SUP / "code/poc_observation/analysis_holes/batched_probe_results.json")
SEEDS = (0, 1, 2)
te = {s: js(SUP / f"results/token_entropy/seed{s}/summary.json") for s in SEEDS}
te_dec = {s: js(SUP / f"results/token_entropy/seed{s}/decisions.json") for s in SEEDS}
te_cf = {s: js(SUP / f"results/token_entropy/seed{s}/counterfactual.json") for s in SEEDS}

ALPHA = BETA = 0.05
A, B = math.log((1 - BETA) / ALPHA), math.log(BETA / (1 - ALPHA))


def llr_path(u, mu0, mu1, sigma):
    u = np.asarray(u)
    return np.cumsum((mu1 - mu0) / sigma**2 * u - (mu1**2 - mu0**2) / (2 * sigma**2))


# ── Figures 1–2: schematic diagrams — no data; vector crop from the PDF ──────
def crop_diagrams():
    import pymupdf
    src = pymupdf.open(PDF)
    for name, page, clip in [("fig01_overview", 1, (105, 70, 505, 296)),
                             ("fig02_pipeline", 2, (115, 85, 505, 172))]:
        r = pymupdf.Rect(*clip)
        out = pymupdf.open(); pg = out.new_page(width=r.width, height=r.height)
        pg.show_pdf_page(pg.rect, src, page, clip=r)
        out.save(OUT / f"{name}.pdf"); pg.get_pixmap(dpi=200).save(OUT / f"{name}.png")
        print(f"  wrote {name} (vector crop of PDF page {page + 1})")


# ── Figure 3: SPRT properties (a) illustrative traces (b) Wald ASN (c) ROC/PR ─
def fig03():
    mu0, mu1, sig, nmax = 0.45, 0.55, 0.20, 50
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 2, figsize=(TEXTW, 2.0))
    for mu, c, lab in [(0.55, NEED, r"$\mu=0.55$ (needs birth)"),
                       (0.45, NONEED, r"$\mu=0.45$ (no need)"),
                       (0.50, GREY, r"$\mu=0.50$ (borderline)")]:
        ax[0].plot(np.arange(1, nmax + 1), llr_path(rng.normal(mu, sig, nmax), mu0, mu1, sig), color=c, label=lab)
    ax[0].axhspan(A, 9, color=NEED, alpha=.07); ax[0].axhspan(-9, B, color=NONEED, alpha=.07)
    ax[0].axhline(A, color=NEED, ls="--", lw=.8); ax[0].axhline(B, color=NONEED, ls="--", lw=.8)
    ax[0].text(1, A + .3, rf"$A=\log\frac{{1-\beta}}{{\alpha}}={A:.2f}$ → BIRTH", fontsize=6.5)
    ax[0].text(1, B - 1.1, rf"$B={B:.2f}$ → SKIP", fontsize=6.5)
    ax[0].set(xlim=(0, nmax), ylim=(-9, 9), xlabel="observation $n$", ylabel=r"log-likelihood ratio $\Lambda_n$",
              title=r"(a) illustrative traces ($\alpha=\beta=0.05$, $\sigma=0.2$)")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=6, handlelength=1.2)

    # Wald ASN with the correct OC exponent h(mu) = (mu0+mu1-2mu)/(mu1-mu0)
    mus = np.linspace(0.30, 0.70, 801); d = mu1 - mu0; mbar = (mu0 + mu1) / 2
    asn = np.empty_like(mus)
    for i, mu in enumerate(mus):
        m = d / sig**2 * (mu - mbar)
        if abs(mu - mbar) < 1e-9:
            asn[i] = -A * B / (d / sig) ** 2
        else:
            h = (mu0 + mu1 - 2 * mu) / d
            L = (math.exp(h * A) - 1) / (math.exp(h * A) - math.exp(h * B))   # P(accept H0)
            asn[i] = (L * B + (1 - L) * A) / m
    ax[1].plot(mus, asn, color="black", label="Wald ASN (no floor/cap)")
    ax[1].plot(mus, np.clip(asn, 20, nmax), color=GREEN, ls="--", label=r"with $n_{\min}=20$, $N_{\max}=50$")
    for v, c in [(mu0, NONEED), (mu1, NEED)]: ax[1].axvline(v, color=c, lw=.7, ls=":")
    ax[1].set(xlabel=r"true per-observation mean $\mu$", ylabel=r"$\mathbb{E}_\mu[N]$", ylim=(0, 55),
              title="(b) expected sample size")
    ax[1].legend(loc="upper right", fontsize=6)
    print(f"  ASN at mu0={np.interp(mu0, mus, asn):.1f}, mid={-A*B/(d/sig)**2:.1f}, mu1={np.interp(mu1, mus, asn):.1f}")
    fig.tight_layout(); save(fig, "fig03ab_sprt_properties")

    # (c) proposal-level discrimination vs proxy label AND vs oracle label
    y_proxy = np.array([t["needs_birth_gt"] for t in elig], int)
    y_orc = np.array([oracle[t["id"]]["posthoc_decision"] for t in elig], int)
    stats = {"mean (all obs.)": [np.mean(t["uq_observations"]) for t in elig],
             "max (all obs.)": [np.max(t["uq_observations"]) for t in elig],
             "mean (first 20)": [np.mean(t["uq_observations"][:20]) for t in elig]}
    fig, ax = plt.subplots(1, 2, figsize=(TEXTW, 1.9))
    for a, y, ttl in [(ax[0], y_proxy, "vs proxy label (mean UQ ≥ 0.65)"), (ax[1], y_orc, "vs counterfactual-oracle label")]:
        for (k, s), c, mk in zip(stats.items(), (BLUE, ORANGE, GREEN), "os^"):
            fpr, tpr, auc = roc(np.array(s), y)
            a.plot(fpr, tpr, color=c, marker=mk, ms=3, label=f"{k}: AUROC={auc:.2f}", drawstyle="steps-post")
            print(f"  ROC {ttl[:9]} {k}: AUROC={auc:.3f}")
        a.plot([0, 1], [0, 1], color=GREY, ls=":", lw=.8)
        a.set(xlabel="FPR", ylabel="TPR", title=f"(c) {ttl}", xlim=(-.03, 1.03), ylim=(-.03, 1.03))
        a.legend(loc="lower right", fontsize=5.8)
    fig.tight_layout(); save(fig, "fig03c_uq_roc")


def roc(score, y):
    order = np.argsort(-score); y = y[order]
    P, N = y.sum(), len(y) - y.sum()
    tpr = np.concatenate([[0], np.cumsum(y) / max(P, 1)]); fpr = np.concatenate([[0], np.cumsum(1 - y) / max(N, 1)])
    # AUROC via Mann-Whitney (ties count 1/2)
    pos, neg = score[y == 1] if False else None, None
    s = np.asarray(score)[order]; pos = s[y == 1]; neg = s[y == 0]
    auc = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]) if len(pos) and len(neg) else float("nan")
    return fpr, tpr, auc


# ── Figure 4a: real LLR trajectory, proposal #2 ──────────────────────────────
def fig04a():
    t = next(x for x in traces if x["id"] == 2)
    mu0, mu1, sig, nmin, nmax = 0.929, 1.145, 0.20, 20, 50
    L = llr_path(t["uq_observations"], mu0, mu1, sig); n = np.arange(1, len(L) + 1)
    fig, ax = plt.subplots(figsize=(TEXTW / 2, 2.0))
    ax.plot(n[:nmin], L[:nmin], color=BLUE, label="LLR, proposal #2 (medical)")
    ax.plot(n[nmin - 1:], L[nmin - 1:], color=BLUE, ls="--", alpha=.5, label="after decision (reference)")
    ax.axhline(A, color=NEED, ls="--", lw=.8); ax.axhline(B, color=NONEED, ls="--", lw=.8)
    ax.axvline(nmin, color=GREY, ls=":", lw=.8); ax.axvline(nmax, color=GREY, ls=":", lw=.8)
    ax.plot(nmin, L[nmin - 1], "o", color=ORANGE, zorder=5)
    ax.annotate(rf"BIRTH at $n={nmin}$ ($\Lambda_{{20}}={L[nmin-1]:.1f}$)", (nmin, L[nmin - 1]), (23, L[nmin - 1] - 22),
                fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=.6))
    ax.text(nmin + .5, 66, r"$n_{\min}$", fontsize=6.5); ax.text(nmax - 5, 40, r"$N_{\max}$", fontsize=6.5)
    ax.set(xlabel="observation $n$", ylabel=r"$\Lambda_n$", title=rf"(a) $|C|={len(L)}$, $\bar u_C={np.mean(t['uq_observations']):.2f}$")
    ax.legend(loc="lower right", fontsize=6)
    print(f"  prop#2: Lambda_20={L[nmin-1]:.1f}, Lambda_final={L[-1]:.1f}, first n with Lambda>=A: {int(np.argmax(L >= A)) + 1}")
    save(fig, "fig04a_llr_prop2")


# ── Figure 7 (= Fig. 4b): deployment threshold sweep, split into 2 panels ────
def fig07():
    sw = poc["threshold_sweep"]; th = [s["threshold"] for s in sw]
    fig, ax = plt.subplots(1, 2, figsize=(TEXTW, 1.9), sharex=True)
    ax[0].plot(th, [s["n_births"] for s in sw], "o-", color=BLUE, label="observation gate")
    ax[0].axhline(poc["confusion_eager"]["n_births"], color=GREY, ls="--", label="eager baseline")
    ax[0].set(ylabel="# LoRA trainings", xlabel=r"UQ gate threshold $\tau_u$", title="(a) training cost", ylim=(0, 8))
    ax[1].plot(th, [s["recall"] for s in sw], "s-", color=GREEN, label="recall")
    ax[1].plot(th, [s["precision"] for s in sw], "^-", color=ORANGE, label="precision")
    ax[1].set(ylabel="score (proxy label)", xlabel=r"UQ gate threshold $\tau_u$", title="(b) decision quality", ylim=(0, 1.05))
    for a in ax: a.axvline(0.55, color=GREY, ls=":", lw=.8)
    ax[0].legend(loc="lower right"); ax[1].legend(loc="lower left")
    fig.tight_layout(); save(fig, "fig07_deploy_sweep")


# ── Figure 5a: recall vs cost; 5c: running means of deployment traces ────────
def fig05():
    fig, ax = plt.subplots(figsize=(TEXTW / 2, 2.0))
    sw = poc["threshold_sweep"]
    ax.plot([s["n_births"] for s in sw], [s["recall"] for s in sw], "o-", color=GREEN, label="observation, deployment")
    th = [t["threshold"] for t in te[0]["threshold_sweep"]]
    nb = [np.mean([te[s]["threshold_sweep"][i]["n_births"] for s in SEEDS]) for i in range(len(th))]
    rc = [np.mean([te[s]["threshold_sweep"][i]["recall"] for s in SEEDS]) for i in range(len(th))]
    ax.plot(nb, rc, "s-", color=BLUE, label="observation, controlled (3 seeds)")
    ax.plot(poc["confusion_eager"]["n_births"], poc["confusion_eager"]["recall"], "X", ms=7, color=VERM, label="eager")
    ce = np.mean([te[s]["confusion_eager"]["n_births"] for s in SEEDS])
    ax.plot(ce, np.mean([te[s]["confusion_eager"]["recall"] for s in SEEDS]), "X", ms=7, color=VERM)
    ax.set(xlabel="# LoRA trainings (cost)", ylabel="recall", ylim=(0, 1.05), xlim=(0, 8), title="(a) recall vs. cost")
    ax.legend(loc="upper left", fontsize=6)
    save(fig, "fig05a_recall_cost")

    fig, ax = plt.subplots(figsize=(TEXTW / 2, 2.0))
    col = {"APPROVED": BLUE, "ABANDONED": VERM, "PROPOSED": GREY}
    for t in elig:
        u = np.asarray(t["uq_observations"]); rm = np.cumsum(u) / np.arange(1, len(u) + 1)
        ax.plot(np.arange(1, len(u) + 1), rm, color=col[t["status"]], lw=1)
        ax.text(len(u) + 1, rm[-1], f"#{t['id']}", fontsize=5.5, va="center")
    for k, c in col.items(): ax.plot([], [], color=c, label=k.lower())
    ax.axhline(0.55, color="black", ls="--", lw=.7, label=r"$\tau_u=0.55$")
    ax.set(xlabel="observation $n$", ylabel="running mean UQ", title="(c) deployment traces")
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.08, 0.5))
    save(fig, "fig05c_running_mean")


# ── Figure 6: synthetic workload composition ──────────────────────────────────
def fig06():
    blocks = ["true_need_med", "true_need_legal", "true_need_code", "trap_a", "trap_b", "noise"]
    labels = ["Med\n(true need)", "Legal FR\n(true need)", "Code\n(true need)", "Trap A\n(physics)", "Trap B\n(legal)", "Noise\n(generic)"]
    cnt = {s: Counter(r["block"] for r in jl(SUP / f"code/poc_observation/data/workload_seed{s}.jsonl")) for s in SEEDS}
    print("  workload per seed:", {s: dict(cnt[s]) for s in SEEDS})
    found = sorted(set().union(*[set(c) for c in cnt.values()]))
    blocks = [b for b in blocks if b in found] + [b for b in found if b not in blocks]
    vals = [np.mean([cnt[s][b] for s in SEEDS]) for b in blocks]
    colors = [NEED if b.startswith("true_need") else (ORANGE if b.startswith("trap") else GREY) for b in blocks]
    fig, ax = plt.subplots(figsize=(TEXTW * .75, 1.9))
    bars = ax.bar(range(len(blocks)), vals, color=colors, width=.7, edgecolor="white", linewidth=1)
    for b, v, bl in zip(bars, vals, blocks):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v:.0f}", ha="center", fontsize=6.5)
    lab = {b: l for b, l in zip(["true_need_med", "true_need_legal", "true_need_code", "trap_a", "trap_b", "noise"], labels)}
    ax.set_xticks(range(len(blocks)), [lab.get(b, b) for b in blocks], fontsize=6.5)
    ax.set(ylabel="queries per seed", title=f"Synthetic workload ({sum(vals):.0f} queries/seed, 3 seeds)")
    ax.grid(axis="x", visible=False)
    save(fig, "fig06_workload")


# ── Figure 8: candidate-cluster space (controlled, 3 seeds) ──────────────────
def fig08():
    fig, ax = plt.subplots(figsize=(TEXTW * .7, 2.2))
    for s in SEEDS:
        blk = {c["cluster_id"]: c["dominant_block"].replace("true_need_", "") for c in te_cf[s]}
        gt = {int(k): v for k, v in te[s]["ground_truth"].items()}
        for e in te_dec[s]["eager"]:
            m = re.search(r"size=(\d+).*mean_uq=([\d.]+)", e["reason"])
            size, mu = int(m.group(1)), float(m.group(2)); g = gt.get(e["cluster_id"])
            ax.scatter(size, mu, s=22, color=NEED if g else NONEED, marker="o" if g else "s",
                       edgecolor="white", linewidth=.6, zorder=3)
            ax.annotate(f"{blk.get(e['cluster_id'], '?')}·s{s}", (size, mu), (3, 2), textcoords="offset points", fontsize=5.5)
    ax.axhline(0.55, color="black", ls="--", lw=.7, label=r"$\tau_u=0.55$")
    ax.axvline(15, color=GREY, ls=":", lw=.8, label=r"$N_C=15$")
    ax.scatter([], [], color=NEED, marker="o", label=r"NeedsBirth ($\tau_\Delta=0.03$)")
    ax.scatter([], [], color=NONEED, marker="s", label="no need")
    ax.set(xscale="log", xlabel="cluster size $|C|$ (log)", ylabel=r"mean cluster UQ $\bar u_C$",
           title="Candidate clusters, controlled experiment (token entropy)")
    ax.legend(fontsize=6, loc="best")
    save(fig, "fig08_cluster_space")


# ── Figure 9: sequential vs batched probe vs eager (deployment) ──────────────
def fig09():
    fig, ax = plt.subplots(figsize=(TEXTW * .6, 2.0))
    bp = batched["batched_probe"]
    groups = {}
    for k, v in bp.items(): groups.setdefault((v["n_births"], v["recall"]), []).append(k.replace("K=", ""))
    for (x, y), ks in groups.items():
        ax.plot(x, y, "s", color=BLUE, ms=6); ax.annotate(f"batched $K\\in\\{{{','.join(ks)}\\}}$", (x, y), (5, -9), textcoords="offset points", fontsize=6)
    sp = batched["sprt"]; ax.plot(sp["n_births"], sp["recall"], "o", mfc="none", mec=GREEN, mew=1.5, ms=10,
                                  label=f"SPRT (mean $N$={batched['sprt_avg_obs']:.1f})")
    eg = batched["eager"]; ax.plot(eg["n_births"], eg["recall"], "X", color=VERM, ms=8, label="eager")
    ax.plot([], [], "s", color=BLUE, label="batched probe")
    ax.set(xlabel="# LoRA trainings", ylabel="recall (proxy label)", xlim=(2, 8), ylim=(0, 1.08),
           title="Deployment, 7 eligible proposals")
    ax.legend(loc="lower right", fontsize=6)
    save(fig, "fig09_batched_vs_sprt")


# ── Figure 10 (replacement): real per-observation multi-UQ on deployment ─────
def fig10():
    rows = jl(SUP / "data/multi_uq_observations.jsonl")
    sigs = ["token_entropy", "semantic_entropy", "expert_disagreement", "backbone_agreement"]
    ids = sorted({r["proposal_id"] for r in rows})
    proxy = {t["id"]: t["needs_birth_gt"] for t in elig}
    fig, ax = plt.subplots(1, 4, figsize=(TEXTW, 1.9), sharex=True)
    rng = np.random.default_rng(1)
    for a, sg in zip(ax, sigs):
        for i, pid in enumerate(ids):
            v = np.array([r["signals"][sg] for r in rows if r["proposal_id"] == pid])
            c = NEED if proxy.get(pid) else NONEED
            a.scatter(i + rng.uniform(-.18, .18, len(v)), v, s=4, color=c, alpha=.55, linewidth=0)
            a.plot([i - .3, i + .3], [v.mean()] * 2, color="black", lw=1)
        a.set_title(sg.replace("_", " "), fontsize=7.5)
        a.set_xticks(range(len(ids)), [f"#{p}\n{'O+' if oracle[p]['posthoc_decision'] else 'O−'}" for p in ids], fontsize=5.5)
    ax[0].set_ylabel("per-query value")
    ax[0].scatter([], [], color=NEED, label="proxy: needs birth"); ax[0].scatter([], [], color=NONEED, label="proxy: no need")
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.08), fontsize=6.5)
    fig.text(.5, -.04, "proposal id; O+/O− = counterfactual-oracle label; black bar = mean", ha="center", fontsize=6.5)
    fig.tight_layout(); save(fig, "fig10_multi_uq_real")
    print(f"  multi-UQ: {len(rows)} observations over proposals {ids}")


# ── Figures 11–13: Gaussianity, autocorrelation, block bootstrap ─────────────
def fig11_13():
    items = [t for t in traces if len(t["uq_observations"]) >= 10][:6]
    fig, axes = plt.subplots(2, 3, figsize=(TEXTW, 3.2))
    for a, t in zip(axes.flat, items):
        u = np.sort(t["uq_observations"]); q = norm.ppf((np.arange(1, len(u) + 1) - .5) / len(u))
        a.scatter(q, u, s=5, color=BLUE, linewidth=0); m, b = np.polyfit(q, u, 1)
        a.plot(q, m * q + b, color="black", lw=.8)
        a.set_title(f"#{t['id']} {t['status'].lower()}, $n$={len(u)}", fontsize=7)
    for a in axes[1]: a.set_xlabel("N(0,1) quantile")
    for a in axes[:, 0]: a.set_ylabel("observed UQ")
    fig.tight_layout(); save(fig, "fig11_qq")

    ac = hole5["autocorr_observed"]["per_proposal"]
    fig, ax = plt.subplots(figsize=(TEXTW * .55, 1.7))
    ax.hist(ac, bins=np.linspace(0, 1, 11), color=VERM, edgecolor="white")
    ax.axvline(np.mean(ac), color="black", ls="--", lw=.8, label=rf"mean $\bar\rho={np.mean(ac):.2f}$")
    ax.set(xlabel="lag-1 autocorrelation of UQ within a proposal", ylabel="# proposals", yticks=[0, 1, 2, 3]); ax.legend(loc="upper left")
    save(fig, "fig12_autocorr")

    bs = hole5["block_sizes"]
    fig, ax = plt.subplots(figsize=(TEXTW * .55, 1.8))
    ax.plot(bs, [hole5["type_I_per_block"][str(b)] for b in bs], "o-", color=VERM, label="empirical type-I")
    ax.plot(bs, [hole5["type_II_per_block"][str(b)] for b in bs], "s-", color=BLUE, label="empirical type-II")
    ax.axhline(0.05, color="black", ls="--", lw=.8, label=r"nominal $\alpha=\beta=0.05$")
    ax.set(xlabel="circular block size $b$ ($b=1$: i.i.d. shuffle)", ylabel="error rate", xticks=bs, ylim=(-.005, .1))
    ax.legend(fontsize=6)
    save(fig, "fig13_block_bootstrap")


# ── Figure 14: reliability (raw min-max vs isotonic LOO) ─────────────────────
def fig14():
    y = np.array(hole4["labels"]); raw = np.array(hole4["raw_predictions"]); iso = np.array(hole4["isotonic_predictions"])
    fig, ax = plt.subplots(figsize=(TEXTW * .45, 2.1))
    ax.plot([0, 1], [0, 1], color=GREY, ls=":", lw=.8)
    rng = np.random.default_rng(2); jit = rng.uniform(-.03, .03, len(y))
    ax.scatter(raw, y + jit, color=VERM, marker="s", s=16, label=f"raw min-max (Brier {hole4['brier_uncalibrated_raw_minmax']:.3f})")
    ax.scatter(iso, y - jit, color=BLUE, marker="o", s=16, label=f"isotonic LOO (Brier {hole4['brier_isotonic_LOO']:.3f})")
    ax.set(xlabel="predicted P(NeedsBirth)", ylabel="proxy label", yticks=[0, 1], xlim=(-.05, 1.05), ylim=(-.15, 1.15))
    ax.legend(fontsize=5.8, loc="center right")
    b_raw, b_iso = np.mean((raw - y) ** 2), np.mean((iso - y) ** 2)
    print(f"  Brier recomputed raw={b_raw:.3f} iso={b_iso:.3f}")
    save(fig, "fig14_calibration")


# ── Figure 15: deployment confusion matrices ─────────────────────────────────
def fig15():
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW * .7, 1.9))
    for a, key, ttl in [(axes[0], "confusion_eager", "Eager"), (axes[1], "confusion_observation_at_0_55", r"Observation ($\tau_u=0.55$)")]:
        c = poc[key]; M = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        a.imshow(M, cmap="Blues", vmin=0, vmax=max(6, M.max()))
        for i in range(2):
            for j in range(2):
                a.text(j, i, M[i, j], ha="center", va="center", fontsize=9, color="white" if M[i, j] > 3 else "black")
        a.set_xticks([0, 1], ["pred. skip", "pred. birth"]); a.set_yticks([0, 1], ["no need", "needs birth"])
        a.set_title(f"{ttl}\nP={c['precision']:.2f} R={c['recall']:.2f} F1={c['f1']:.2f}", fontsize=7.5); a.grid(False)
        print(f"  {ttl}: n={M.sum()} {c}")
    fig.tight_layout(); save(fig, "fig15_confusion")


if __name__ == "__main__":
    for f in (crop_diagrams, fig03, fig04a, fig05, fig06, fig07, fig08, fig09, fig10, fig11_13, fig14, fig15):
        print(f.__name__); f()
