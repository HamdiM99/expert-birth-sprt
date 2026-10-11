"""HOLE 26 (local scorer) — score MASS-style gradient surge vs the post-hoc
oracle, and compare with token-entropy SPRT (deployment) and multi-UQ
composite (HOLE 24).

Inputs:
  results/HOLE_26_mass_gradient_surge.json
  data/posthoc_counterfactual.json
  results/HOLE_24_calibrated_uq_oracle.json

Outputs:
  results/HOLE_26_mass_oracle_score.json
  results/HOLE_26_mass_oracle_table.tex
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT   = Path(__file__).resolve().parents[3]
RESULT = ROOT / "results"
DATA   = ROOT / "data"
SURGE  = RESULT / "HOLE_26_mass_gradient_surge.json"
ORACLE = DATA   / "posthoc_counterfactual.json"
HOLE24 = RESULT / "HOLE_24_calibrated_uq_oracle.json"
OUT_J  = RESULT / "HOLE_26_mass_oracle_score.json"
OUT_T  = RESULT / "HOLE_26_mass_oracle_table.tex"


def score(decisions, oracle_pos):
    pids = sorted(decisions)
    tp = fp = fn = tn = 0
    rows = []
    for p in pids:
        gate = (decisions[p] == "BIRTH")
        truth = (p in oracle_pos)
        cell = ("TP" if gate and truth else "FP" if gate and not truth
                else "FN" if not gate and truth else "TN")
        rows.append({"pid": p, "gate": decisions[p],
                     "oracle": "BIRTH" if truth else "SKIP", "cell": cell})
        tp += cell == "TP"; fp += cell == "FP"; fn += cell == "FN"; tn += cell == "TN"
    n = tp+fp+fn+tn
    p_  = tp/(tp+fp) if tp+fp else 0.0
    r_  = tp/(tp+fn) if tp+fn else 0.0
    f1  = 2*p_*r_/(p_+r_) if p_+r_ else 0.0
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":p_,"recall":r_,
            "f1":f1,"agreement":tp+tn,"n":n,"matrix":rows}


def main():
    surge_data = json.load(open(SURGE))
    oracle = json.load(open(ORACLE))
    oracle_pos = {p["id"] for p in oracle["proposals"] if p["posthoc_decision"]}
    print(f"Oracle BIRTH proposals: {sorted(oracle_pos)}")

    # Surge per proposal
    surge = {p["proposal_id"]: p["grad_surge"] for p in surge_data["proposals"]
             if "grad_surge" in p}
    pids = sorted(surge)
    print("\nGradient surge by proposal:")
    for p in pids:
        truth = "BIRTH" if p in oracle_pos else "SKIP"
        print(f"  P{p}: surge={surge[p]:>6.2f}  oracle={truth}")

    # Threshold sweep
    surge_vals = sorted(surge.values())
    thresholds = sorted(set([round(t, 1) for t in
        np.linspace(min(surge_vals)*0.95, max(surge_vals)*1.05, 12)]))
    sweep = []
    for th in thresholds:
        decs = {p: ("BIRTH" if surge[p] >= th else "SKIP") for p in pids}
        s = score(decs, oracle_pos)
        s["threshold"] = th
        sweep.append(s)

    # Top-K (k = 3 matches the count of oracle BIRTHs)
    sorted_by_surge = sorted(pids, key=lambda p: -surge[p])
    topk_results = []
    for k in (2, 3, 4):
        birth = set(sorted_by_surge[:k])
        decs = {p: ("BIRTH" if p in birth else "SKIP") for p in pids}
        s = score(decs, oracle_pos)
        s["topk"] = k
        topk_results.append(s)

    # Best by agreement
    best_threshold = max(sweep, key=lambda x: x["agreement"])
    best_topk = max(topk_results, key=lambda x: x["agreement"])
    print(f"\nBest threshold: th={best_threshold['threshold']:.1f}  "
          f"agreement={best_threshold['agreement']}/7  "
          f"P={best_threshold['precision']:.2f} R={best_threshold['recall']:.2f} "
          f"F1={best_threshold['f1']:.2f}")
    print(f"Best top-K: k={best_topk['topk']}  "
          f"agreement={best_topk['agreement']}/7  "
          f"P={best_topk['precision']:.2f} R={best_topk['recall']:.2f} "
          f"F1={best_topk['f1']:.2f}")

    # Compare with HOLE 24 best (multi-UQ ceiling)
    h24 = json.load(open(HOLE24))
    h24_best = max(h24["configs"], key=lambda c: c["agreement"])
    print(f"\nHOLE 24 (multi-UQ best): {h24_best['config']}  "
          f"agreement={h24_best['agreement']}/7  F1={h24_best['f1']:.2f}")

    # Combined: MASS BIRTH ∩ multi-UQ BIRTH (must be confirmed by both)
    h24_decs = h24_best["decisions"]
    mass_decs = {p: ("BIRTH" if surge[p] >= best_threshold["threshold"] else "SKIP")
                 for p in pids}
    combined = {}
    for p in pids:
        sp = str(p)
        h24_b = (h24_decs.get(sp, h24_decs.get(p, "SKIP")) == "BIRTH")
        mass_b = (mass_decs[p] == "BIRTH")
        combined[p] = "BIRTH" if (h24_b and mass_b) else "SKIP"
    s_comb = score(combined, oracle_pos)
    s_comb["config"] = "multi-UQ AND MASS-surge"
    print(f"\nCombined (multi-UQ ∧ MASS): agreement={s_comb['agreement']}/7  "
          f"P={s_comb['precision']:.2f} R={s_comb['recall']:.2f} F1={s_comb['f1']:.2f}")

    # Union (BIRTH if either signal fires)
    union = {}
    for p in pids:
        sp = str(p)
        h24_b = (h24_decs.get(sp, h24_decs.get(p, "SKIP")) == "BIRTH")
        mass_b = (mass_decs[p] == "BIRTH")
        union[p] = "BIRTH" if (h24_b or mass_b) else "SKIP"
    s_union = score(union, oracle_pos)
    s_union["config"] = "multi-UQ OR MASS-surge"
    print(f"Union (multi-UQ ∨ MASS):     agreement={s_union['agreement']}/7  "
          f"F1={s_union['f1']:.2f}")

    out = {
        "oracle_birth_proposals": sorted(oracle_pos),
        "surge_by_proposal": surge,
        "threshold_sweep": sweep,
        "topk_results": topk_results,
        "best_threshold": best_threshold,
        "best_topk":      best_topk,
        "hole24_best":    {"config": h24_best["config"],
                           "agreement": h24_best["agreement"],
                           "f1": h24_best["f1"]},
        "combined_AND":   s_comb,
        "combined_OR":    s_union,
    }
    json.dump(out, open(OUT_J, "w"), indent=2)
    print(f"\nwrote {OUT_J}")

    # LaTeX table
    rows = ["\\begin{tabular}{lcccccc}",
            "\\toprule",
            "Birth signal & Agreement & TP & FP & FN & TN & F1 \\\\",
            "\\midrule",
            f"Token entropy (deployment) & 2/7 & 1 & 3 & 2 & 1 & 0.29 \\\\",
            f"Multi-UQ composite (HOLE 24) & {h24_best['agreement']}/7 & "
            f"{h24_best['tp']} & {h24_best['fp']} & {h24_best['fn']} & "
            f"{h24_best['tn']} & {h24_best['f1']:.2f} \\\\",
            f"MASS gradient surge (HOLE 26) & {best_threshold['agreement']}/7 & "
            f"{best_threshold['tp']} & {best_threshold['fp']} & "
            f"{best_threshold['fn']} & {best_threshold['tn']} & "
            f"{best_threshold['f1']:.2f} \\\\",
            f"multi-UQ $\\wedge$ MASS & {s_comb['agreement']}/7 & "
            f"{s_comb['tp']} & {s_comb['fp']} & {s_comb['fn']} & "
            f"{s_comb['tn']} & {s_comb['f1']:.2f} \\\\",
            "\\bottomrule",
            "\\end{tabular}"]
    OUT_T.write_text("\n".join(rows))
    print(f"wrote {OUT_T}")


if __name__ == "__main__":
    main()
