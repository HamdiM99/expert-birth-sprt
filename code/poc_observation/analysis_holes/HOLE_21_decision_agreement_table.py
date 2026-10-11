"""HOLE 21 final — produce the paper-ready decision-agreement table."""
import json, math
from pathlib import Path
import numpy as np

ANALYSIS = Path("../../../data")
INP = ANALYSIS / "multi_uq_observations.jsonl"
RES_JSON = Path("../../../results/HOLE_21_uq_signal_robustness.json")
TABLE_OUT = Path("../../../results/HOLE_21_decision_agreement_table.tex")

r = json.loads(RES_JSON.read_text())
decs = {}; oracle = {}; proxy = {}
for sig, res in r["per_signal"].items():
    if "skipped" in res: continue
    decs[sig] = {row["id"]: row["glr"] for row in res["rows"]}
    for row in res["rows"]:
        oracle[row["id"]] = row["oracle"]
        proxy[row["id"]]  = row["proxy"]

# ordering: entropy family first, then disagreement signals, then composite
order = ["token_entropy", "semantic_entropy", "expert_disagreement",
         "backbone_agreement", "composite_uq"]
sigs = [s for s in order if s in decs]
pids = sorted(oracle.keys())

# pairwise agreement
agree = {(s1, s2): sum(1 for pid in pids if decs[s1][pid] == decs[s2][pid])
         for s1 in sigs for s2 in sigs}

# write a LaTeX table
LABEL = {
    "token_entropy": "Token entropy",
    "semantic_entropy": "Semantic entropy",
    "expert_disagreement": "Expert disagreement",
    "backbone_agreement": "Backbone agreement",
    "composite_uq": "Composite UQ",
}
SHORT = {"token_entropy":"TE","semantic_entropy":"SE",
         "expert_disagreement":"ED","backbone_agreement":"BA",
         "composite_uq":"Comp."}

lines = []
lines.append(r"\begin{tabular}{l " + " ".join(["c"]*len(sigs)) + "}")
lines.append(r"\toprule")
lines.append(" & " + " & ".join(SHORT[s] for s in sigs) + r" \\")
lines.append(r"\midrule")
for s1 in sigs:
    cells = []
    for s2 in sigs:
        v = agree[(s1, s2)]
        if s1 == s2:
            cells.append(r"\textemdash")
        else:
            cells.append(f"{v}/7")
    lines.append(LABEL[s1] + " & " + " & ".join(cells) + r" \\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
TABLE_OUT.write_text("\n".join(lines))
print(f"wrote {TABLE_OUT}\n")
print("\n".join(lines))

# Also: average pairwise agreement excluding self
print()
n = len(sigs)
total = sum(agree[(s1,s2)] for s1 in sigs for s2 in sigs if s1 != s2)
mean_agree = total / (n*(n-1))
print(f"Mean pairwise agreement: {mean_agree:.2f}/7  ({mean_agree/7*100:.0f}%)")

# entropy-family subgroup: TE, SE, Comp
ent_fam = ["token_entropy","semantic_entropy","composite_uq"]
ef_total = sum(agree[(s1,s2)] for s1 in ent_fam for s2 in ent_fam if s1!=s2)
ef_mean = ef_total / (len(ent_fam)*(len(ent_fam)-1))
print(f"Entropy-family (TE/SE/Comp) pairwise agreement: {ef_mean:.2f}/7  ({ef_mean/7*100:.0f}%)")
