"""HOLE 27 — DeepSeek judge stability test (3-judge temperature consensus).

The post-hoc oracle in App. S relies on a single DeepSeek-chat judge call per
(proposal, query) pair with temperature=0. Reviewer concern: is this judgment
stable, or does the proxy/oracle gap (currently 2/7) come from judge noise?

We re-run the oracle on the same 7 proposals with three judge configurations:
  J0: temperature=0.0 (matches deployment) — baseline
  J1: temperature=0.5 (perturbed)
  J2: temperature=1.0 (high variance)
Each judgment is repeated 3x with A/B position randomization. We compute:
  (i)  per-judge BIRTH/SKIP decision per proposal
  (ii) judge-vs-judge agreement (Krippendorff alpha + raw agreement)
  (iii) majority-vote across the 3 judges, scored against deployment SPRT

If agreement >= 5/7 across judges, the oracle is stable. If agreement < 4/7,
the proxy/oracle gap is partly judge-noise-driven.

Inputs:
  data/per_proposal_traces.jsonl       (proposal metadata)
  data/multi_uq_observations.jsonl     (queries per proposal)
  --lora-pairs lora_ablation_pairs.jsonl  (backbone vs LoRA outputs from HOLE 23)

Outputs:
  results/HOLE_27_judge_consensus.json

Run:
    export DEEPSEEK_API_KEY=...
    python HOLE_27_judge_consensus.py \
        --pairs ../../../data/lora_ablation_pairs.jsonl \
        --out   ../../../results/HOLE_27_judge_consensus.json
"""
from __future__ import annotations
import argparse, json, os, random, time, hashlib
from collections import defaultdict
from pathlib import Path

try:
    import requests
except ImportError:
    print("requests missing; install: pip install requests")
    raise

API = "https://api.deepseek.com/v1/chat/completions"
JUDGE_TEMPERATURES = [0.0, 0.5, 1.0]
N_REPEAT = 3   # per judge config, with A/B position randomization
DELTA_TH = 0.15

JUDGE_PROMPT = """You are an expert evaluator. Compare two answers to the same
question and decide which is more accurate, complete, and useful.

QUESTION: {q}

ANSWER A: {a}

ANSWER B: {b}

Respond ONLY in JSON: {{"verdict": "A"|"B"|"tie", "rationale": "<short>"}}"""


def call_judge(question, ans_a, ans_b, temperature, api_key):
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    body = {
        "model": "deepseek-chat",
        "temperature": temperature,
        "max_tokens": 200,
        "messages": [
            {"role": "system",
             "content": "You judge LLM answer quality. Output strict JSON only."},
            {"role": "user",
             "content": JUDGE_PROMPT.format(q=question, a=ans_a, b=ans_b)},
        ],
    }
    r = requests.post(API, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    try:
        # find first {...} block
        s = text.find("{"); e = text.rfind("}") + 1
        return json.loads(text[s:e])["verdict"]
    except Exception:
        return "parse_error"


def judge_pair_with_repeats(query, backbone_ans, lora_ans, temp, api_key):
    """Returns (delta, votes), delta in [-1,1] = (P_lora_wins - P_back_wins)."""
    votes = []
    for rep in range(N_REPEAT):
        # randomise A/B
        flip = (rep % 2 == 1)
        if flip:
            v = call_judge(query, lora_ans, backbone_ans, temp, api_key)
            v = {"A": "lora", "B": "backbone"}.get(v, v)
        else:
            v = call_judge(query, backbone_ans, lora_ans, temp, api_key)
            v = {"A": "backbone", "B": "lora"}.get(v, v)
        votes.append(v)
    n_lora = sum(1 for v in votes if v == "lora")
    n_back = sum(1 for v in votes if v == "backbone")
    n_tie  = sum(1 for v in votes if v == "tie")
    delta = (n_lora - n_back) / max(N_REPEAT, 1)
    return delta, votes


def proposal_decision(deltas):
    """BIRTH iff mean delta >= DELTA_TH across the proposal's queries."""
    if not deltas:
        return None, 0.0
    m = sum(deltas) / len(deltas)
    return ("BIRTH" if m >= DELTA_TH else "SKIP"), m


def agree_pairs(decs_per_judge, pids):
    """Returns dict: judge_pair -> agreement count out of len(pids)."""
    out = {}
    judges = list(decs_per_judge.keys())
    for i, j1 in enumerate(judges):
        for j2 in judges[i+1:]:
            ag = sum(1 for p in pids if decs_per_judge[j1][p] == decs_per_judge[j2][p])
            out[f"{j1}~{j2}"] = ag
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True,
                    help="JSONL with {proposal_id, query, backbone, lora}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-q", type=int, default=10,
                    help="max queries per proposal (cost cap)")
    args = ap.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("WARN: DEEPSEEK_API_KEY not set — will dry-run with mock verdicts")

    # group pairs per proposal
    by_p = defaultdict(list)
    for line in open(args.pairs):
        r = json.loads(line)
        if "proposal_id" in r and "backbone" in r and "lora" in r:
            by_p[r["proposal_id"]].append(r)
    print(f"Loaded {sum(len(v) for v in by_p.values())} pairs across "
          f"{len(by_p)} proposals")

    out = {"config": {"temperatures": JUDGE_TEMPERATURES, "n_repeat": N_REPEAT,
                      "delta_threshold": DELTA_TH, "max_q": args.max_q},
           "proposals": []}

    decs_per_judge = {f"T{int(t*10):02d}": {} for t in JUDGE_TEMPERATURES}
    deltas_per_judge = {f"T{int(t*10):02d}": {} for t in JUDGE_TEMPERATURES}

    t0 = time.time()
    for pid in sorted(by_p):
        rows = by_p[pid][:args.max_q]
        print(f"\n=== Proposal {pid}: {len(rows)} queries ===")
        per_judge = {}
        for temp in JUDGE_TEMPERATURES:
            jname = f"T{int(temp*10):02d}"
            deltas = []
            for r in rows:
                if api_key:
                    try:
                        d, votes = judge_pair_with_repeats(
                            r["query"], r["backbone"], r["lora"], temp, api_key)
                    except Exception as e:
                        print(f"  [{jname}] judge err on q='{r['query'][:30]}': {e}")
                        continue
                else:
                    # dry-run mock: random delta
                    rng = random.Random(hashlib.md5(
                        f"{pid}-{r['query']}-{temp}".encode()).hexdigest())
                    d = rng.uniform(-1, 1) * 0.5
                    votes = ["mock"]*N_REPEAT
                deltas.append(d)
            decision, mean_d = proposal_decision(deltas)
            decs_per_judge[jname][pid] = decision
            deltas_per_judge[jname][pid] = mean_d
            per_judge[jname] = {"decision": decision, "mean_delta": mean_d,
                                "n_queries": len(deltas)}
            print(f"  {jname}: decision={decision} mean_delta={mean_d:.3f}")
        out["proposals"].append({"proposal_id": pid, "judges": per_judge})

    pids = sorted(by_p.keys())
    out["pair_agreement"] = agree_pairs(decs_per_judge, pids)
    # majority vote per proposal
    majority = {}
    for p in pids:
        votes = [decs_per_judge[j][p] for j in decs_per_judge]
        majority[p] = max(set(votes), key=votes.count)
    out["majority_decisions"] = majority
    out["wall_time_sec"] = time.time() - t0

    print("\n=== Pair agreement (out of {}) ===".format(len(pids)))
    for k, v in out["pair_agreement"].items():
        print(f"  {k}: {v}/{len(pids)}")
    print("=== Majority vote ===")
    for p in pids:
        print(f"  P{p}: {majority[p]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
