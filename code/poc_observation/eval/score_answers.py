"""Score generated answers vs gold reference.

Two complementary signals:
  - rouge_l : deterministic, reproducible, captures lexical overlap
  - llm_judge : Anthropic Claude or OpenAI GPT-4o (configurable),
                rubric-based scoring on factuality + specificity + completeness

If no API key is available, llm_judge falls back to a heuristic
(answer length variance + token overlap with key technical vocabulary).
"""
from __future__ import annotations
import json, os, re
from collections import Counter
from typing import Dict, List, Optional


def _tokenize(s: str) -> List[str]:
    return re.findall(r"\w+", s.lower())


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F1 (longest common subsequence)."""
    a, b = _tokenize(pred), _tokenize(ref)
    if not a or not b:
        return 0.0
    # LCS via DP, O(|a||b|)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[len(a)][len(b)]
    p = lcs / len(a); r = lcs / len(b)
    return 2 * p * r / (p + r) if p + r else 0.0


JUDGE_RUBRIC = """You are a strict, fair evaluator. Compare a candidate answer to a gold reference.
Score on a 0–3 scale for each axis:
  - factual: Does the candidate match the factual claims of the gold? (0=wrong, 3=fully correct)
  - specific: Does the candidate provide concrete, domain-specific detail (not vague generalities)? (0=vague, 3=specific)
  - complete: Does the candidate cover the same scope as the gold? (0=missing core, 3=fully covers)

Return ONLY a JSON object: {"factual": int, "specific": int, "complete": int, "rationale": "..."}.
"""


def llm_judge(pred: str, ref: str, query: str,
              provider: str = "anthropic",
              model: str = "claude-sonnet-4-5",
              api_key: Optional[str] = None) -> Dict:
    """Returns {factual, specific, complete, total (sum/9), rationale}.
    Falls back to heuristic if no API key."""
    if provider == "anthropic":
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return _heuristic_judge(pred, ref, query)
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model=model,
                max_tokens=400,
                system=JUDGE_RUBRIC,
                messages=[{"role": "user", "content": (
                    f"QUERY: {query}\n\nGOLD REFERENCE:\n{ref}\n\n"
                    f"CANDIDATE ANSWER:\n{pred}\n\n"
                    f"Return only the JSON object, no preamble."
                )}],
            )
            text = msg.content[0].text
            # Robust JSON extraction
            m = re.search(r"\{.*\}", text, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
            f = int(data.get("factual", 0)); s = int(data.get("specific", 0)); c = int(data.get("complete", 0))
            return {"factual": f, "specific": s, "complete": c,
                    "total": (f + s + c) / 9.0,
                    "rationale": data.get("rationale", "")[:300],
                    "judge": f"{provider}/{model}"}
        except Exception as exc:
            return {"factual": 0, "specific": 0, "complete": 0, "total": 0.0,
                    "rationale": f"judge error: {exc}", "judge": "error"}
    elif provider == "openai":
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return _heuristic_judge(pred, ref, query)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": JUDGE_RUBRIC},
                          {"role": "user", "content": f"QUERY: {query}\n\nGOLD: {ref}\n\nCANDIDATE: {pred}\n\nJSON only."}],
                temperature=0.0, max_tokens=400,
            )
            text = r.choices[0].message.content
            m = re.search(r"\{.*\}", text, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
            f = int(data.get("factual", 0)); s = int(data.get("specific", 0)); c = int(data.get("complete", 0))
            return {"factual": f, "specific": s, "complete": c,
                    "total": (f + s + c) / 9.0,
                    "rationale": data.get("rationale", "")[:300],
                    "judge": f"{provider}/{model}"}
        except Exception as exc:
            return {"factual": 0, "specific": 0, "complete": 0, "total": 0.0,
                    "rationale": f"judge error: {exc}", "judge": "error"}
    return _heuristic_judge(pred, ref, query)


def _heuristic_judge(pred: str, ref: str, query: str) -> Dict:
    """No-API fallback: score based on token overlap of technical vocabulary
    (anything ≥ 4 chars) and length adequacy. Range total ∈ [0,1]."""
    pt = set(_tokenize(pred)); rt = set(_tokenize(ref))
    technical = {t for t in rt if len(t) >= 4}
    if not technical:
        return {"factual": 1, "specific": 1, "complete": 1, "total": 0.33,
                "rationale": "no technical vocab in gold", "judge": "heuristic"}
    overlap = len(pt & technical) / len(technical)
    len_ratio = min(len(_tokenize(pred)) / max(1, len(_tokenize(ref))), 2.0)
    len_score = 1.0 if 0.5 <= len_ratio <= 1.8 else 0.5
    f = round(3 * overlap)
    c = round(3 * (overlap * 0.7 + len_score * 0.3))
    s = round(3 * (overlap * 0.5 + (len(pt) / 200) * 0.5 if len(pt) > 50 else overlap))
    return {"factual": min(3, f), "specific": min(3, s), "complete": min(3, c),
            "total": (f + s + c) / 9.0,
            "rationale": f"heuristic overlap={overlap:.2f} len_ratio={len_ratio:.2f}",
            "judge": "heuristic"}


def score_answer(pred: str, ref: str, query: str = "",
                 use_judge: bool = True, **judge_kwargs) -> Dict:
    rl = rouge_l(pred, ref)
    out = {"rouge_l": rl}
    if use_judge:
        j = llm_judge(pred, ref, query, **judge_kwargs)
        out.update(j)
    return out
