"""Counterfactual evaluation: train a LoRA on EVERY detected cluster
(regardless of gate decision) and measure whether the LoRA actually
improves performance over the bare backbone.

This is what gives us the *true* needs_birth label:
    needs_birth = (LoRA accuracy - backbone accuracy) > improvement_threshold

For the paper this is the most expensive but most defensible signal —
without it, neither Eager nor Observation gates can be evaluated.

Pipeline per cluster:
  1. Take cluster's gold test set (block-matched gold/{block}.jsonl)
  2. Score backbone answers on gold
  3. Train LoRA on cluster (lora_train.train_lora)
  4. Score LoRA answers on the same gold
  5. delta = LoRA_score - backbone_score
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict

GOLD_DIR = Path(__file__).resolve().parent.parent / "data" / "gold"

# Map workload block names → gold filenames
_BLOCK_TO_GOLD = {
    "true_need_med": "med",
    "true_need_legal": "legal",
    "true_need_code": "code",
    "trap_a": "trap_a",
    "trap_b": "trap_b",
}


def load_gold(block: str) -> List[Dict]:
    name = _BLOCK_TO_GOLD.get(block, block)
    p = GOLD_DIR / f"{name}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open()]


def _block_distribution(cluster: Dict) -> Dict[str, int]:
    return cluster.get("block_distribution", {})


def dominant_block(cluster: Dict) -> str:
    bd = _block_distribution(cluster)
    if not bd:
        return "noise"
    return max(bd.items(), key=lambda kv: kv[1])[0]


def _generate(model, tok, query: str, max_new_tokens: int = 256) -> str:
    import torch
    msgs = [{"role": "user", "content": query}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    full = tok.decode(out[0], skip_special_tokens=True)
    return full[len(prompt):].strip()


def evaluate_cluster(cluster: Dict,
                     base_model, tok,
                     score_fn,
                     lora_train_fn=None,
                     train_lora: bool = True,
                     output_dir: str = "./poc_observation/results/lora_adapters",
                     items: List[Dict] = None) -> Dict:
    """Returns {backbone_score, lora_score, delta, n_test, dominant_block}."""
    block = dominant_block(cluster)
    gold = load_gold(block)
    if not gold:
        return {"cluster_id": cluster["cluster_id"], "dominant_block": block,
                "error": f"no gold for block {block}"}

    # 1. Backbone scores
    bb_scores = []
    for ex in gold[:5]:  # cap at 5 to keep compute bounded
        pred = _generate(base_model, tok, ex["q"])
        s = score_fn(pred, ex["a"], ex["q"])
        bb_scores.append(s)
    bb_total = sum(s.get("total", s.get("rouge_l", 0)) for s in bb_scores) / len(bb_scores)

    result = {
        "cluster_id": cluster["cluster_id"],
        "dominant_block": block,
        "n_test": len(bb_scores),
        "backbone_score": bb_total,
        "backbone_per_example": bb_scores,
    }

    if not train_lora or lora_train_fn is None:
        result["lora_score"] = None
        result["delta"] = None
        return result

    # 2. Train LoRA
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_meta = lora_train_fn(cluster, items, base_model, tok, output_dir)
    result["lora_train_meta"] = train_meta

    # 3. LoRA scores (model is now wrapped with PEFT)
    from peft import PeftModel
    adapter_path = train_meta["adapter_path"]
    # Reload adapter on a fresh wrapper
    lora_model = PeftModel.from_pretrained(base_model, adapter_path)
    lora_scores = []
    for ex in gold[:5]:
        pred = _generate(lora_model, tok, ex["q"])
        s = score_fn(pred, ex["a"], ex["q"])
        lora_scores.append(s)
    lora_total = sum(s.get("total", s.get("rouge_l", 0)) for s in lora_scores) / len(lora_scores)
    result["lora_score"] = lora_total
    result["lora_per_example"] = lora_scores
    result["delta"] = lora_total - bb_total
    # Tear down adapter wrapping
    try:
        lora_model.unload()
    except Exception:
        pass
    return result


def label_needs_birth(cf_result: Dict, improvement_threshold: float = 0.10) -> bool:
    """Ground truth label: cluster genuinely needs an expert iff LoRA
    improves over backbone by `improvement_threshold` (in [0,1] score)."""
    if cf_result.get("delta") is None:
        return False
    return cf_result["delta"] >= improvement_threshold
