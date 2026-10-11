"""Three uncertainty quantification signals on a HuggingFace causal LM.

Each signal returns a single scalar in roughly [0, 5] (higher = more uncertain).
All signals operate on a SINGLE forward pass (no extra training).

Signals
-------
  token_entropy(model, tok, query)
        Mean Shannon entropy of next-token distribution over generated tokens.
        Robust, fast, mainstream choice in UQ literature.

  ensemble_disagreement(model, tok, query, T=4)
        Variance of answer length / first-token across T sampled generations.
        Captures epistemic uncertainty via output diversity.

  mc_dropout(model, tok, query, T=5)
        Mean entropy across T forward passes with dropout enabled
        (requires model.train() mode but no parameter updates).

Returns
-------
    {"signal_name": float, "metadata": {...}}
"""
from __future__ import annotations
import math
import torch
from typing import Optional


def _generate_with_scores(model, tok, query: str, max_new_tokens: int = 64,
                          do_sample: bool = False, temperature: float = 1.0):
    msgs = [{"role": "user", "content": query}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=0.95 if do_sample else 1.0,
            pad_token_id=tok.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
    return out


def token_entropy(model, tok, query: str, max_new_tokens: int = 64) -> float:
    """Mean Shannon entropy (nats) of next-token logits over generated steps."""
    out = _generate_with_scores(model, tok, query, max_new_tokens=max_new_tokens, do_sample=False)
    if not out.scores:
        return 0.0
    ent = []
    for step_logits in out.scores:
        probs = torch.softmax(step_logits, dim=-1)
        # Numerical stability: cap probs to [eps, 1]
        eps = 1e-12
        probs = probs.clamp(min=eps)
        e = -(probs * probs.log()).sum(dim=-1)
        ent.append(e.mean().item())
    return float(sum(ent) / len(ent))


def ensemble_disagreement(model, tok, query: str, T: int = 4,
                          max_new_tokens: int = 64) -> float:
    """Coefficient of variation of generated answer lengths across T sampled passes.

    High disagreement = different lengths = different content paths = uncertainty.
    """
    lengths = []
    first_tokens = []
    for _ in range(T):
        out = _generate_with_scores(model, tok, query, max_new_tokens=max_new_tokens,
                                    do_sample=True, temperature=0.9)
        seq = out.sequences[0]
        # Strip prompt: assume sequences = prompt + generated
        gen = seq[-len(out.scores):] if out.scores else seq
        # Length until eos
        eos = (gen == tok.eos_token_id).nonzero(as_tuple=True)[0]
        L = eos[0].item() if len(eos) else len(gen)
        lengths.append(L)
        first_tokens.append(gen[0].item() if len(gen) else 0)
    if not lengths:
        return 0.0
    mean = sum(lengths) / len(lengths)
    var = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    cv = math.sqrt(var) / (mean + 1e-9)
    # Disagreement on first token (categorical) — additional signal
    first_disagree = len(set(first_tokens)) / T
    # Combine: scale to roughly [0, 5]
    return float(2.0 * cv + 2.0 * first_disagree)


def mc_dropout(model, tok, query: str, T: int = 5, max_new_tokens: int = 64) -> float:
    """Mean token entropy across T forward passes with dropout enabled."""
    was_training = model.training
    # Force dropout layers active without updating gradients
    def _enable_dropout(m):
        for mod in m.modules():
            if mod.__class__.__name__.lower().startswith("dropout"):
                mod.train()
    _enable_dropout(model)
    ents = []
    try:
        for _ in range(T):
            ents.append(token_entropy(model, tok, query, max_new_tokens=max_new_tokens))
    finally:
        if not was_training:
            model.eval()
    return float(sum(ents) / len(ents)) if ents else 0.0


SIGNALS = {
    "token_entropy": token_entropy,
    "ensemble_disagreement": ensemble_disagreement,
    "mc_dropout": mc_dropout,
}


def compute(signal_name: str, model, tok, query: str, **kwargs) -> float:
    fn = SIGNALS.get(signal_name)
    if fn is None:
        raise ValueError(f"Unknown signal {signal_name}; choose from {list(SIGNALS)}")
    return fn(model, tok, query, **kwargs)
