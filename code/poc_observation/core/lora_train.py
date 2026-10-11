"""Train a LoRA adapter on a cluster's queries (with cheap synthetic answers).

For the POC we don't have human-curated training data per cluster, so:
  - We use the cluster's queries (member_indices)
  - We pair each with a backbone-generated answer that has been
    "improved" via a stronger system prompt (`SENIOR_PREFIX`) — this
    follows the original 1H POC design specification.
  - LoRA rank=8, 1 epoch — ~3 min on L4 for ~50 queries.

Returns a path to the saved adapter, plus training metrics.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from typing import List, Dict, Optional

SENIOR_PREFIX_TEMPLATES = {
    "true_need_med": "You are an expert clinical physician with 30 years of experience. Answer with precise medical reasoning, citing pathophysiology and evidence-based management.\n\n",
    "true_need_legal": "Vous êtes un juriste expérimenté en droit français. Citez précisément les articles applicables, le raisonnement juridique et la jurisprudence si pertinente.\n\n",
    "true_need_code": "You are a senior algorithms engineer. Provide a working, optimal implementation with complexity analysis and edge-case handling.\n\n",
    "trap_a": "Provide a clear, concise answer.\n\n",
    "trap_b": "Répondez clairement et brièvement.\n\n",
    "noise": "Answer concisely.\n\n",
}


def _generate_senior_answer(model, tok, query: str, prefix: str,
                            max_new_tokens: int = 256) -> str:
    import torch
    msgs = [{"role": "user", "content": prefix + query}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    full = tok.decode(out[0], skip_special_tokens=True)
    # Crude prompt strip
    if prefix in full:
        return full.split(prefix, 1)[-1].split(query, 1)[-1].strip()
    return full[len(prompt):].strip()


def train_lora(cluster: Dict,
               items: List[Dict],
               base_model,
               tok,
               output_dir: str,
               lora_r: int = 8,
               epochs: int = int(__import__("os").environ.get("POC_LORA_EPOCHS", "1")),
               lr: float = 2e-4,
               batch_size: int = 4,
               max_examples: int = 50) -> Dict:
    """Train a LoRA adapter on the cluster. Returns metrics + adapter path.

    NOTE: this requires `peft` and `transformers`. On a fresh deploy run
    `./venv/bin/pip install peft accelerate datasets` first.
    """
    import torch
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    )
    from datasets import Dataset

    member_qs = [items[i]["query"] for i in cluster["member_indices"][:max_examples]]
    blocks = [items[i].get("block", "noise") for i in cluster["member_indices"][:max_examples]]
    # Generate "senior" answers
    pairs = []
    for q, b in zip(member_qs, blocks):
        prefix = SENIOR_PREFIX_TEMPLATES.get(b, SENIOR_PREFIX_TEMPLATES["noise"])
        a = _generate_senior_answer(base_model, tok, q, prefix)
        pairs.append({"query": q, "answer": a})

    # Build training set
    def fmt(ex):
        msgs = [
            {"role": "user", "content": ex["query"]},
            {"role": "assistant", "content": ex["answer"]},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False)
        enc = tok(text, truncation=True, max_length=512, padding="max_length")
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds = Dataset.from_list(pairs).map(fmt, remove_columns=["query", "answer"])

    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    peft_model = get_peft_model(base_model, cfg)
    peft_model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=peft_model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    train_out = trainer.train()
    adapter_path = Path(output_dir) / f"cluster_{cluster['cluster_id']}_adapter"
    peft_model.save_pretrained(adapter_path)

    return {
        "cluster_id": cluster["cluster_id"],
        "adapter_path": str(adapter_path),
        "n_examples": len(pairs),
        "final_loss": float(train_out.training_loss),
        "epochs": epochs,
        "lora_r": lora_r,
    }
