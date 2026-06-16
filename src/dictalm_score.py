#!/usr/bin/env python3
"""Likelihood scoring with local DictaLM 2.0 (base) — the fallback reranker.

DictaLM 2.0 is a *base* (pretrained) Hebrew LM, so we use it the way base models are
meant to be used: it assigns a probability to any string. Instead of asking "is this a
word?", we score candidate readings and pick the most probable one — optionally in the
context of the whole line. This disambiguates letters the classifier was unsure about
(e.g. the last letter of קיץ vs קיף vs קיל).

Used as a *sparse fallback* only — CPU + 7B is slow, so the wordlist + final-form rule
should resolve the bulk first.

    from dictalm_score import DictaLM
    lm = DictaLM()
    lm.best_of(["קיץ", "קיף", "קיל"])                       # -> "קיץ"
    lm.best_of(["קיץ", "קיף"], context="בא הסתיו ואחריו ה")  # context-aware
"""
from pathlib import Path

import torch

MODEL_DIR = "/mnt/ssd2/cyttic/models/dictalm2"


class DictaLM:
    def __init__(self, model_dir: str = MODEL_DIR, dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"loading DictaLM from {model_dir} (bf16, CPU) — first load is slow...")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype, low_cpu_mem_usage=True)
        self.model.eval()

    @torch.no_grad()
    def logprob(self, text: str) -> float:
        """Total log-likelihood the model assigns to `text` (higher = more plausible)."""
        ids = self.tok(text, return_tensors="pt").input_ids
        out = self.model(ids, labels=ids)
        # loss is mean NLL per token; multiply back by token count for a total score
        n = ids.shape[1]
        return float(-out.loss.item() * n)

    @torch.no_grad()
    def avg_logprob(self, text: str) -> float:
        """Length-normalized log-likelihood (per token) — fairer across unequal lengths."""
        ids = self.tok(text, return_tensors="pt").input_ids
        out = self.model(ids, labels=ids)
        return float(-out.loss.item())

    def best_of(self, candidates, context: str = "", normalize: bool = True):
        """Return the most probable candidate (optionally within a line context)."""
        scorer = self.avg_logprob if normalize else self.logprob
        scored = [(c, scorer((context + c) if context else c)) for c in candidates]
        scored.sort(key=lambda t: -t[1])
        return scored[0][0], scored


if __name__ == "__main__":
    lm = DictaLM()
    tests = [
        (["קיץ", "קיף", "קיל", "קיך"], ""),                 # real word is קיץ (summer)
        (["שלום", "שלוס", "שלוף"], ""),                     # שלום (peace/hello)
        (["ארץ", "ארף", "ארן"], ""),                        # ארץ (land)
        (["בית", "כית", "ביm"], ""),                        # בית (house)
    ]
    for cands, ctx in tests:
        best, scored = lm.best_of(cands, ctx)
        ranking = "  ".join(f"{c}:{s:.2f}" for c, s in scored)
        print(f"\ncontext={ctx!r}\n  winner: {best}\n  {ranking}")
