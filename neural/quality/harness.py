"""OVERNIGHT-5 — reusable Neural quality-retention harness.

Distills the CAPACITY-4 quality-v2 methodology into model-agnostic pieces:

1. ``tf_metrics``  — teacher-forced behavioral metrics between two logit
   tensors (top-1 agreement, median |Δlogprob| of the corpus token, mean and
   p95 KL, perplexity of side A).
2. ``null_envelope_check`` — scores a system-under-test against a measured
   fp-summation-order NULL DISTRIBUTION (n runs of an exact-weight reference
   under different valid accumulation orders). This is the load-bearing
   lesson of CAPACITY-4: at 30+ layer depth, behavioral disagreement between
   two *exact-weight* executions is dominated by fp-order chaos, so a quality
   gate is only meaningful relative to that measured floor — never against
   absolute thresholds calibrated on quantization-scale error.
3. ``rel_l2_stats`` — per-expert (or per-module) output error against a
   float64 reference: the fp-order-robust instrument for representation
   changes (quantization, kernel swaps).

Free-generation token agreement is deliberately NOT provided: autoregressive
divergence amplifies ulp-scale differences and measures nothing about
quality (Q80-16, CAPACITY-4).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

TF_METRIC_KEYS = ("top1", "med_dlp", "mean_kl", "p95_kl", "ppl")


def tf_metrics(logits_a: torch.Tensor, logits_b: torch.Tensor,
               next_tokens: torch.Tensor) -> dict[str, float]:
    """Teacher-forced comparison of A against anchor B.

    logits_* : [T, vocab] float; next_tokens: [T-? aligned] int64 — the
    ground-truth continuation for each position (len == logits rows).
    ppl is side A's perplexity on the corpus (anchor-independent)."""
    assert logits_a.shape == logits_b.shape
    n = next_tokens.numel()
    a = logits_a[:n].float()
    b = logits_b[:n].float()
    lp_a = torch.log_softmax(a, dim=-1)
    lp_b = torch.log_softmax(b, dim=-1)
    ch_a = lp_a.gather(1, next_tokens[:, None])[:, 0]
    ch_b = lp_b.gather(1, next_tokens[:, None])[:, 0]
    kl = torch.nn.functional.kl_div(lp_a, lp_b, log_target=True,
                                    reduction="none").sum(-1)
    return {"top1": (a.argmax(-1) == b.argmax(-1)).float().mean().item(),
            "med_dlp": (ch_b - ch_a).abs().median().item(),
            "mean_kl": kl.mean().item(),
            "p95_kl": kl.quantile(0.95).item(),
            "ppl": torch.exp(-ch_a.mean()).item()}


@dataclass
class EnvelopeVerdict:
    accepted: bool
    checks: dict[str, dict]


def null_envelope_check(system_metrics: dict[str, float],
                        null_metrics: list[dict[str, float]],
                        n_sigma: float = 4.0,
                        keys: tuple = TF_METRIC_KEYS) -> EnvelopeVerdict:
    """Accept the system iff, for every metric, |system - null_mean| <=
    n_sigma * null_std. Nulls must be >= 3 runs of an exact-weight reference
    under distinct valid fp accumulation orders on the SAME corpus/anchor.
    4-sigma default: real (quantization-scale) defects sit >=10 sigma out,
    while small-n envelopes have high min-max false-fail rates."""
    assert len(null_metrics) >= 3, "need >=3 null orders for a distribution"
    checks: dict[str, dict] = {}
    ok = True
    for k in keys:
        vals = torch.tensor([m[k] for m in null_metrics], dtype=torch.float64)
        mu = vals.mean().item()
        sd = max(vals.std().item(), 1e-12)
        within = abs(system_metrics[k] - mu) <= n_sigma * sd
        checks[k] = {"system": system_metrics[k], "null_mean": mu,
                     "null_std": sd, "n_sigma": n_sigma, "within": within}
        ok = ok and within
    return EnvelopeVerdict(accepted=ok, checks=checks)


def rel_l2_stats(y: torch.Tensor, y_ref64: torch.Tensor,
                 mean_max: float = 0.01, cos_min: float = 0.9995,
                 p99_max: float = 0.03) -> dict:
    """Per-row relative-L2/cosine of y against a float64 reference, with the
    CAPACITY-4 instrument-A default thresholds (~3x above the bf16 output-
    rounding floor, >=12x below INT4-quantization-class error)."""
    assert y_ref64.dtype == torch.float64
    yd = y.double()
    rel = (yd - y_ref64).norm(dim=-1) / y_ref64.norm(dim=-1).clamp(min=1e-12)
    cos = torch.nn.functional.cosine_similarity(yd, y_ref64, dim=-1)
    out = {"mean_rel_l2": rel.mean().item(),
           "p99_rel_l2": rel.quantile(0.99).item(),
           "mean_cos": cos.mean().item(),
           "thresholds": {"mean_max": mean_max, "cos_min": cos_min,
                          "p99_max": p99_max}}
    out["pass"] = (out["mean_rel_l2"] <= mean_max
                   and out["mean_cos"] >= cos_min
                   and out["p99_rel_l2"] <= p99_max)
    return out
