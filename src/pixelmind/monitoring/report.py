"""Monitoring indicators (dossier §11.3), computed from the audit stores.

Data quality, model usage and outputs, service latency, human review, RAG/LLM usage and
cost, and input drift. Everything is derived from what is already persisted: no extra
instrumentation to keep in sync.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from pixelmind.monitoring.drift import drift_report

# USD per million tokens (input, output), Claude API list prices as of 2026-11.
# Estimate only: check current pricing before relying on it.
LLM_PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}


def _pcts(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50": None, "p95": None}
    return {"n": len(values), "p50": round(float(np.percentile(values, 50)), 3),
            "p95": round(float(np.percentile(values, 95)), 3)}  # fmt: skip


def _share(counter: Counter, total: int) -> dict:
    return {k: {"count": v, "share": round(v / total, 4)} for k, v in counter.most_common()}


def build_monitoring(
    analyses: list[dict], workflows: list[dict], questions: list[dict],
    reference_stats: Path | None, drift_window: int = 200,
) -> dict:  # fmt: skip
    n = len(analyses)
    done = [a for a in analyses if a.get("prediction")]
    ratios = [a["prediction"]["changed_pixel_ratio"] for a in done]
    issues = Counter(i["code"] for a in analyses for i in a.get("quality_issues", [])
                     if i["status"] == "FAIL")  # fmt: skip

    reference = None
    if reference_stats and reference_stats.exists():
        reference = json.loads(reference_stats.read_text())
    recent_stats = [a["input_stats"] for a in done if a.get("input_stats")][:drift_window]

    llm = [q for q in questions if q["generation"]["llm_called"]]
    tokens_in = sum(q["generation"]["usage"].get("input_tokens", 0) for q in llm)
    tokens_out = sum(q["generation"]["usage"].get("output_tokens", 0) for q in llm)
    cost = 0.0
    priced = True
    for q in llm:
        u, model = q["generation"]["usage"], q["generation"]["model"]
        if not u:
            continue
        price = LLM_PRICES.get(model)
        if price is None:
            priced = False
            continue
        cost += (u.get("input_tokens", 0) * price[0] + u.get("output_tokens", 0) * price[1]) / 1e6

    reviewed = [w for w in workflows if w.get("review_decision")]
    return {
        "data_quality": {
            "analyses": n,
            "quality_status": _share(Counter(a["quality_status"] for a in analyses), n)
            if n
            else {},
            "rejected_reasons": dict(issues.most_common()),
        },
        "model": {
            "usage": dict(Counter(a["model"]["name"] for a in analyses).most_common()),
            "changed_pixel_ratio": _pcts(ratios),
            "empty_mask_rate": round(float(np.mean([r == 0 for r in ratios])), 4)
            if ratios
            else None,
            "over_half_changed_rate": round(float(np.mean([r > 0.5 for r in ratios])), 4)
            if ratios
            else None,
            "f1_vs_reference": _pcts(
                [
                    a["metrics"]["f1"]
                    for a in done
                    if a.get("metrics") and a["metrics"]["f1"] is not None
                ]
            ),
        },  # fmt: skip
        "service": {
            "analysis_latency_s": _pcts([a["timings"]["total_s"] for a in analyses]),
            "inference_latency_s": _pcts(
                [a["timings"]["inference_s"] for a in done if "inference_s" in a["timings"]]
            ),
        },  # fmt: skip
        "review": {
            "workflows": len(workflows),
            "status": dict(Counter(w["status"] for w in workflows).most_common()),
            "decisions": dict(Counter(w["review_decision"] for w in reviewed).most_common()),
            "review_rate": round(len(reviewed) / len(workflows), 4) if workflows else None,
        },
        "rag": {
            "questions": len(questions),
            "abstention_rate": round(float(np.mean([q["abstained"] for q in questions])), 4)
            if questions
            else None,
            "llm_calls": len(llm),
            "tokens": {"input": tokens_in, "output": tokens_out},
            "estimated_cost_usd": round(cost, 4) if priced else None,
            "latency_s": _pcts([q["latency_s"] for q in questions]),
        },  # fmt: skip
        "drift": drift_report(reference, recent_stats),
    }
