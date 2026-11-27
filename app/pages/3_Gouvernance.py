"""Governance: served models and their metrics, documentary index, analyses and reviews."""

import re
from pathlib import Path

import api_client as api
import pandas as pd
import streamlit as st
from ui import page, show_error

page("Gouvernance", "🧭")

try:
    models = api.get("/api/v1/models")
    workflows = api.get("/api/v1/workflows", limit=500)
    mon = api.get("/api/v1/monitoring")
except api.ApiError as exc:
    show_error(exc)
    st.stop()


def pct(x):
    return "–" if x is None else f"{x:.0%}"


st.subheader("Monitoring")
dq, md, sv, rv, rag = mon["data_quality"], mon["model"], mon["service"], mon["review"], mon["rag"]
c = st.columns(5)
c[0].metric("Analyses", dq["analyses"])
c[1].metric(
    "Rejet qualité",
    pct((dq["quality_status"].get("FAIL") or {}).get("share", 0)) if dq["analyses"] else "–",
)
c[2].metric(
    "Latence p95",
    f"{sv['analysis_latency_s']['p95']} s" if sv["analysis_latency_s"]["p95"] is not None else "–",
)
c[3].metric("Taux de revue", pct(rv["review_rate"]))
cost = rag["estimated_cost_usd"]
c[4].metric(
    "Coût LLM estimé",
    f"{cost:.2f} $" if cost is not None else "–",
    help=f"{rag['llm_calls']} appels, {rag['tokens']['input']:,} tokens en entrée, "
    f"{rag['tokens']['output']:,} en sortie (tarifs publics 2026-11, estimation)",
)
left, right = st.columns(2)
with left:
    st.markdown("**Qualité des données**")
    st.dataframe(pd.DataFrame(dq["quality_status"]).T, width="stretch")
    st.markdown(
        f"Masques vides : {pct(md['empty_mask_rate'])} · plus de 50 % changé : "
        f"{pct(md['over_half_changed_rate'])} · abstention RAG : "
        f"{pct(rag['abstention_rate'])}"
    )
with right:
    drift = mon["drift"]
    st.markdown("**Dérive des entrées** (PSI contre LEVIR-CD train)")
    if drift["status"] == "insufficient_data":
        st.info(f"{drift['n']} analyses : au moins {drift['min_n']} sont nécessaires.")
    elif drift["status"] == "no_reference":
        st.info("Statistiques de référence absentes (python -m pixelmind.monitoring.drift).")
    else:
        if drift["alert"]:
            st.error(drift["action"])
        st.dataframe(
            pd.DataFrame(drift["features"]).T[
                ["label", "psi", "level", "reference_mean", "current_mean"]
            ],
            width="stretch",
        )

st.subheader("Modèles servis")
st.caption(
    "Seuil choisi sur la validation ; métriques au niveau pixel sur LEVIR-CD test. "
    "Version = SHA-256 des poids."
)
rows = [
    {
        "modèle": m["name"],
        "type": m["kind"],
        "version": m["version"],
        "seuil": m["threshold"],
        **{k: m["test_metrics"][k] for k in ("f1", "iou", "precision", "recall", "ap")},
    }
    for m in models["models"]
]
st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    width="stretch",
    column_config={
        k: st.column_config.NumberColumn(format="percent")
        for k in ("f1", "iou", "precision", "recall", "ap")
    },
)

st.subheader("Index documentaire (RAG)")
try:
    idx = api.get("/api/v1/rag/index")
    st.markdown(
        f"{idx['n_chunks']} passages · corpus `{idx['corpus_sha256'][:12]}…` · "
        f"construit le {idx['created_at']}  \nEmbeddings : `{idx['embedder']}`"
    )
    st.dataframe(
        pd.DataFrame(idx["documents"])[["doc_id", "doc_type", "path", "licence"]],
        hide_index=True,
        width="stretch",
    )
except api.ApiError as exc:
    st.info(f"Index indisponible : {exc.detail}")

st.subheader("Workflows de revue")
if workflows:
    df = pd.DataFrame(workflows)
    st.bar_chart(df["status"].value_counts())
    st.dataframe(df, hide_index=True, width="stretch")
else:
    st.info("Aucun workflow pour l'instant.")

docs = Path(__file__).resolve().parents[2] / "docs"
for title, rel in (
    ("Benchmark LEVIR-CD", "experiments/levircd_benchmark.md"),
    ("Évaluation du RAG", "experiments/rag_eval.md"),
    ("Fiche dataset LEVIR-CD", "data/levircd_card.md"),
    ("Model card – baselines", "model_cards/baselines.md"),
):
    path = docs / rel
    if path.exists():
        with st.expander(title):
            # images are relative to docs/: not reachable from here, so they are dropped
            st.markdown(re.sub(r"!\[[^\]]*\]\([^)]*\)", "", path.read_text()))
