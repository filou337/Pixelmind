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
except api.ApiError as exc:
    show_error(exc)
    st.stop()

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
