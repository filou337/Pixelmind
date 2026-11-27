"""PIXELMIND – overview. Run: streamlit run app/Accueil.py"""

from collections import Counter

import api_client as api
import pandas as pd
import streamlit as st
from ui import page, show_error

page("PIXELMIND", "🛰️")
st.caption(
    "Détection de changements de bâtiments entre deux images (T1/T2). Les résultats sont des "
    "signaux d'aide à l'analyse, à vérifier par un humain."
)

try:
    health = api.get("/api/v1/health")
    models = api.get("/api/v1/models")
    analyses = api.get("/api/v1/analyses", limit=500)
    workflows = api.get("/api/v1/workflows", limit=500)
except api.ApiError as exc:
    show_error(exc)
    st.stop()

status = Counter(a["status"] for a in analyses)
pending = [w for w in workflows if w["status"] == "awaiting_review"]
c = st.columns(4)
c[0].metric("Analyses", len(analyses))
c[1].metric("Rejetées au contrôle qualité", status.get("rejected", 0))
c[2].metric("Revues en attente", len(pending))
c[3].metric("Modèle par défaut", models["default"] or "aucun")
st.caption(f"API {health['version']} · {health['models']} modèle(s) servi(s)")

if pending:
    st.subheader("Revues en attente")
    for w in pending[:10]:
        st.markdown(f"- `{w['analysis_id']}` · workflow `{w['workflow_id']}` · {w['updated_at']}")
    st.page_link(
        "pages/1_Nouvelle_analyse.py", label="Ouvrir une analyse pour la revoir", icon="🔍"
    )

st.subheader("Dernières analyses")
if analyses:
    st.dataframe(pd.DataFrame(analyses), hide_index=True, width="stretch")
else:
    st.info("Aucune analyse pour l'instant.")
st.page_link("pages/1_Nouvelle_analyse.py", label="Lancer une nouvelle analyse", icon="➕")
