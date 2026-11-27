"""AI research: questions on the documentation or on one analysis, answered with sources."""

import api_client as api
import streamlit as st
from ui import citations, page, show_error

page("Recherche IA", "💬")
st.caption(
    "Les réponses s'appuient uniquement sur la documentation indexée (et sur l'analyse "
    "choisie). Chaque affirmation renvoie à sa source ; sans information, l'assistant le dit."
)

with st.sidebar:
    try:
        recent = [a for a in api.get("/api/v1/analyses", limit=100) if a["status"] != "rejected"]
    except api.ApiError as exc:
        show_error(exc)
        st.stop()
    target = st.selectbox("Contexte", ["Documentation", *[a["analysis_id"] for a in recent]])
    mode = st.selectbox("Recherche", ["hybrid", "dense", "sparse"])
    doc_types = st.multiselect(
        "Types de documents", ["architecture", "adr", "dataset", "model_card", "report", "readme"]
    )
    if st.button("Effacer la conversation"):
        st.session_state.chat = []

st.session_state.setdefault("chat", [])
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("result"):
            r = msg["result"]
            citations(r["citations"])
            g = r["generation"]
            st.caption(
                f"{g['model'] or g['provider']} · {r['retrieval']['mode']} · "
                f"{r['latency_s']} s" + (" · abstention" if r["abstained"] else "")
            )

if question := st.chat_input("Votre question"):
    st.session_state.chat.append({"role": "user", "content": question})
    try:
        with st.spinner("Recherche des sources…"):
            if target == "Documentation":
                r = api.post(
                    "/api/v1/questions",
                    json={"question": question, "mode": mode, "doc_types": doc_types or None},
                )
            else:
                r = api.post(f"/api/v1/analyses/{target}/questions", json={"question": question})
        st.session_state.chat.append({"role": "assistant", "content": r["answer"], "result": r})
    except api.ApiError as exc:
        st.session_state.chat.append({"role": "assistant", "content": f"⚠️ {exc.detail}"})
    st.rerun()
