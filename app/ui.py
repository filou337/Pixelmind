"""Shared presentation helpers."""

import streamlit as st
from api_client import ApiError

STATUS_LABELS = {
    "completed": ("Terminé", "green"),
    "completed_with_warnings": ("Terminé avec alertes", "orange"),
    "rejected": ("Rejeté", "red"),
    "awaiting_review": ("En attente de revue", "blue"),
    "running": ("En cours", "gray"),
    "PASS": ("PASS", "green"),
    "WARN": ("WARN", "orange"),
    "REVIEW": ("REVIEW", "blue"),
    "FAIL": ("FAIL", "red"),
}


def badge(status: str) -> str:
    label, color = STATUS_LABELS.get(status, (status, "gray"))
    return f":{color}-badge[{label}]"


def page(title: str, icon: str) -> None:
    st.set_page_config(page_title=f"PIXELMIND – {title}", page_icon=icon, layout="wide")
    st.title(f"{icon} {title}")


def show_error(exc: ApiError) -> None:
    if exc.status == 0:
        st.error(f"{exc.detail}. Lancez l'API : `uvicorn pixelmind.api.main:app`.")
    else:
        st.error(f"Erreur {exc.status} : {exc.detail}")


def citations(cits: list[dict]) -> None:
    if not cits:
        return
    with st.expander(f"Sources ({len(cits)})"):
        for c in cits:
            st.markdown(f"**[{c['n']}] {c['title']}**  \n`{c['source']}`")
            st.caption(c["cited_text"][:600])
