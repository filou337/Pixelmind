"""New analysis: upload T1/T2, compare, review, report."""

import io

import api_client as api
import pandas as pd
import streamlit as st
from PIL import Image
from ui import badge, citations, page, show_error

page("Nouvelle analyse", "🔍")


@st.cache_data(show_spinner=False)
def image(uri: str) -> Image.Image:
    return Image.open(io.BytesIO(api.artifact(uri))).convert("RGB")


def open_analysis(aid: str) -> None:
    st.session_state.analysis = api.get(f"/api/v1/analyses/{aid}")
    flows = api.get(f"/api/v1/analyses/{aid}/workflows")
    st.session_state.workflow = (
        api.get(f"/api/v1/workflows/{flows[0]['workflow_id']}") if flows else None
    )


# --- sidebar: reopen an existing analysis ------------------------------------
with st.sidebar:
    try:
        recent = api.get("/api/v1/analyses", limit=100)
    except api.ApiError as exc:
        show_error(exc)
        st.stop()
    if recent:
        labels = {f"{a['analysis_id']} · {a['status']}": a["analysis_id"] for a in recent}
        pick = st.selectbox("Rouvrir une analyse", ["—", *labels])
        if pick != "—" and st.button("Ouvrir"):
            open_analysis(labels[pick])

# --- upload form ---------------------------------------------------------------
models = api.get("/api/v1/models")
names = [m["name"] for m in models["models"]]
with st.form("new"):
    c1, c2, c3 = st.columns(3)
    t1 = c1.file_uploader("Image T1 (avant)", type=["png", "jpg", "jpeg", "tif", "tiff"])
    t2 = c2.file_uploader("Image T2 (après)", type=["png", "jpg", "jpeg", "tif", "tiff"])
    ref = c3.file_uploader("Masque de référence (facultatif)", type=["png", "tif", "tiff"])
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox(
        "Modèle",
        names,
        index=names.index(models["default"]) if models["default"] in names else 0,
        help="Seuil choisi sur la validation ; F1 test dans Gouvernance",
    )
    gsd = c2.number_input("Résolution au sol (m/pixel, 0 = inconnue)", 0.0, 1000.0, 0.0, 0.1)
    explain = c3.checkbox("Explication documentaire (appel LLM)", value=False)
    submitted = st.form_submit_button("Analyser", type="primary")

if submitted:
    if not (t1 and t2):
        st.warning("Les images T1 et T2 sont obligatoires.")
    else:
        files = {"t1": (t1.name, t1.getvalue()), "t2": (t2.name, t2.getvalue())}
        if ref:
            files["reference_mask"] = (ref.name, ref.getvalue())
        data = {"model": model, **({"gsd_m": gsd} if gsd > 0 else {})}
        try:
            with st.spinner("Analyse en cours…"):
                st.session_state.analysis = api.post("/api/v1/analyses", files=files, data=data)
            with st.spinner("Contrôle qualité et rapport…"):
                aid = st.session_state.analysis["analysis_id"]
                st.session_state.workflow = api.post(
                    f"/api/v1/analyses/{aid}/workflows", json={"explain": explain}
                )
        except api.ApiError as exc:
            show_error(exc)

a = st.session_state.get("analysis")
if not a:
    st.stop()

# --- result ---------------------------------------------------------------------
st.divider()
st.subheader(f"Analyse `{a['analysis_id']}`")
st.markdown(
    f"{badge(a['status'])} {badge(a['quality_status'])} · modèle `{a['model']['name']}` "
    f"(`{a['model']['version']}`, seuil {a['model']['threshold']})"
)
for w in a["warnings"]:
    st.warning(w)

p = a["prediction"]
if p:
    c = st.columns(4)
    c[0].metric("Pixels changés", f"{p['changed_pixel_ratio']:.2%}")
    c[1].metric("Régions", p["n_regions"])
    c[2].metric(
        "Surface estimée",
        f"{p['changed_area_m2_estimate']:,.0f} m²"
        if p["changed_area_m2_estimate"] is not None
        else "non calculée",
    )
    if a["metrics"]:
        c[3].metric("F1 vs référence", f"{a['metrics']['f1']:.1%}")

    arts = a["artifacts"]
    left, right = st.columns([3, 2])
    with left:
        mix = st.slider("T1 ⟷ T2", 0, 100, 100, help="0 = avant, 100 = après")
        show_overlay = st.toggle("Superposer les changements détectés", value=True)
        base = Image.blend(image(arts["t1.png"]), image(arts["t2.png"]), mix / 100)
        if show_overlay:
            mask = image(arts["mask.png"]).convert("L")
            red = Image.new("RGB", base.size, (255, 40, 40))
            base = Image.composite(Image.blend(base, red, 0.5), base, mask)
        st.image(base, width="stretch")
    with right:
        tabs = st.tabs(
            ["Probabilité", "Masque", *(["Référence"] if "reference_mask.png" in arts else [])]
        )
        tabs[0].image(image(arts["probability.png"]), width="stretch")
        tabs[1].image(image(arts["mask.png"]), width="stretch")
        if len(tabs) > 2:
            tabs[2].image(image(arts["reference_mask.png"]), width="stretch")
        if p["regions"]:
            st.dataframe(pd.DataFrame(p["regions"][:20]), hide_index=True, height=220)
else:
    for i in a["quality_issues"]:
        st.error(f"{i['status']} `{i['code']}` : {i['message']}")

# --- workflow: review + report --------------------------------------------------
wf = st.session_state.get("workflow")
st.divider()
st.subheader("Contrôle, revue et rapport")
if wf is None:
    if st.button("Lancer le contrôle qualité et le rapport"):
        st.session_state.workflow = api.post(
            f"/api/v1/analyses/{a['analysis_id']}/workflows", json={"explain": False}
        )
        st.rerun()
    st.stop()

st.markdown(f"Workflow `{wf['workflow_id']}` · {badge(wf['status'])}")
for w in wf["warnings"]:
    st.warning(w)
if wf["research"]:
    with st.container(border=True):
        st.markdown("**Explication documentaire (générée)**")
        st.markdown(wf["research"]["answer"])
        citations(wf["research"]["citations"])

if wf["status"] == "awaiting_review":
    st.info(
        "Revue humaine requise :\n\n" + "\n".join(f"- {r}" for r in wf["pending_review"]["reasons"])
    )
    regions = (p or {}).get("regions", [])[:20]
    with st.form("review"):
        decision = st.radio(
            "Décision",
            ["approved", "corrected", "rejected"],
            horizontal=True,
            format_func={
                "approved": "Valider",
                "corrected": "Valider avec corrections",
                "rejected": "Rejeter",
            }.get,
        )
        reviewer = st.text_input("Relecteur")
        comment = st.text_area("Commentaire")
        feedback = []
        if regions:
            st.caption("Retour par région (enregistré comme feedback, sans réentraînement)")
            for i, r in enumerate(regions[:5], start=1):
                label = st.selectbox(
                    f"Région {i} {r['bbox']} ({r['area_px']} px)",
                    ["—", "confirmed", "false_positive"],
                    key=f"fb{i}",
                )
                if label != "—":
                    feedback.append({"bbox": r["bbox"], "label": label})
        if st.form_submit_button("Enregistrer la revue", type="primary"):
            if not reviewer.strip():
                st.warning("Indiquez le nom du relecteur.")
            else:
                try:
                    st.session_state.workflow = api.post(
                        f"/api/v1/workflows/{wf['workflow_id']}/review",
                        json={
                            "decision": decision,
                            "reviewer": reviewer.strip(),
                            "comment": comment or None,
                            "region_feedback": feedback,
                        },
                    )
                    st.rerun()
                except api.ApiError as exc:
                    show_error(exc)
elif wf["report_uri"]:
    report = api.artifact(wf["report_uri"]).decode()
    st.download_button(
        "Télécharger le rapport (Markdown)", report, file_name=f"rapport_{a['analysis_id']}.md"
    )
    with st.expander("Rapport", expanded=True):
        st.markdown(report.replace("![Superposition des changements sur T2](overlay.png)", ""))
