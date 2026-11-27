"""Markdown analysis report: measured facts, review, sourced explanation, limits, trace.

Deterministic template: the only generated text is the RAG explanation, shown in its own
section with its citations, separate from measured facts (dossier §9.4).
"""


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x:.1%}"


def render_report(state: dict, version: str) -> str:
    a = state["analysis"]
    aid, m, p = a["analysis_id"], a["model"], a.get("prediction")
    model_f1 = _pct((a.get("model_test_metrics") or {}).get("f1"))
    status = {"completed": "Terminé", "rejected": "Rejeté"}.get(state["status"], state["status"])
    L = [
        f"# Rapport d'analyse {aid}",
        "",
        f"- **Statut final** : {status}",
        f"- Analyse créée le {a['created_at']} · workflow `{state['workflow_id']}` · "
        f"PIXELMIND {version}",
        "",
        "> Ce rapport est une aide à l'analyse. Les changements signalés sont des candidats "
        "à vérifier ; ils ne prouvent ni la légalité, ni la date, ni la nature d'une "
        "construction.",
        "",
        "## 1. Faits mesurés",
        "",
        f"- Modèle : `{m['name']}` ({m['kind']}, version `{m['version']}`), seuil "
        f"{m['threshold']}/255, F1 test LEVIR-CD {model_f1}",
        f"- Qualité des données : **{a['quality_status']}**",
    ]
    if p:
        area = (f"{p['changed_area_m2_estimate']} m² (estimation, résolution déclarée "
                f"{a['parameters']['gsd_m']} m/pixel)" if p["changed_area_m2_estimate"] is not None
                else "non calculée (résolution au sol non fournie)")  # fmt: skip
        L += [
            f"- Pixels changés : {p['changed_pixel_ratio']:.2%} de l'image",
            f"- Surface changée : {area}",
            f"- Régions changées : {p['n_regions']}",
        ]
        if p["regions"]:
            L += ["", "| Région | Boîte [x0, y0, x1, y1] | Pixels | Score moyen |",
                  "|---|---|---|---|"]  # fmt: skip
            L += [f"| {i} | {r['bbox']} | {r['area_px']} | {r['mean_score']} |"
                  for i, r in enumerate(p["regions"][:10], start=1)]  # fmt: skip
        L += ["", "![Superposition des changements sur T2](overlay.png)"]
    else:
        L.append("- Aucune prédiction : l'analyse a été rejetée au contrôle qualité.")
    if a.get("metrics"):
        mt = a["metrics"]
        L += ["", f"Comparaison au masque de référence : F1 {_pct(mt['f1'])}, IoU "
                  f"{_pct(mt['iou'])}, précision {_pct(mt['precision'])}, rappel "
                  f"{_pct(mt['recall'])}."]  # fmt: skip

    L += ["", "## 2. Qualité et contrôle", ""]
    issues = a.get("quality_issues", [])
    L += [f"- {i['status']} `{i['code']}` : {i['message']}" for i in issues] or [
        "- Aucune anomalie détectée sur les entrées."
    ]
    gate = state["gate"]
    label = {"pass": "validé automatiquement", "review": "revue humaine requise",
             "rejected": "rejeté"}[gate["decision"]]  # fmt: skip
    L.append(f"- Contrôle qualité du workflow : **{label}**")
    L += [f"  - {r}" for r in gate["reasons"]]

    L += ["", "## 3. Revue humaine", ""]
    review = state.get("review")
    if review:
        L.append(f"- Décision : **{review['decision']}** par {review.get('reviewer', '?')} "
                 f"le {review['reviewed_at']}")  # fmt: skip
        if review.get("comment"):
            L.append(f"- Commentaire : {review['comment']}")
        fb = review.get("region_feedback") or []
        if fb:
            L.append(f"- Retours sur {len(fb)} région(s), enregistrés comme feedback "
                     "(pas de réentraînement automatique) :")  # fmt: skip
            L += [f"  - {f['bbox']} → {f['label']}" + (f" ({f['note']})" if f.get("note") else "")
                  for f in fb]  # fmt: skip
    else:
        L.append("- Non requise." if gate["decision"] == "pass" else "- Sans objet.")

    research = state.get("research")
    if research:
        L += ["", "## 4. Explication documentaire (générée)", "",
              f"*Question : {research['question']}* — modèle `{research['model']}`", "",
              research["answer"]]  # fmt: skip
        if research["citations"]:
            L += ["", "Sources :"]
            L += [f"- [{c['n']}] {c['title']} (`{c['source']}`)" for c in research["citations"]]

    L += ["", "## 5. Alertes et limites", ""]
    L += [f"- Alerte : {w}" for w in a.get("warnings", []) + state.get("warnings", [])]
    L += [f"- {lim}" for lim in a.get("limitations", [])]

    L += ["", "## 6. Traçabilité", "",
          f"- Analyse `{aid}`, modèle `{m['name']}` version `{m['version']}`"]  # fmt: skip
    for role, info in a.get("inputs", {}).items():
        L.append(f"- Entrée {role} : `{info['filename']}`, sha256 `{info['sha256'][:16]}…`")
    L.append("- Étapes : " + " → ".join(f"{s['node']} ({s['at'][11:19]})" for s in state["trace"]))
    return "\n".join(L) + "\n"
