# ADR 0006 – Workflow d'agents LangGraph et interface Streamlit

- Statut : accepté
- Date : 2026-11-27

## Contexte

Le Sprint 6 doit permettre une démonstration de bout en bout (dossier §9 et §12) :
analyser, expliquer, faire valider par un humain si nécessaire, produire un rapport.
Le dossier recommande un graphe explicite plutôt qu'un essaim d'agents autonomes.

## Décision

- **Un seul graphe LangGraph, explicite**, lancé après l'analyse déterministe :
  `load → research (option) → quality_gate → human_review (si besoin) → report → persist`.
- **Pas de décision par le LLM** : les règles du contrôle qualité et le routage sont du
  code Python testé. La revue est requise si la qualité des données n'est pas PASS ou si
  le résultat sort de la distribution d'entraînement. Une analyse rejetée ne passe ni
  par l'explication ni par la revue.
- **Le seul appel LLM est l'explication documentaire**, faite par le service RAG. Elle
  est facultative (`explain`) et rangée dans une section séparée du rapport. Si le LLM
  est indisponible, le workflow continue en mode dégradé, avec une alerte.
- **Revue humaine** : le graphe s'arrête (`interrupt`) et reprend sur la décision
  `approved`, `corrected` ou `rejected`, avec le nom du relecteur, un commentaire et des
  retours par région. L'état est enregistré dans SQLite par le checkpointer LangGraph :
  une revue peut arriver après un redémarrage de l'API. Les retours sont conservés comme
  feedback, sans réentraînement automatique (dossier §2.2).
- **Rapport Markdown déterministe** (`artifacts/analyses/{id}/report.md`) : faits
  mesurés, qualité, revue, explication générée avec ses sources, alertes et limites,
  traçabilité (empreintes des entrées, version du modèle, étapes du graphe).
- **Interface Streamlit sans logique métier** : les pages n'appellent que l'API HTTP.
  Elle compte quatre pages (Accueil, Nouvelle analyse, Recherche IA, Gouvernance) au
  lieu des sept du dossier : Overview et Experiments sont regroupés dans Accueil et
  Gouvernance, Compare et Metrics dans Nouvelle analyse.

## Conséquences

- Les étapes « Vision tool », « Geo tool » et « Quality agent » du dossier ne sont pas
  des agents séparés : elles correspondent au service d'analyse déterministe et au nœud
  `quality_gate`. Ce n'est utile d'en faire des agents qu'en présence de décisions non
  déterministes.
- `docker compose up` lance l'API (port 8000) et l'interface (port 8501) depuis la même
  image.
