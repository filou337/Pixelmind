# Changelog

## 0.1.0 – 2026-11-27 – Version de démonstration

Première version de bout en bout de PIXELMIND, construite en 8 sprints (S0 à S7).

- **Données (S1)** : téléchargement LEVIR-CD vérifié (MD5), manifeste versionné,
  contrôle qualité PASS/WARN/REVIEW/FAIL, détection des fuites entre splits, rapport EDA.
- **Baselines (S2)** : différences d'images et gradient boosting, métriques pixel cumulées
  par histogramme, seuil choisi sur val. Meilleur F1 test : 43,1 % (`gbm`).
- **Deep learning (S3)** : U-Net et Siamese U-Net, BCE + Dice, MLflow, notebook Kaggle.
  *Entraînement GPU à lancer.*
- **API (S4)** : analyse T1/T2 synchrone, validation des fichiers envoyés, régions,
  overlay, stockage SQLite et audit, Docker.
- **RAG (S5)** : recherche hybride locale (Qdrant, embeddings multilingues, BM25),
  réponses de Claude avec citations, abstention. Recall@5 : 92 %.
- **Agents et interface (S6)** : workflow LangGraph (contrôle qualité, revue humaine
  avec reprise, rapport), interface Streamlit en quatre pages.
- **Gouvernance (S7)** : clés d'API nommées, journal d'accès, audit par analyse,
  monitoring, dérive des entrées (PSI et χ²), CI avec pip-audit, Trivy et Dependabot,
  runbook.
