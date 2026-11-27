# Évaluation du RAG documentaire

Généré par `python -m pixelmind.rag.evaluate`. Jeu : `configs/rag_eval_questions.yaml` (25 questions avec réponse dans le corpus, 6 hors corpus). Index : 128 passages, corpus sha256 `a8c27bc0b9c6…`, embeddings `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 + Qdrant/bm25(french)`.

Un passage est jugé pertinent s'il vient d'un document attendu et contient un mot-clé attendu.

## Recherche (k = 5)

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---|---|---|---|
| dense | 48% | 80% | 88% | 0.645 |
| sparse | 60% | 80% | 84% | 0.697 |
| hybrid | 64% | 84% | 92% | 0.743 |

Questions sans passage pertinent dans le top 5 (dense) :
- Où sont stockés les résultats des analyses ?
- Les très petits changements sont-ils bien détectés par les baselines ?
- Quel outil sert à suivre les expériences d'entraînement ?

Questions sans passage pertinent dans le top 5 (sparse) :
- Quels statuts de qualité des données existent et que signifient-ils ?
- Où sont stockés les résultats des analyses ?
- Que doit contenir une model card ?
- Quel outil sert à suivre les expériences d'entraînement ?

Questions sans passage pertinent dans le top 5 (hybrid) :
- Où sont stockés les résultats des analyses ?
- Quel outil sert à suivre les expériences d'entraînement ?

## Abstention (seuil sur le meilleur score cosinus dense)

- Questions avec réponse : score min 0.458, médiane 0.575
- Questions hors corpus : score max 0.442, médiane 0.258
- Seuil retenu : **0.45** (exactitude équilibrée 100% sur ce jeu). Avec seulement 31 questions, ce seuil est choisi et mesuré sur les mêmes données : c'est une estimation optimiste, à confirmer sur des questions nouvelles.
