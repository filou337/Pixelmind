# ADR 0005 – RAG documentaire : recherche locale hybride, génération par Claude avec citations

- Statut : accepté
- Date : 2026-11-27

## Contexte

Le Sprint 5 ajoute la consultation documentaire (dossier §8) : répondre à des questions
sur le projet et sur une analyse, avec des sources vérifiables et une abstention quand
l'information manque. La machine dispose de 5,7 Go de RAM et n'a pas de GPU.

## Décision

- **Corpus en liste blanche** (`configs/rag_corpus.yaml`) : dossier d'architecture, ADR,
  fiche dataset, model cards, rapports, README. Chaque document porte un type et une
  licence. Aucun document externe n'est indexé pour l'instant.
- **Découpage par section** (titres Markdown, titres numérotés du PDF), fenêtres de
  1 200 caractères au plus. Chaque passage a un identifiant stable et garde sa source,
  sa section et sa page.
- **Recherche locale, sans serveur** : Qdrant embarqué, embeddings denses multilingues
  (`paraphrase-multilingual-MiniLM-L12-v2`, ONNX, CPU) et BM25 avec racinisation
  française, fusion RRF. Le mode hybride a été retenu après mesure
  (`docs/experiments/rag_eval.md`) : Recall@5 de 92 % et MRR de 0,743, contre 88 % et
  0,645 en dense seul.
- **Génération : API Claude** (`claude-opus-5-5` par défaut, configurable). Les
  passages sont envoyés en blocs `search_result` avec les citations activées : chaque
  affirmation de la réponse renvoie au passage qui la soutient. Le prompt impose de
  répondre uniquement à partir des sources et de s'abstenir sinon. Si Claude refuse de
  répondre pour raison de sécurité, l'API bascule sur un modèle de repli (`fallbacks`).
- **Abstention en deux temps** : sous un score cosinus de 0,40, le système répond qu'il
  ne sait pas sans appeler le LLM. Au-dessus, c'est Claude qui décide de s'abstenir.
  Le seuil mesuré (0,45) sépare les deux groupes avec une marge très faible ; le seuil
  de 0,40 est volontairement plus bas pour laisser les cas limites au LLM. En mode
  extractif, sans LLM, on applique 0,45.
- **Questions sur une analyse** : les résultats mesurés sont transmis sous forme de
  faits citables (une phrase par fait). Le LLM n'a le droit de donner aucun chiffre,
  aucune surface et aucune localisation qui n'y figure pas.
- **Traçabilité** : chaque question est journalisée dans SQLite
  (`artifacts/rag/qa_log.db`) avec les passages retrouvés et leurs scores, la réponse,
  les citations, le modèle, la version du prompt, les tokens et la version de l'index
  (empreinte SHA-256 du corpus).

## Conséquences

- Pas de service supplémentaire : Qdrant tourne dans le processus de l'API. Un seul
  processus peut ouvrir l'index à la fois, donc il faut reconstruire l'index API arrêtée.
- Le modèle d'embeddings (~220 Mo) est chargé à la première question, pas au démarrage.
- Sans identifiants Claude, l'API répond 503 sur les questions. Le mode extractif
  (`PIXELMIND_LLM_PROVIDER=extractive`) reste disponible.
- La qualité des réponses générées (fidélité, citations valides) se mesure avec
  `python -m pixelmind.rag.evaluate --generate`. Cette commande appelle l'API et coûte
  donc des tokens.
