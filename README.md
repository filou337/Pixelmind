# PIXELMIND

**Multimodal Geospatial Intelligence Platform** : détection de changements de
bâtiments entre deux images satellitaires (T1/T2), avec évaluation reproductible,
puis une couche RAG/agents pour expliquer les résultats.

Architecture de référence : [`docs/PIXELMIND_Dossier_Architecture_Technique.pdf`](docs/PIXELMIND_Dossier_Architecture_Technique.pdf).
Décisions : [`docs/adr/`](docs/adr/).

## Statut

Version **0.1.0** de démonstration ([CHANGELOG](CHANGELOG.md)). Reste à faire : l'entraînement GPU du Sprint 3, sur Kaggle.

| Sprint | Contenu | Statut |
|---|---|---|
| S0 | Repo, conventions, CI, Docker | ✅ |
| S1 | LEVIR-CD, manifeste, EDA, qualité | ✅ |
| S2 | Baselines (différence d'images, gradient boosting) | ✅ |
| S3 | U-Net, Siamese U-Net, MLflow | ⏳ code prêt, entraînement GPU à lancer |
| S4 | API d'analyse, stockage | ✅ |
| S5 | RAG (Qdrant + Claude) | ✅ |
| S6 | Agents (LangGraph) + Streamlit | ✅ |
| S7 | Gouvernance / MLOps | ✅ |

## Démarrage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # sans GPU NVIDIA
pip install -e ".[dev]"
cp .env.example .env

pytest                 # tests
ruff check . && ruff format --check .   # lint
uvicorn pixelmind.api.main:app --reload # API → http://localhost:8000/docs
```

## Données (Sprint 1)

```bash
python -m pixelmind.data.download     # LEVIR-CD (~2,5 Go, MD5 vérifié) -> data/raw/levircd/{train,val,test}
python -m pixelmind.data.manifest     # manifeste + statut qualité -> data/manifests/levircd.jsonl
python -m pixelmind.data.eda          # rapport -> docs/data/levircd_quality_report.md
```

Le manifeste (versionné dans Git, sans images) contient pour chaque paire : chemins,
SHA-256, dimensions, modes, ratio de pixels changés, statut `PASS/WARN/REVIEW/FAIL`
et anomalies. Le split officiel train/val/test est conservé tel quel ; une même
image présente dans deux splits est marquée `FAIL` (fuite de données).

Rapport : [`docs/data/levircd_quality_report.md`](docs/data/levircd_quality_report.md).

## Baselines (Sprint 2)

```bash
python -m pixelmind.experiments.baselines            # tous les modèles
python -m pixelmind.experiments.baselines --models normdiff gbm
python -m pixelmind.experiments.report               # regénère le rapport comparatif
```

Protocole commun à tous les modèles (`pixelmind.evaluation.protocol`) : seuil choisi
sur **val** uniquement, métriques pixel sur **test** (précision, rappel, F1/Dice, IoU,
PR-AUC), par scène, et analyse d'erreurs par quantité de changement. Les métriques sont
cumulées par histogramme de scores et ne demandent donc que très peu de RAM.

Rapport comparatif (baselines + DL) : [`docs/experiments/levircd_benchmark.md`](docs/experiments/levircd_benchmark.md).

## Deep Learning (Sprint 3)

U-Net (T1 et T2 concaténées, 6 canaux) et Siamese U-Net (encodeur partagé, fusion
`|f1 − f2|` ou concaténation à chaque échelle), loss BCE + Dice, AdamW + cosine,
early stopping sur le F1 de validation, suivi MLflow (SQLite).

```bash
# local, CPU : vérifie le pipeline (pas la qualité)
python -m pixelmind.models.dl.train --config configs/smoke_cpu.yaml
# GPU (Kaggle/Colab) : notebooks/03_train_dl_kaggle.ipynb, ou directement
python -m pixelmind.models.dl.train --config configs/siamese_unet.yaml --set epochs=50
python -m pixelmind.experiments.evaluate_checkpoint artifacts/models/siamese_unet_diff_best.pt
mlflow ui --backend-store-uri sqlite:///artifacts/models/mlflow.db
```

Pourquoi entraîner sur Kaggle : voir [ADR 0003](docs/adr/0003-entrainement-dl-sur-gpu-distant.md).

## API d'analyse (Sprint 4)

```bash
docker compose up --build        # ou : uvicorn pixelmind.api.main:app
# documentation interactive : http://localhost:8000/docs
curl -F t1=@avant.png -F t2=@apres.png -F gsd_m=0.5 http://localhost:8000/api/v1/analyses
```

| Route | Usage |
|---|---|
| `GET /api/v1/health` | État du service et nombre de modèles servis |
| `GET /api/v1/models` | Modèles disponibles : version, seuil (choisi sur val), métriques test |
| `POST /api/v1/analyses` | `t1`, `t2` (PNG/JPEG/TIFF RGB), optionnels : `reference_mask`, `model`, `gsd_m` |
| `GET /api/v1/analyses` | Dernières analyses |
| `GET /api/v1/analyses/{id}` | Résultat complet |
| `GET /api/v1/analyses/{id}/artifacts/{nom}` | `mask.png`, `overlay.png`, `probability.png`, `t1.png`, `t2.png`, `reference_mask.png` |

Chaque analyse renvoie :
- le statut qualité (`PASS/WARN/FAIL`) et les anomalies détectées ;
- le modèle utilisé (nom, version = SHA-256 des poids, seuil) ;
- le ratio de pixels changés et les régions détectées ;
- la surface en m², uniquement si `gsd_m` est fourni ;
- les métriques, si un masque de référence est envoyé ;
- les alertes, les limites d'interprétation et le besoin de revue humaine.

Une paire incohérente (tailles différentes, fichier illisible, image trop grande…) est
**rejetée** avec une réponse 422 qui explique la raison. Elle est quand même enregistrée
pour l'audit.

Le serveur sert les modèles présents dans `artifacts/` : baselines (Sprint 2) et
checkpoints évalués (Sprint 3). Choix d'architecture :
[ADR 0004](docs/adr/0004-api-analyse-synchrone-et-stockage.md).

## RAG documentaire (Sprint 5)

```bash
python -m pixelmind.rag.index        # indexe configs/rag_corpus.yaml -> artifacts/rag/ (API arrêtée)
python -m pixelmind.rag.evaluate     # Recall@k / MRR par mode + seuil d'abstention (gratuit)
```

| Route | Usage |
|---|---|
| `POST /api/v1/questions` | `{"question": "...", "mode": "hybrid", "k": 5, "doc_types": ["adr"]}` |
| `POST /api/v1/analyses/{id}/questions` | Question sur une analyse : ses résultats mesurés + la documentation |

La recherche est locale : Qdrant embarqué, embeddings multilingues et BM25, fusion RRF.
La réponse est générée par Claude (`claude-opus-5-5` par défaut, voir `.env.example`)
avec des citations `[n]` vers les passages sources. Sans information pertinente, le
système s'abstient. Chaque question est journalisée dans `artifacts/rag/qa_log.db`.
Pour fonctionner sans LLM : `PIXELMIND_LLM_PROVIDER=extractive`.

Évaluation : [`docs/experiments/rag_eval.md`](docs/experiments/rag_eval.md). Choix
d'architecture : [ADR 0005](docs/adr/0005-rag-documentaire.md).

## Démonstration : agents et interface (Sprint 6)

```bash
docker compose up --build     # API → http://localhost:8000/docs · interface → http://localhost:8501
# ou en local :
uvicorn pixelmind.api.main:app &
streamlit run app/Accueil.py
```

L'interface compte quatre pages : Accueil, Nouvelle analyse, Recherche IA, Gouvernance.
Après chaque analyse, un **workflow LangGraph** :
1. contrôle la qualité ;
2. demande, en option, une explication sourcée au RAG ;
3. s'arrête pour une **revue humaine** si nécessaire (Valider, Valider avec corrections,
   Rejeter, avec des retours par région) ;
4. produit un **rapport Markdown** traçable.

| Route | Usage |
|---|---|
| `POST /api/v1/analyses/{id}/workflows` | `{"explain": true}` : lance le workflow |
| `GET /api/v1/workflows/{id}` | État, raisons de la revue, explication, lien du rapport |
| `POST /api/v1/workflows/{id}/review` | `{"decision": "approved", "reviewer": "…", "comment": "…", "region_feedback": [...]}` |
| `GET /api/v1/analyses/{id}/artifacts/report.md` | Rapport |

Choix d'architecture : [ADR 0006](docs/adr/0006-agents-langgraph-et-interface.md).

## Gouvernance et exploitation (Sprint 7)

- **Clés d'API** : `PIXELMIND_API_KEYS=alice:<clé>,…`, à envoyer dans l'en-tête
  `X-API-Key`. Le nom de la clé est enregistré comme auteur de chaque analyse, question
  et revue.
- `GET /api/v1/monitoring` : qualité des données, sorties des modèles, latences, revues,
  usage et coût du LLM, **dérive des entrées** par rapport à LEVIR-CD train (PSI et test
  χ²). Ces indicateurs s'affichent dans la page Gouvernance.
- `GET /api/v1/analyses/{id}/audit` : historique complet d'un résultat.
- CI : lint, tests, `pip-audit`, construction de l'image, scan Trivy, test de démarrage.
  Elle tourne aussi chaque semaine, et Dependabot propose les mises à jour.
- [Runbook](docs/runbook.md) : démarrage, sauvegarde, publication et retour arrière d'un
  modèle, incidents. Choix d'architecture :
  [ADR 0007](docs/adr/0007-gouvernance-monitoring-derive.md).

## Structure

```
src/pixelmind/   code applicatif (api/, data/, models/, evaluation/, services/, rag/, agents/)
app/             interface Streamlit (client de l'API, sans logique métier)
configs/         configs d'entraînement, corpus RAG et jeu d'évaluation
tests/           tests unitaires et d'intégration
docs/            dossier d'architecture, ADR, model cards
data/, artifacts/  locaux, ignorés par Git
```

## Données et licence

Le dataset MVP est [LEVIR-CD](https://justchenhao.github.io/LEVIR/) (Chen & Shi, 2020),
**réservé à un usage académique, non commercial**. Les images et masques ne sont
jamais versionnés dans ce dépôt. Ce projet est un projet d'apprentissage/portfolio,
pas un produit commercial.
